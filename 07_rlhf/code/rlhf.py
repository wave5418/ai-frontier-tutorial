"""
RLHF 与人类对齐 - 代码实现
========================

本文件包含 RLHF（Reinforcement Learning from Human Feedback）的完整代码实现，
包括 PPO 算法、Reward Model、DPO 等核心组件。

作者：AI Frontier Tutorial
版本：1.0
日期：2026 年 3 月
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import deque


# ============================================================
# 第一部分：PPO 算法实现（简化版）
# ============================================================

@dataclass
class PPOConfig:
    """PPO 算法配置"""
    # 训练参数
    lr: float = 1e-5
    epochs: int = 4
    batch_size: int = 64
    mini_batch_size: int = 8
    
    # PPO 超参数
    clip_epsilon: float = 0.2  # 截断范围
    gamma: float = 0.99  # 折扣因子
    lam: float = 0.95  # GAE 参数
    vf_coef: float = 0.5  # 价值函数系数
    max_grad_norm: float = 0.5  # 梯度裁剪
    
    # KL 约束
    kl_coef: float = 0.1  # KL 惩罚系数
    target_kl: float = 0.02  # 目标 KL 散度
    
    # 其他
    entropy_coef: float = 0.01  # 熵系数（鼓励探索）
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ActorCritic(nn.Module):
    """
    Actor-Critic 网络
    
    同时输出动作概率（策略）和状态价值
    在 RLHF 中，Actor 是语言模型，Critic 是价值头
    """
    
    def __init__(self, model_name: str, pad_token_id: int = 0):
        super().__init__()
        # 加载预训练语言模型作为基础
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.config = self.model.config
        self.pad_token_id = pad_token_id
        
        # 价值头：从隐藏状态预测标量价值
        hidden_size = self.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        
        Returns:
            logits: token 预测 logits
            value: 状态价值估计
            hidden_states: 隐藏状态（用于计算 log_prob）
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # 最后一层隐藏状态
        
        # 使用最后一个非 padding token 的隐藏状态计算价值
        # 找到每个序列的最后一个有效 token
        last_token_idx = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        last_hidden = hidden_states[
            torch.arange(batch_size),
            last_token_idx
        ]
        
        value = self.value_head(last_hidden).squeeze(-1)
        
        return logits, value, hidden_states
    
    def get_log_probs(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        计算采取特定动作的对数概率
        
        Args:
            logits: token logits [batch_size, seq_len, vocab_size]
            actions: 实际采取的动作（token IDs）[batch_size, seq_len]
        
        Returns:
            log_probs: 对数概率 [batch_size, seq_len]
        """
        log_probs = F.log_softmax(logits, dim=-1)
        # 收集实际采取动作的概率
        action_log_probs = log_probs.gather(
            dim=2, 
            index=actions.unsqueeze(-1)
        ).squeeze(-1)
        return action_log_probs
    
    def get_entropy(self, logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        计算策略熵（鼓励探索）
        
        Args:
            logits: token logits
            attention_mask: 注意力掩码
        
        Returns:
            entropy: 平均熵值
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # 计算每个 token 的熵
        entropy = -(probs * log_probs).sum(dim=-1)
        # 只对非 padding token 求平均
        entropy = (entropy * attention_mask).sum() / attention_mask.sum()
        return entropy


class PPOTrainer:
    """
    PPO 训练器
    
    实现近端策略优化算法，用于 RLHF 的强化学习阶段
    """
    
    def __init__(self, actor_critic: ActorCritic, ref_model: AutoModelForCausalLM, 
                 config: PPOConfig):
        self.actor_critic = actor_critic
        self.ref_model = ref_model  # 参考模型（用于 KL 约束）
        self.config = config
        
        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.actor_critic.parameters(),
            lr=config.lr
        )
        
        # 训练统计
        self.stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_div': [],
            'rewards': []
        }
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                    masks: torch.Tensor) -> torch.Tensor:
        """
        计算广义优势估计（GAE）
        
        GAE 在偏差和方差之间取得平衡，是 PPO 的关键组件
        
        Args:
            rewards: 奖励序列 [batch_size, seq_len]
            values: 价值估计 [batch_size, seq_len]
            masks: 有效 token 掩码 [batch_size, seq_len]
        
        Returns:
            advantages: 优势函数估计
        """
        batch_size, seq_len = rewards.shape
        
        # 初始化
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        # 从后向前计算 GAE
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                # 最后一步：TD 残差 = 奖励 - 价值
                delta = rewards[:, t] - values[:, t]
            else:
                # TD 残差 = 奖励 + γ*V(s') - V(s)
                delta = rewards[:, t] + self.config.gamma * values[:, t + 1] * masks[:, t + 1] - values[:, t]
            
            # GAE 递归计算
            last_gae = delta + self.config.gamma * self.config.lam * masks[:, t + 1] * last_gae
            advantages[:, t] = last_gae
        
        # 标准化优势（减少方差）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def compute_kl_penalty(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        计算与参考模型的 KL 散度
        
        KL 约束防止策略偏离太远，保持语言流畅性
        
        Returns:
            kl_div: KL 散度
        """
        with torch.no_grad():
            ref_logits = self.ref_model(input_ids, attention_mask).logits
        
        # 当前策略的 logits
        curr_logits, _, _ = self.actor_critic(input_ids, attention_mask)
        
        # 计算 KL 散度
        ref_probs = F.softmax(ref_logits, dim=-1)
        curr_log_probs = F.log_softmax(curr_logits, dim=-1)
        
        # KL = sum(p * log(p/q))
        kl_div = (ref_probs * (curr_log_probs - torch.log(ref_probs + 1e-8))).sum(dim=-1)
        kl_div = (kl_div * attention_mask).sum() / attention_mask.sum()
        
        return kl_div
    
    def ppo_step(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 actions: torch.Tensor, old_log_probs: torch.Tensor,
                 rewards: torch.Tensor, values: torch.Tensor) -> Dict[str, float]:
        """
        执行一次 PPO 更新
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            actions: 动作序列
            old_log_probs: 旧策略的对数概率
            rewards: 奖励信号
            values: 价值估计
        
        Returns:
            stats: 训练统计信息
        """
        batch_size = input_ids.shape[0]
        
        # 计算 GAE 优势
        advantages = self.compute_gae(rewards, values, attention_mask)
        
        # 多次优化（PPO 的 epochs）
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        
        for epoch in range(self.config.epochs):
            # 小批量训练
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                idx = indices[start:end]
                
                # 获取小批量数据
                mb_input_ids = input_ids[idx]
                mb_attention_mask = attention_mask[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_rewards = rewards[idx]
                
                # 前向传播
                logits, value, _ = self.actor_critic(mb_input_ids, mb_attention_mask)
                
                # 计算当前策略的对数概率
                curr_log_probs = self.actor_critic.get_log_probs(logits, mb_actions)
                curr_log_probs = (curr_log_probs * mb_attention_mask).sum(dim=1) / mb_attention_mask.sum(dim=1)
                
                # 计算重要性采样比率
                ratio = torch.exp(curr_log_probs - mb_old_log_probs)
                
                # PPO 截断目标
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失
                value_loss = F.mse_loss(value, mb_rewards)
                
                # 熵（鼓励探索）
                entropy = self.actor_critic.get_entropy(logits, mb_attention_mask)
                
                # KL 惩罚
                kl_div = self.compute_kl_penalty(mb_input_ids, mb_attention_mask)
                
                # 总损失
                loss = (
                    policy_loss 
                    + self.config.vf_coef * value_loss 
                    - self.config.entropy_coef * entropy
                    + self.config.kl_coef * kl_div
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    self.config.max_grad_norm
                )
                
                self.optimizer.step()
                
                # 累积统计
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl += kl_div.item()
        
        # 计算平均统计
        num_updates = self.config.epochs * (batch_size // self.config.mini_batch_size)
        stats = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'kl_div': total_kl / num_updates,
            'reward': rewards.mean().item()
        }
        
        return stats
    
    def train(self, dataloader: DataLoader, num_epochs: int) -> List[Dict]:
        """
        完整训练循环
        
        Args:
            dataloader: 数据加载器
            num_epochs: 训练轮数
        
        Returns:
            all_stats: 所有 epoch 的统计信息
        """
        all_stats = []
        
        for epoch in range(num_epochs):
            epoch_stats = {
                'policy_loss': [],
                'value_loss': [],
                'entropy': [],
                'kl_div': [],
                'reward': []
            }
            
            for batch in dataloader:
                # 移动数据到设备
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                
                # 执行 PPO 步骤
                stats = self.ppo_step(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    actions=batch['actions'],
                    old_log_probs=batch['old_log_probs'],
                    rewards=batch['rewards'],
                    values=batch['values']
                )
                
                # 记录统计
                for key, value in stats.items():
                    epoch_stats[key].append(value)
            
            # 计算 epoch 平均
            avg_stats = {k: np.mean(v) for k, v in epoch_stats.items()}
            all_stats.append(avg_stats)
            
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Policy Loss: {avg_stats['policy_loss']:.4f}")
            print(f"  Value Loss: {avg_stats['value_loss']:.4f}")
            print(f"  Entropy: {avg_stats['entropy']:.4f}")
            print(f"  KL Div: {avg_stats['kl_div']:.4f}")
            print(f"  Reward: {avg_stats['reward']:.4f}")
        
        return all_stats


# ============================================================
# 第二部分：Reward Model 实现
# ============================================================

class RewardModel(nn.Module):
    """
    奖励模型
    
    输入：提示 + 响应
    输出：标量奖励值（越高表示越好）
    """
    
    def __init__(self, model_name: str, pad_token_id: int = 0):
        super().__init__()
        # 使用预训练模型作为基础
        self.model = AutoModel.from_pretrained(model_name)
        self.config = self.model.config
        self.pad_token_id = pad_token_id
        
        hidden_size = self.config.hidden_size
        
        # 奖励头：多层感知机
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)  # 标量输出
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算奖励
        
        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        
        Returns:
            rewards: 奖励值 [batch_size]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.last_hidden_state
        
        # 使用最后一个非 padding token 的隐藏状态
        last_token_idx = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        last_hidden = hidden_states[
            torch.arange(batch_size),
            last_token_idx
        ]
        
        rewards = self.reward_head(last_hidden).squeeze(-1)
        return rewards


class RewardModelTrainer:
    """
    奖励模型训练器
    
    使用人类偏好数据训练奖励模型
    """
    
    def __init__(self, reward_model: RewardModel, config: PPOConfig):
        self.reward_model = reward_model
        self.config = config
        
        self.optimizer = torch.optim.AdamW(
            self.reward_model.parameters(),
            lr=config.lr
        )
        
        self.stats = {
            'loss': [],
            'accuracy': [],
            'margin': []
        }
    
    def bradley_terry_loss(self, reward_chosen: torch.Tensor, 
                           reward_rejected: torch.Tensor) -> torch.Tensor:
        """
        Bradley-Terry 损失函数
        
        基于偏好概率建模：P(chosen > rejected) = σ(r_chosen - r_rejected)
        
        Args:
            reward_chosen: 被偏好响应的奖励
            reward_rejected: 不被偏好响应的奖励
        
        Returns:
            loss: 偏好损失
        """
        # 计算奖励差异
        logits = reward_chosen - reward_rejected
        
        # 二元交叉熵损失（最大化偏好概率）
        # loss = -log(σ(r_chosen - r_rejected))
        loss = -F.logsigmoid(logits).mean()
        
        return loss
    
    def train_step(self, input_ids_chosen: torch.Tensor, 
                   attention_mask_chosen: torch.Tensor,
                   input_ids_rejected: torch.Tensor,
                   attention_mask_rejected: torch.Tensor) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            input_ids_chosen: 被偏好响应的输入 IDs
            attention_mask_chosen: 被偏好响应的掩码
            input_ids_rejected: 不被偏好响应的输入 IDs
            attention_mask_rejected: 不被偏好响应的掩码
        
        Returns:
            stats: 训练统计
        """
        # 前向传播
        reward_chosen = self.reward_model(input_ids_chosen, attention_mask_chosen)
        reward_rejected = self.reward_model(input_ids_rejected, attention_mask_rejected)
        
        # 计算损失
        loss = self.bradley_terry_loss(reward_chosen, reward_rejected)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.reward_model.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        # 计算统计
        accuracy = (reward_chosen > reward_rejected).float().mean().item()
        margin = (reward_chosen - reward_rejected).mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'margin': margin
        }
    
    def train(self, dataloader: DataLoader, num_epochs: int) -> List[Dict]:
        """
        完整训练循环
        
        Args:
            dataloader: 偏好数据加载器
            num_epochs: 训练轮数
        
        Returns:
            all_stats: 训练统计
        """
        all_stats = []
        
        for epoch in range(num_epochs):
            epoch_stats = {
                'loss': [],
                'accuracy': [],
                'margin': []
            }
            
            for batch in dataloader:
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                
                stats = self.train_step(
                    input_ids_chosen=batch['input_ids_chosen'],
                    attention_mask_chosen=batch['attention_mask_chosen'],
                    input_ids_rejected=batch['input_ids_rejected'],
                    attention_mask_rejected=batch['attention_mask_rejected']
                )
                
                for key, value in stats.items():
                    epoch_stats[key].append(value)
            
            avg_stats = {k: np.mean(v) for k, v in epoch_stats.items()}
            all_stats.append(avg_stats)
            
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Loss: {avg_stats['loss']:.4f}")
            print(f"  Accuracy: {avg_stats['accuracy']:.4f}")
            print(f"  Margin: {avg_stats['margin']:.4f}")
        
        return all_stats


# ============================================================
# 第三部分：DPO 损失函数实现
# ============================================================

class DPOTrainer:
    """
    DPO（Direct Preference Optimization）训练器
    
    直接在偏好数据上优化策略，无需显式奖励模型
    """
    
    def __init__(self, policy_model: AutoModelForCausalLM, 
                 ref_model: AutoModelForCausalLM,
                 config: PPOConfig,
                 beta: float = 0.1):
        """
        Args:
            policy_model: 当前策略模型
            ref_model: 参考模型（冻结）
            config: 配置
            beta: 温度参数，控制偏离参考模型的程度
        """
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.config = config
        self.beta = beta
        
        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.lr
        )
        
        self.stats = {
            'loss': [],
            'accuracy': [],
            'rewards_chosen': [],
            'rewards_rejected': []
        }
    
    def get_batch_log_probs(self, model: AutoModelForCausalLM,
                            input_ids: torch.Tensor,
                            attention_mask: torch.Tensor,
                            labels: torch.Tensor) -> torch.Tensor:
        """
        计算给定序列的对数概率
        
        Args:
            model: 语言模型
            input_ids: 输入 IDs
            attention_mask: 注意力掩码
            labels: 标签（用于计算 log_prob 的 token）
        
        Returns:
            log_probs: 序列的对数概率 [batch_size]
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # 计算每个 token 的对数概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 收集标签 token 的对数概率
        token_log_probs = log_probs.gather(
            dim=2, 
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # 对序列求和（只对非 padding token）
        sequence_log_probs = (token_log_probs * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        
        return sequence_log_probs
    
    def dpo_loss(self, input_ids_chosen: torch.Tensor,
                 attention_mask_chosen: torch.Tensor,
                 labels_chosen: torch.Tensor,
                 input_ids_rejected: torch.Tensor,
                 attention_mask_rejected: torch.Tensor,
                 labels_rejected: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        DPO 损失函数
        
        L_DPO = -log σ(β * (log π(y_w|x) - log π_ref(y_w|x)) 
                      - β * (log π(y_l|x) - log π_ref(y_l|x)))
        
        Args:
            input_ids_chosen: 被偏好响应的输入
            attention_mask_chosen: 被偏好响应的掩码
            labels_chosen: 被偏好响应的标签
            input_ids_rejected: 不被偏好响应的输入
            attention_mask_rejected: 不被偏好响应的掩码
            labels_rejected: 不被偏好响应的标签
        
        Returns:
            loss: DPO 损失
            stats: 统计信息
        """
        # 计算当前策略的对数概率
        policy_log_probs_chosen = self.get_batch_log_probs(
            self.policy_model, input_ids_chosen, attention_mask_chosen, labels_chosen
        )
        policy_log_probs_rejected = self.get_batch_log_probs(
            self.policy_model, input_ids_rejected, attention_mask_rejected, labels_rejected
        )
        
        # 计算参考策略的对数概率（no grad）
        with torch.no_grad():
            ref_log_probs_chosen = self.get_batch_log_probs(
                self.ref_model, input_ids_chosen, attention_mask_chosen, labels_chosen
            )
            ref_log_probs_rejected = self.get_batch_log_probs(
                self.ref_model, input_ids_rejected, attention_mask_rejected, labels_rejected
            )
        
        # 计算隐式奖励（相对于参考模型的偏差）
        implicit_reward_chosen = self.beta * (policy_log_probs_chosen - ref_log_probs_chosen)
        implicit_reward_rejected = self.beta * (policy_log_probs_rejected - ref_log_probs_rejected)
        
        # DPO 损失
        logits = implicit_reward_chosen - implicit_reward_rejected
        loss = -F.logsigmoid(logits).mean()
        
        # 统计信息
        stats = {
            'accuracy': (logits > 0).float().mean().item(),
            'rewards_chosen': implicit_reward_chosen.mean().item(),
            'rewards_rejected': implicit_reward_rejected.mean().item(),
            'margin': (implicit_reward_chosen - implicit_reward_rejected).mean().item()
        }
        
        return loss, stats
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        单步训练
        
        Returns:
            stats: 训练统计
        """
        batch = {k: v.to(self.config.device) for k, v in batch.items()}
        
        # 计算 DPO 损失
        loss, stats = self.dpo_loss(
            input_ids_chosen=batch['input_ids_chosen'],
            attention_mask_chosen=batch['attention_mask_chosen'],
            labels_chosen=batch['labels_chosen'],
            input_ids_rejected=batch['input_ids_rejected'],
            attention_mask_rejected=batch['attention_mask_rejected'],
            labels_rejected=batch['labels_rejected']
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        stats['loss'] = loss.item()
        return stats
    
    def train(self, dataloader: DataLoader, num_epochs: int) -> List[Dict]:
        """完整训练循环"""
        all_stats = []
        
        for epoch in range(num_epochs):
            epoch_stats = {
                'loss': [],
                'accuracy': [],
                'rewards_chosen': [],
                'rewards_rejected': [],
                'margin': []
            }
            
            for batch in dataloader:
                stats = self.train_step(batch)
                for key, value in stats.items():
                    epoch_stats[key].append(value)
            
            avg_stats = {k: np.mean(v) for k, v in epoch_stats.items()}
            all_stats.append(avg_stats)
            
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Loss: {avg_stats['loss']:.4f}")
            print(f"  Accuracy: {avg_stats['accuracy']:.4f}")
            print(f"  Margin: {avg_stats['margin']:.4f}")
        
        return all_stats


# ============================================================
# 第四部分：IPO 和 KTO 实现
# ============================================================

class IPOTrainer:
    """
    IPO（Identity Preference Optimization）训练器
    
    使用平方损失代替 sigmoid，更稳定
    """
    
    def __init__(self, policy_model: AutoModelForCausalLM,
                 ref_model: AutoModelForCausalLM,
                 config: PPOConfig,
                 beta: float = 0.1,
                 tau: float = 0.5):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.config = config
        self.beta = beta
        self.tau = tau  # 目标边际
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.lr
        )
    
    def ipo_loss(self, batch: Dict) -> torch.Tensor:
        """
        IPO 损失函数
        
        L_IPO = (log(π(y_w)/π_ref(y_w)) - log(π(y_l)/π_ref(y_l)) - τ)²
        """
        # 复用 DPO 的对数概率计算逻辑
        dpo_trainer = DPOTrainer(self.policy_model, self.ref_model, self.config, self.beta)
        
        policy_log_probs_chosen = dpo_trainer.get_batch_log_probs(
            self.policy_model, 
            batch['input_ids_chosen'],
            batch['attention_mask_chosen'],
            batch['labels_chosen']
        )
        policy_log_probs_rejected = dpo_trainer.get_batch_log_probs(
            self.policy_model,
            batch['input_ids_rejected'],
            batch['attention_mask_rejected'],
            batch['labels_rejected']
        )
        
        with torch.no_grad():
            ref_log_probs_chosen = dpo_trainer.get_batch_log_probs(
                self.ref_model,
                batch['input_ids_chosen'],
                batch['attention_mask_chosen'],
                batch['labels_chosen']
            )
            ref_log_probs_rejected = dpo_trainer.get_batch_log_probs(
                self.ref_model,
                batch['input_ids_rejected'],
                batch['attention_mask_rejected'],
                batch['labels_rejected']
            )
        
        # 计算边际
        margin = (policy_log_probs_chosen - ref_log_probs_chosen) - \
                 (policy_log_probs_rejected - ref_log_probs_rejected)
        
        # 平方损失
        loss = ((margin - self.tau) ** 2).mean()
        
        return loss
    
    def train_step(self, batch: Dict) -> float:
        """单步训练"""
        batch = {k: v.to(self.config.device) for k, v in batch.items()}
        
        loss = self.ipo_loss(batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        return loss.item()


class KTOTrainer:
    """
    KTO（Kahneman-Tversky Optimization）训练器
    
    基于前景理论，不需要成对偏好，只需好坏标签
    """
    
    def __init__(self, policy_model: AutoModelForCausalLM,
                 ref_model: AutoModelForCausalLM,
                 config: PPOConfig,
                 beta: float = 0.1,
                 delta_desirable: float = 0.0,
                 delta_undesirable: float = 0.0,
                 lambda_desirable: float = 1.0,
                 lambda_undesirable: float = 1.0):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.config = config
        self.beta = beta
        
        # 前景理论参数
        self.delta_desirable = delta_desirable  # 期望响应的参考点
        self.delta_undesirable = delta_undesirable  # 不期望响应的参考点
        self.lambda_desirable = lambda_desirable  # 期望响应的权重
        self.lambda_undesirable = lambda_undesirable  # 不期望响应的权重
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.lr
        )
    
    def kto_loss(self, input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 labels: torch.Tensor,
                 is_desirable: torch.Tensor) -> torch.Tensor:
        """
        KTO 损失函数
        
        基于前景理论：
        - 对期望响应：最大化超过参考点的程度
        - 对不期望响应：最小化超过参考点的程度
        
        Args:
            is_desirable: 布尔张量，表示响应是否期望
        """
        # 计算当前策略和参考策略的对数概率
        dpo_trainer = DPOTrainer(self.policy_model, self.ref_model, self.config, self.beta)
        
        policy_log_probs = dpo_trainer.get_batch_log_probs(
            self.policy_model, input_ids, attention_mask, labels
        )
        
        with torch.no_grad():
            ref_log_probs = dpo_trainer.get_batch_log_probs(
                self.ref_model, input_ids, attention_mask, labels
            )
        
        # 隐式奖励
        implicit_reward = self.beta * (policy_log_probs - ref_log_probs)
        
        # 分别处理期望和不期望的响应
        desirable_mask = is_desirable.float()
        undesirable_mask = 1 - desirable_mask
        
        # 期望响应的损失：希望奖励超过参考点
        loss_desirable = self.lambda_desirable * (1 - F.sigmoid(implicit_reward - self.delta_desirable))
        loss_desirable = (loss_desirable * desirable_mask).sum() / (desirable_mask.sum() + 1e-8)
        
        # 不期望响应的损失：希望奖励低于参考点
        loss_undesirable = self.lambda_undesirable * (1 - F.sigmoid(self.delta_undesirable - implicit_reward))
        loss_undesirable = (loss_undesirable * undesirable_mask).sum() / (undesirable_mask.sum() + 1e-8)
        
        return loss_desirable + loss_undesirable
    
    def train_step(self, batch: Dict) -> float:
        """单步训练"""
        batch = {k: v.to(self.config.device) for k, v in batch.items()}
        
        loss = self.kto_loss(
            batch['input_ids'],
            batch['attention_mask'],
            batch['labels'],
            batch['is_desirable']
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        return loss.item()


# ============================================================
# 第五部分：完整 RLHF 训练流程示例
# ============================================================

class RLHFPipeline:
    """
    完整的 RLHF 训练流程
    
    整合 SFT、Reward Model 训练、PPO 优化三个阶段
    """
    
    def __init__(self, model_name: str, config: PPOConfig):
        self.model_name = model_name
        self.config = config
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 初始化模型
        self.sft_model = None
        self.reward_model = None
        self.policy_model = None
        self.ref_model = None
    
    def prepare_sft_data(self, prompts: List[str], responses: List[str]) -> Dataset:
        """准备 SFT 数据"""
        
        class SFTDataset(Dataset):
            def __init__(self, prompts, responses, tokenizer, max_length=512):
                self.prompts = prompts
                self.responses = responses
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.prompts)
            
            def __getitem__(self, idx):
                # 拼接提示和响应
                text = f"{self.prompts[idx]} {self.responses[idx]}"
                
                # 分词
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].squeeze(0)
                attention_mask = encoding['attention_mask'].squeeze(0)
                
                # 标签（用于计算损失）
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100  # 忽略 padding
                
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }
        
        return SFTDataset(prompts, responses, self.tokenizer)
    
    def prepare_preference_data(self, prompts: List[str], 
                                chosen_responses: List[str],
                                rejected_responses: List[str]) -> Dataset:
        """准备偏好数据"""
        
        class PreferenceDataset(Dataset):
            def __init__(self, prompts, chosen, rejected, tokenizer, max_length=512):
                self.prompts = prompts
                self.chosen = chosen
                self.rejected = rejected
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.prompts)
            
            def __getitem__(self, idx):
                # 分别处理 chosen 和 rejected
                text_chosen = f"{self.prompts[idx]} {self.chosen[idx]}"
                text_rejected = f"{self.prompts[idx]} {self.rejected[idx]}"
                
                encoding_chosen = self.tokenizer(
                    text_chosen,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                encoding_rejected = self.tokenizer(
                    text_rejected,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids_chosen = encoding_chosen['input_ids'].squeeze(0)
                attention_mask_chosen = encoding_chosen['attention_mask'].squeeze(0)
                labels_chosen = input_ids_chosen.clone()
                
                input_ids_rejected = encoding_rejected['input_ids'].squeeze(0)
                attention_mask_rejected = encoding_rejected['attention_mask'].squeeze(0)
                labels_rejected = input_ids_rejected.clone()
                
                return {
                    'input_ids_chosen': input_ids_chosen,
                    'attention_mask_chosen': attention_mask_chosen,
                    'labels_chosen': labels_chosen,
                    'input_ids_rejected': input_ids_rejected,
                    'attention_mask_rejected': attention_mask_rejected,
                    'labels_rejected': labels_rejected
                }
        
        return PreferenceDataset(prompts, chosen_responses, rejected_responses, self.tokenizer)
    
    def stage1_sft(self, sft_dataloader: DataLoader, num_epochs: int = 3):
        """
        第一阶段：监督微调（SFT）
        """
        print("=" * 60)
        print("阶段 1: 监督微调 (SFT)")
        print("=" * 60)
        
        # 加载模型
        self.sft_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.sft_model.to(self.config.device)
        
        optimizer = torch.optim.AdamW(
            self.sft_model.parameters(),
            lr=self.config.lr
        )
        
        # 训练循环
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch in sft_dataloader:
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                
                outputs = self.sft_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(sft_dataloader)
            print(f"SFT Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # 保存 SFT 模型作为参考模型
        self.ref_model = self.sft_model
    
    def stage2_reward_model(self, preference_dataloader: DataLoader, 
                            num_epochs: int = 3):
        """
        第二阶段：奖励模型训练
        """
        print("\n" + "=" * 60)
        print("阶段 2: 奖励模型训练")
        print("=" * 60)
        
        # 初始化奖励模型
        self.reward_model = RewardModel(self.model_name)
        self.reward_model.to(self.config.device)
        
        # 训练
        trainer = RewardModelTrainer(self.reward_model, self.config)
        trainer.train(preference_dataloader, num_epochs)
    
    def stage3_ppo(self, preference_dataloader: DataLoader, 
                   num_epochs: int = 3):
        """
        第三阶段：PPO 强化学习优化
        """
        print("\n" + "=" * 60)
        print("阶段 3: PPO 强化学习优化")
        print("=" * 60)
        
        # 初始化策略模型（从 SFT 模型开始）
        self.policy_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.policy_model.to(self.config.device)
        
        # 包装为 Actor-Critic
        actor_critic = ActorCritic(self.model_name)
        actor_critic.to(self.config.device)
        
        # 复制 SFT 权重
        actor_critic.model.load_state_dict(self.sft_model.state_dict())
        
        # 初始化 PPO 训练器
        ppo_trainer = PPOTrainer(actor_critic, self.ref_model, self.config)
        
        # 训练（注意：实际使用中需要生成响应并计算奖励）
        # 这里简化处理，实际实现需要更复杂的采样和奖励计算
        print("PPO 训练需要采样轨迹并计算奖励，此处为简化示例")
        print("完整实现请参考 PPOTrainer 类")
    
    def run_full_pipeline(self, sft_data: Tuple, preference_data: Tuple):
        """
        运行完整的 RLHF 流程
        
        Args:
            sft_data: (prompts, responses) 用于 SFT
            preference_data: (prompts, chosen, rejected) 用于偏好学习
        """
        # 准备数据
        sft_dataset = self.prepare_sft_data(*sft_data)
        sft_dataloader = DataLoader(
            sft_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        pref_dataset = self.prepare_preference_data(*preference_data)
        pref_dataloader = DataLoader(
            pref_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # 执行三个阶段
        self.stage1_sft(sft_dataloader)
        self.stage2_reward_model(pref_dataloader)
        self.stage3_ppo(pref_dataloader)
        
        print("\n" + "=" * 60)
        print("RLHF 训练完成!")
        print("=" * 60)


# ============================================================
# 第六部分：使用示例
# ============================================================

def example_usage():
    """
    使用示例
    """
    print("RLHF 代码实现示例")
    print("=" * 60)
    
    # 配置
    config = PPOConfig(
        lr=1e-5,
        batch_size=8,
        epochs=2
    )
    
    # 示例数据
    sft_prompts = [
        "请解释什么是机器学习？",
        "如何学习 Python 编程？",
        "量子力学的基本原理是什么？"
    ]
    
    sft_responses = [
        "机器学习是人工智能的一个分支，它使计算机能够从数据中学习...",
        "学习 Python 的好方法包括：1. 学习基础语法 2. 做项目练习...",
        "量子力学是描述微观粒子行为的物理学理论，核心原理包括..."
    ]
    
    pref_prompts = [
        "请解释什么是机器学习？",
        "如何学习 Python 编程？"
    ]
    
    chosen_responses = [
        "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并改进...",
        "学习 Python 的好方法包括：1. 学习基础语法 2. 做项目练习 3. 阅读优秀代码..."
    ]
    
    rejected_responses = [
        "机器学习就是让机器变聪明。",
        "学 Python 就是多写代码。"
    ]
    
    print("\n示例数据已准备")
    print(f"SFT 样本数：{len(sft_prompts)}")
    print(f"偏好样本数：{len(pref_prompts)}")
    
    # 注意：实际运行需要：
    # 1. 安装依赖：torch, transformers
    # 2. 有可用的 GPU（推荐）
    # 3. 足够的内存
    
    print("\n运行完整 RLHF 流程:")
    print("pipeline = RLHFPipeline('gpt2', config)")
    print("pipeline.run_full_pipeline(")
    print("    sft_data=(sft_prompts, sft_responses),")
    print("    preference_data=(pref_prompts, chosen_responses, rejected_responses)")
    print(")")
    
    # DPO 示例
    print("\n" + "=" * 60)
    print("DPO 训练示例:")
    print("=" * 60)
    print("""
# 初始化模型
policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
ref_model = AutoModelForCausalLM.from_pretrained('gpt2')

# 初始化 DPO 训练器
dpo_trainer = DPOTrainer(policy_model, ref_model, config, beta=0.1)

# 训练
dpo_trainer.train(preference_dataloader, num_epochs=3)
    """)
    
    print("\n代码实现完成！")
    print("详细说明请参考 theory.md 文档")


if __name__ == "__main__":
    example_usage()
