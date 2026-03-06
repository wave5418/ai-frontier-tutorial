"""
第 26 章：RLHF 实战 - 完整代码实现
====================================

本模块实现了 RLHF（Reinforcement Learning from Human Feedback）的完整流程，包括：
- SFT（Supervised Fine-Tuning）训练
- Reward Model 训练
- PPO（Proximal Policy Optimization）微调
- 端到端示例

作者：AI 前沿技术教程
日期：2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import warnings
import os


# ============================================================
# 第一部分：SFT（Supervised Fine-Tuning）
# ============================================================

@dataclass
class SFTSample:
    """SFT 训练样本"""
    instruction: str    # 指令
    input: str = ""     # 额外输入（可选）
    output: str = ""    # 期望输出
    history: List[Tuple[str, str]] = field(default_factory=list)  # 对话历史


class SFTDataset(Dataset):
    """SFT 数据集"""
    
    def __init__(
        self,
        samples: List[SFTSample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        prompt_template: str = None
    ):
        """
        初始化 SFT 数据集
        
        Args:
            samples: SFT 样本列表
            tokenizer: 分词器
            max_length: 最大序列长度
            prompt_template: 提示模板（可选）
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or "{instruction}\n{input}\n{output}"
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 构建完整文本
        if sample.history:
            # 多轮对话
            history_text = "\n".join([
                f"Human: {h[0]}\nAssistant: {h[1]}" 
                for h in sample.history
            ])
            full_text = f"{history_text}\nHuman: {sample.instruction}\nAssistant: {sample.output}"
        else:
            # 单轮对话
            full_text = f"Human: {sample.instruction}\nAssistant: {sample.output}"
        
        # 分词
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 准备 labels（将 prompt 部分的 label 设为 -100，不计算损失）
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # 找到 prompt 和 response 的分界点
        prompt_text = f"Human: {sample.instruction}\nAssistant: "
        prompt_encoded = self.tokenizer(prompt_text, return_tensors='pt')
        prompt_length = prompt_encoded['input_ids'].shape[1]
        
        # 创建 labels：prompt 部分为 -100，response 部分为实际 token
        labels = input_ids.clone()
        labels[:prompt_length] = -100  # 忽略 prompt 部分的损失
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class SFTTrainer:
    """
    SFT 训练器
    
    封装了 Supervised Fine-Tuning 的完整流程
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        max_length: int = 512,
        gradient_accumulation_steps: int = 1,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        output_dir: str = "./sft_output",
        device: Optional[str] = None
    ):
        """
        初始化 SFT 训练器
        
        Args:
            model: 预训练模型
            tokenizer: 分词器
            learning_rate: 学习率
            batch_size: 批大小
            max_length: 最大序列长度
            gradient_accumulation_steps: 梯度累积步数
            num_epochs: 训练轮数
            warmup_ratio: 学习率预热比例
            output_dir: 输出目录
            device: 训练设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.output_dir = output_dir
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 训练状态
        self.global_step = 0
        self.training_history = []
    
    def train(
        self,
        train_dataset: SFTDataset,
        eval_dataset: Optional[SFTDataset] = None
    ) -> Dict[str, List[float]]:
        """
        执行 SFT 训练
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 验证数据集（可选）
            
        Returns:
            训练历史
        """
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # 计算总步数和预热步数
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        print(f"开始 SFT 训练")
        print(f"设备：{self.device}")
        print(f"训练样本数：{len(train_dataset)}")
        print(f"学习率：{self.learning_rate}")
        print("-" * 50)
        
        training_history = {'loss': [], 'eval_loss': []}
        
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loader):
                # 将数据移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
                
                # 反向传播
                (loss / self.gradient_accumulation_steps).backward()
                
                # 梯度累积更新
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # 日志
                if self.global_step % 10 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"Step {self.global_step}: Loss={avg_loss:.4f}")
                    training_history['loss'].append(avg_loss)
            
            # Epoch 结束
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            print(f"\nEpoch {epoch + 1}/{self.num_epochs} 完成，平均损失：{avg_epoch_loss:.4f}")
            
            # 验证
            if eval_dataset is not None:
                eval_loss = self.evaluate(eval_dataset)
                training_history['eval_loss'].append(eval_loss)
                print(f"验证损失：{eval_loss:.4f}")
            
            # 保存检查点
            self.save_checkpoint(f"epoch-{epoch + 1}")
        
        # 保存最终模型
        self.save_checkpoint("final")
        
        self.training_history = training_history
        return training_history
    
    def _collate_fn(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """数据整理函数"""
        return {
            'input_ids': torch.stack([s['input_ids'] for s in samples]),
            'attention_mask': torch.stack([s['attention_mask'] for s in samples]),
            'labels': torch.stack([s['labels'] for s in samples])
        }
    
    def evaluate(self, eval_dataset: SFTDataset) -> float:
        """评估"""
        self.model.eval()
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                total_loss += outputs.loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_path = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_path, exist_ok=True)
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        print(f"检查点已保存到：{checkpoint_path}")
    
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """生成文本"""
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================================
# 第二部分：Reward Model 训练
# ============================================================

@dataclass
class PreferenceSample:
    """偏好样本（用于 Reward Model 训练）"""
    prompt: str           # 输入提示
    chosen: str          # 偏好回复
    rejected: str        # 拒绝回复


class RewardModelDataset(Dataset):
    """Reward Model 数据集"""
    
    def __init__(
        self,
        samples: List[PreferenceSample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 分别编码 chosen 和 rejected
        chosen_text = f"{sample.prompt} {sample.chosen}"
        rejected_text = f"{sample.prompt} {sample.rejected}"
        
        chosen_encoded = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        rejected_encoded = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_encoded['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_encoded['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_encoded['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_encoded['attention_mask'].squeeze(0)
        }


class RewardModel(nn.Module):
    """
    Reward Model
    
    基于预训练语言模型，输出单个标量奖励分数
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        pooling: str = "mean"  # "mean", "last", "cls"
    ):
        """
        初始化 Reward Model
        
        Args:
            base_model: 基座模型
            pooling: 池化方式
        """
        super().__init__()
        self.base_model = base_model
        self.pooling = pooling
        
        # 获取隐藏层维度
        hidden_size = base_model.config.hidden_size
        
        # 奖励头：将隐藏状态映射到标量
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            奖励分数 [batch_size]
        """
        # 获取模型输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 获取最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # 池化
        if self.pooling == "mean":
            # 平均池化
            mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.pooling == "last":
            # 最后一个非 padding token
            lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(len(lengths), device=lengths.device)
            pooled = hidden_states[batch_indices, lengths]
        elif self.pooling == "cls":
            # CLS token（如果有的话）
            pooled = hidden_states[:, 0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # 计算奖励分数
        reward = self.reward_head(pooled).squeeze(-1)  # [batch_size]
        
        return reward


class RewardModelTrainer:
    """Reward Model 训练器"""
    
    def __init__(
        self,
        reward_model: RewardModel,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float = 1e-5,
        batch_size: int = 16,
        max_length: int = 512,
        num_epochs: int = 2,
        output_dir: str = "./reward_model_output",
        device: Optional[str] = None
    ):
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.reward_model.to(self.device)
        
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.reward_model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        os.makedirs(output_dir, exist_ok=True)
    
    def train(
        self,
        train_dataset: RewardModelDataset,
        eval_dataset: Optional[RewardModelDataset] = None
    ) -> Dict[str, List[float]]:
        """
        训练 Reward Model
        
        使用 pairwise 损失：最大化 P(chosen > rejected)
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        print(f"开始 Reward Model 训练")
        print(f"设备：{self.device}")
        print(f"训练样本数：{len(train_dataset)}")
        print("-" * 50)
        
        training_history = {'loss': [], 'accuracy': []}
        
        for epoch in range(self.num_epochs):
            self.reward_model.train()
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 计算 chosen 和 rejected 的奖励分数
                chosen_rewards = self.reward_model(
                    batch['chosen_input_ids'],
                    batch['chosen_attention_mask']
                )
                rejected_rewards = self.reward_model(
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask']
                )
                
                # Pairwise 损失：-log(σ(chosen - rejected))
                logits = chosen_rewards - rejected_rewards
                losses = -F.logsigmoid(logits)
                loss = losses.mean()
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 计算准确率
                accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy
                num_batches += 1
            
            # Epoch 结束
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_accuracy = epoch_accuracy / max(num_batches, 1)
            print(f"Epoch {epoch + 1}/{self.num_epochs}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
            training_history['loss'].append(avg_loss)
            training_history['accuracy'].append(avg_accuracy)
        
        # 保存模型
        self.save_checkpoint("final")
        
        return training_history
    
    def _collate_fn(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            'chosen_input_ids': torch.stack([s['chosen_input_ids'] for s in samples]),
            'chosen_attention_mask': torch.stack([s['chosen_attention_mask'] for s in samples]),
            'rejected_input_ids': torch.stack([s['rejected_input_ids'] for s in samples]),
            'rejected_attention_mask': torch.stack([s['rejected_attention_mask'] for s in samples])
        }
    
    def save_checkpoint(self, name: str):
        checkpoint_path = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_path, exist_ok=True)
        self.reward_model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        print(f"Reward Model 已保存到：{checkpoint_path}")
    
    def predict(self, text: str) -> float:
        """预测单个文本的奖励分数"""
        self.reward_model.eval()
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            reward = self.reward_model(
                encoded['input_ids'],
                encoded['attention_mask']
            )
        
        return reward.item()


# ============================================================
# 第三部分：PPO 微调
# ============================================================

class PPOBuffer:
    """PPO 经验回放缓冲区"""
    
    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.reset()
    
    def reset(self):
        """清空缓冲区"""
        self.input_ids = []
        self.attention_masks = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.advantages = []
    
    def add(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor
    ):
        """添加经验"""
        self.input_ids.append(input_ids)
        self.attention_masks.append(attention_mask)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
    
    def compute_advantages(self, gamma: float = 0.99, lam: float = 0.95):
        """
        使用 GAE（Generalized Advantage Estimation）计算优势函数
        
        Args:
            gamma: 折扣因子
            lam: GAE 参数
        """
        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values)
        
        # 计算 TD 误差
        deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
        
        # 计算优势函数
        advantages = []
        adv = 0
        for delta in reversed(deltas):
            adv = delta + gamma * lam * adv
            advantages.insert(0, adv)
        advantages = torch.stack(advantages)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages
    
    def get_batch(self, batch_size: int):
        """获取批处理数据"""
        indices = torch.randperm(len(self.input_ids))
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            yield {
                'input_ids': torch.stack([self.input_ids[j] for j in batch_indices]),
                'attention_mask': torch.stack([self.attention_masks[j] for j in batch_indices]),
                'log_probs': torch.stack([self.log_probs[j] for j in batch_indices]),
                'rewards': torch.stack([self.rewards[j] for j in batch_indices]),
                'advantages': torch.stack([self.advantages[j] for j in batch_indices])
            }


class PPOTrainer:
    """
    PPO 训练器
    
    实现 Proximal Policy Optimization 算法用于 RLHF
    """
    
    def __init__(
        self,
        policy_model: PreTrainedModel,
        ref_model: PreTrainedModel,
        reward_model: RewardModel,
        value_model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float = 1e-6,
        batch_size: int = 128,
        mini_batch_size: int = 32,
        ppo_epochs: int = 4,
        clip_range: float = 0.2,
        kl_coef: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_coef: float = 0.1,
        max_length: int = 512,
        output_dir: str = "./ppo_output",
        device: Optional[str] = None
    ):
        """
        初始化 PPO 训练器
        
        Args:
            policy_model: 策略模型（将被优化）
            ref_model: 参考模型（冻结，用于 KL 约束）
            reward_model: 奖励模型
            value_model: 价值模型
            tokenizer: 分词器
            learning_rate: 学习率
            batch_size: 批大小
            mini_batch_size: PPO 小批大小
            ppo_epochs: 每批数据的 PPO 更新轮数
            clip_range: PPO clip 范围
            kl_coef: KL 约束系数
            gamma: 折扣因子
            lam: GAE 参数
            value_coef: 价值损失系数
            max_length: 最大序列长度
            output_dir: 输出目录
            device: 训练设备
        """
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.lam = lam
        self.value_coef = value_coef
        self.max_length = max_length
        self.output_dir = output_dir
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.policy_model.to(self.device)
        self.ref_model.to(self.device)
        self.reward_model.to(self.device)
        self.value_model.to(self.device)
        
        # 冻结参考模型和奖励模型
        self.ref_model.eval()
        self.reward_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            list(self.policy_model.parameters()) + list(self.value_model.parameters()),
            lr=learning_rate
        )
        
        # 经验缓冲区
        self.buffer = PPOBuffer(capacity=batch_size)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_with_logprobs(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成回复并计算对数概率
        
        Returns:
            generated_ids: 生成的 token IDs
            log_probs: 生成 token 的对数概率
        """
        self.policy_model.eval()
        
        with torch.no_grad():
            outputs = self.policy_model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = outputs.sequences
        
        # 计算对数概率
        scores = outputs.scores  # List of [batch_size, vocab_size]
        log_probs = []
        
        for i, score in enumerate(scores):
            log_prob = F.log_softmax(score, dim=-1)
            # 获取实际生成的 token
            gen_token = generated_ids[:, prompt_ids.shape[1] + i]
            token_log_prob = log_probs.gather(1, gen_token.unsqueeze(-1)).squeeze(-1)
            log_probs.append(token_log_prob)
        
        log_probs = torch.stack(log_probs, dim=1)  # [batch_size, num_new_tokens]
        
        return generated_ids, log_probs
    
    def compute_rewards(
        self,
        generated_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompts: List[str]
    ) -> torch.Tensor:
        """
        计算生成回复的奖励分数
        
        奖励 = Reward Model 输出 - KL 散度惩罚
        """
        # 解码生成的文本
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 用 Reward Model 计算奖励
        rewards = []
        for text in generated_texts:
            reward = self.reward_model.predict(text)
            rewards.append(reward)
        rewards = torch.tensor(rewards, device=self.device)
        
        # 计算 KL 散度惩罚
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            ref_log_probs = F.log_softmax(ref_outputs.logits, dim=-1)
            
            policy_outputs = self.policy_model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            policy_log_probs = F.log_softmax(policy_outputs.logits, dim=-1)
            
            # KL 散度
            kl = (policy_log_probs - ref_log_probs) * attention_mask.unsqueeze(-1)
            kl = kl.sum(dim=(1, 2)) / attention_mask.sum(dim=1)
        
        # 带 KL 惩罚的奖励
        final_rewards = rewards - self.kl_coef * kl
        
        return final_rewards
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict]:
        """
        执行单个 PPO 训练步骤
        
        Returns:
            loss: 总损失
            metrics: 指标字典
        """
        self.policy_model.train()
        self.value_model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        old_log_probs = batch['log_probs'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        
        # 计算当前策略的对数概率
        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        
        # 简化：使用平均对数概率
        current_log_probs = log_probs.mean(dim=(1, 2))
        
        # 计算概率比率
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # PPO 损失（clipped surrogate objective）
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        values = self.value_model(input_ids, attention_mask)
        value_loss = F.mse_loss(values.squeeze(), rewards)
        
        # KL 散度惩罚
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_log_probs = F.log_softmax(ref_outputs.logits, dim=-1)
            kl = (log_probs - ref_log_probs).mean()
        kl_penalty = kl.mean()
        
        # 总损失
        loss = policy_loss + self.value_coef * value_loss + self.kl_coef * kl_penalty
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl': kl_penalty.item(),
            'reward': rewards.mean().item()
        }
        
        return loss.item(), metrics
    
    def train(
        self,
        prompts: List[str],
        num_episodes: int = 100,
        max_new_tokens: int = 128
    ) -> Dict[str, List[float]]:
        """
        执行 PPO 训练
        
        Args:
            prompts: 提示列表
            num_episodes: 训练轮数
            max_new_tokens: 最大生成 token 数
            
        Returns:
            训练历史
        """
        print(f"开始 PPO 训练")
        print(f"设备：{self.device}")
        print(f"Prompt 数量：{len(prompts)}")
        print(f"训练轮数：{num_episodes}")
        print("-" * 50)
        
        training_history = {
            'loss': [],
            'reward': [],
            'kl': []
        }
        
        for episode in range(num_episodes):
            # 1. Rollout 阶段：收集经验
            self.buffer.reset()
            
            # 采样一批 prompt
            batch_prompts = np.random.choice(prompts, size=min(self.batch_size, len(prompts)), replace=False)
            
            for prompt in batch_prompts:
                # 编码 prompt
                encoded = self.tokenizer(prompt, return_tensors='pt', max_length=self.max_length, truncation=True)
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # 生成回复
                generated_ids, log_probs = self.generate_with_logprobs(
                    input_ids, attention_mask, max_new_tokens
                )
                
                # 计算奖励
                rewards = self.compute_rewards(generated_ids, attention_mask, [prompt])
                
                # 计算价值
                with torch.no_grad():
                    values = self.value_model(generated_ids, attention_mask)
                
                # 添加到缓冲区
                self.buffer.add(
                    generated_ids[0],
                    attention_mask[0],
                    log_probs[0].mean(),
                    rewards[0],
                    values[0].mean()
                )
            
            # 2. 计算优势函数
            self.buffer.compute_advantages(gamma=self.gamma, lam=self.lam)
            
            # 3. PPO 更新阶段
            episode_loss = 0.0
            episode_reward = 0.0
            episode_kl = 0.0
            num_updates = 0
            
            for ppo_epoch in range(self.ppo_epochs):
                for batch in self.buffer.get_batch(self.mini_batch_size):
                    loss, metrics = self.train_step(batch)
                    episode_loss += loss
                    episode_reward += metrics['reward']
                    episode_kl += metrics['kl']
                    num_updates += 1
            
            # 记录指标
            avg_loss = episode_loss / max(num_updates, 1)
            avg_reward = episode_reward / max(num_updates, 1)
            avg_kl = episode_kl / max(num_updates, 1)
            
            training_history['loss'].append(avg_loss)
            training_history['reward'].append(avg_reward)
            training_history['kl'].append(avg_kl)
            
            # 日志
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}: "
                      f"Loss={avg_loss:.4f}, Reward={avg_reward:.4f}, KL={avg_kl:.4f}")
            
            # 保存检查点
            if (episode + 1) % 50 == 0:
                self.save_checkpoint(f"episode-{episode + 1}")
        
        # 保存最终模型
        self.save_checkpoint("final")
        
        return training_history
    
    def save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_path = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        self.policy_model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # 保存价值模型
        value_path = os.path.join(checkpoint_path, "value_model")
        os.makedirs(value_path, exist_ok=True)
        torch.save(self.value_model.state_dict(), os.path.join(value_path, "pytorch_model.bin"))
        
        print(f"检查点已保存到：{checkpoint_path}")


# ============================================================
# 第四部分：端到端示例
# ============================================================

def create_sft_samples() -> List[SFTSample]:
    """创建示例 SFT 数据"""
    return [
        SFTSample(
            instruction="请解释什么是机器学习。",
            output="机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出决策或预测，而无需明确编程。"
        ),
        SFTSample(
            instruction="写一首关于春天的短诗。",
            output="春风拂面花自开，\n绿柳垂丝燕归来。\n万物复苏生机现，\n一年之计在此时。"
        ),
        SFTSample(
            instruction="如何学习编程？",
            output="学习编程的建议：1) 选择一门入门语言如 Python；2) 理解基本概念如变量、循环、函数；3) 通过项目实践；4) 阅读他人代码；5) 持续学习和练习。"
        )
    ]


def create_preference_samples() -> List[PreferenceSample]:
    """创建示例偏好数据"""
    return [
        PreferenceSample(
            prompt="请解释量子力学。",
            chosen="量子力学是描述微观粒子行为的物理学理论，核心概念包括波粒二象性、不确定性原理和量子叠加。",
            rejected="量子力学就是研究很小很小的东西的物理学。"
        ),
        PreferenceSample(
            prompt="推荐几本好书。",
            chosen="推荐书籍：1)《人类简史》- 尤瓦尔·赫拉利；2)《思考，快与慢》- 丹尼尔·卡尼曼；3)《原则》- 瑞·达利欧。",
            rejected="书都挺好的，你可以随便看看。"
        )
    ]


def run_rlhf_pipeline_example():
    """
    完整的 RLHF 流程示例
    
    注意：这是一个简化示例，实际使用需要：
    1. 更大的数据集
    2. 更长的训练时间
    3. 更细致的超参数调整
    """
    print("\n" + "=" * 60)
    print("RLHF 完整流程示例")
    print("=" * 60)
    
    try:
        # 配置
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"\n[1/3] 加载基础模型：{model_name}")
        
        # 加载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        # ========== 阶段 1: SFT ==========
        print("\n" + "=" * 60)
        print("阶段 1: SFT（Supervised Fine-Tuning）")
        print("=" * 60)
        
        sft_samples = create_sft_samples()
        sft_dataset = SFTDataset(sft_samples, tokenizer, max_length=256)
        
        sft_trainer = SFTTrainer(
            model=base_model,
            tokenizer=tokenizer,
            learning_rate=2e-5,
            batch_size=2,
            num_epochs=1,
            output_dir="./sft_example_output",
            device=device
        )
        
        # 实际训练时取消注释
        # sft_history = sft_trainer.train(sft_dataset)
        print("SFT 训练配置完成（示例模式，未实际训练）")
        
        # ========== 阶段 2: Reward Model ==========
        print("\n" + "=" * 60)
        print("阶段 2: Reward Model 训练")
        print("=" * 60)
        
        # 创建 Reward Model
        reward_base = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        reward_model = RewardModel(reward_base, pooling="mean")
        
        preference_samples = create_preference_samples()
        rm_dataset = RewardModelDataset(preference_samples, tokenizer, max_length=256)
        
        rm_trainer = RewardModelTrainer(
            reward_model=reward_model,
            tokenizer=tokenizer,
            learning_rate=1e-5,
            batch_size=2,
            num_epochs=1,
            output_dir="./rm_example_output",
            device=device
        )
        
        # 实际训练时取消注释
        # rm_history = rm_trainer.train(rm_dataset)
        print("Reward Model 训练配置完成（示例模式，未实际训练）")
        
        # ========== 阶段 3: PPO ==========
        print("\n" + "=" * 60)
        print("阶段 3: PPO 微调")
        print("=" * 60)
        
        # 创建价值模型（简化：使用与 Reward Model 相同的架构）
        value_base = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        value_model = RewardModel(value_base, pooling="mean")
        
        prompts = [s.instruction for s in sft_samples]
        
        ppo_trainer = PPOTrainer(
            policy_model=base_model,
            ref_model=base_model,  # 实际应使用 SFT 后的模型副本
            reward_model=reward_model,
            value_model=value_model,
            tokenizer=tokenizer,
            learning_rate=1e-6,
            batch_size=2,
            ppo_epochs=2,
            output_dir="./ppo_example_output",
            device=device
        )
        
        # 实际训练时取消注释
        # ppo_history = ppo_trainer.train(prompts, num_episodes=10)
        print("PPO 训练配置完成（示例模式，未实际训练）")
        
        print("\n" + "=" * 60)
        print("RLHF 流程示例完成！")
        print("=" * 60)
        print("\n提示：实际使用请确保：")
        print("  1. 安装依赖：pip install transformers torch accelerate")
        print("  2. 准备充足的数据集")
        print("  3. 使用 GPU 进行训练")
        print("  4. 根据任务调整超参数")
        
    except ImportError as e:
        print(f"需要安装依赖：{e}")
        print("\n请运行：pip install transformers torch accelerate")


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    print("第 26 章：RLHF 实战 - 代码示例")
    print("=" * 60)
    
    # 运行完整流程示例
    run_rlhf_pipeline_example()
    
    print("\n完整教程请参考 theory.md")
