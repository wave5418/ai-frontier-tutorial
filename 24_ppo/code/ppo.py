#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 24 章 PPO 与策略梯度 - 代码实现
包含：Actor-Critic 架构、GAE 优势估计、PPO-Clip 算法、训练循环
支持连续和离散动作空间
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from collections import deque
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import gymnasium as gym


# ============================================================
# 1. Actor-Critic 网络架构
# ============================================================

class ActorCritic(nn.Module):
    """
    Actor-Critic 网络
    
    共享底层特征，输出策略和价值
    支持连续和离散动作空间
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_size: int = 64, 
                 continuous: bool = False,
                 action_std: float = 0.5):
        """
        初始化 Actor-Critic 网络
        
        参数:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_size: 隐藏层大小
            continuous: 是否为连续动作空间
            action_std: 连续动作的标准差（可学习或固定）
        """
        super(ActorCritic, self).__init__()
        
        self.continuous = continuous
        self.action_dim = action_dim
        
        # 共享的特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        # Actor 网络（策略）
        if continuous:
            # 连续动作：输出动作分布的均值
            self.actor_mean = nn.Linear(hidden_size, action_dim)
            # 可学习的对数标准差
            self.actor_log_std = nn.Parameter(torch.ones(1, action_dim) * np.log(action_std))
        else:
            # 离散动作：输出动作概率
            self.actor = nn.Linear(hidden_size, action_dim)
        
        # Critic 网络（价值）
        self.critic = nn.Linear(hidden_size, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """正交初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # 最后一层初始化为零
        if self.continuous:
            nn.init.constant_(self.actor_mean.weight, 0)
            nn.init.constant_(self.actor_mean.bias, 0)
        else:
            nn.init.constant_(self.actor.weight, 0)
            nn.init.constant_(self.actor.bias, 0)
        
        nn.init.constant_(self.critic.weight, 0)
        nn.init.constant_(self.critic.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            state: 状态输入
        
        返回:
            action_logits: 动作 logits（离散）或均值（连续）
            value: 状态价值
        """
        # 共享特征
        features = self.shared(state)
        
        # Actor 输出
        if self.continuous:
            action_mean = self.actor_mean(features)
            action_logits = action_mean
        else:
            action_logits = self.actor(features)
        
        # Critic 输出
        value = self.critic(features)
        
        return action_logits, value
    
    def get_action(self, state: torch.Tensor, 
                   deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor]:
        """
        采样动作
        
        参数:
            state: 状态
            deterministic: 是否使用确定性策略
        
        返回:
            action: 采样动作
            log_prob: 动作的对数概率
        """
        action_logits, _ = self.forward(state)
        
        if self.continuous:
            # 连续动作空间
            action_std = torch.exp(self.actor_log_std).expand_as(action_logits)
            dist = Normal(action_logits, action_std)
            
            if deterministic:
                action = action_logits
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            
            # 裁剪动作到 [-1, 1]
            action = torch.clamp(action, -1, 1)
        else:
            # 离散动作空间
            dist = Categorical(logits=action_logits)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1, keepdim=True)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action.squeeze()).unsqueeze(-1)
        
        return action.numpy(), log_prob
    
    def evaluate_actions(self, state: torch.Tensor, 
                         action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作（用于 PPO 更新）
        
        参数:
            state: 状态
            action: 动作
        
        返回:
            log_prob: 动作的对数概率
            value: 状态价值
            entropy: 策略熵
        """
        action_logits, value = self.forward(state)
        
        if self.continuous:
            action_std = torch.exp(self.actor_log_std).expand_as(action_logits)
            dist = Normal(action_logits, action_std)
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
        else:
            dist = Categorical(logits=action_logits)
            log_prob = dist.log_prob(action.squeeze()).unsqueeze(-1)
            entropy = dist.entropy().unsqueeze(-1)
        
        return log_prob, value, entropy
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        _, value = self.forward(state)
        return value


# ============================================================
# 2. GAE (Generalized Advantage Estimation)
# ============================================================

def compute_gae(rewards: np.ndarray, 
                values: np.ndarray, 
                dones: np.ndarray, 
                gamma: float = 0.99, 
                lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 GAE 优势函数
    
    参数:
        rewards: 奖励序列 (T,)
        values: 价值估计 (T+1,)，包含终止状态的价值
        dones: 终止标志 (T,)
        gamma: 折扣因子
        lam: GAE λ 参数
    
    返回:
        advantages: 优势函数估计 (T,)
        returns: 回报估计 (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T)
    
    # 计算 TD 误差
    # δ_t = r_t + γV(s_{t+1}) - V(s_t)
    td_deltas = np.zeros(T)
    for t in range(T):
        next_value = 0 if dones[t] else values[t + 1]
        td_deltas[t] = rewards[t] + gamma * next_value - values[t]
    
    # GAE 计算
    # A_t = δ_t + γλδ_{t+1} + (γλ)^2δ_{t+2} + ...
    last_gae = 0
    for t in reversed(range(T)):
        if dones[t]:
            last_gae = 0
        else:
            last_gae = td_deltas[t] + gamma * lam * last_gae
        advantages[t] = last_gae
    
    # 计算回报：returns = advantages + values
    returns = advantages + values[:T]
    
    return advantages, returns


# ============================================================
# 3. PPO-Clip 算法实现
# ============================================================

class PPO:
    """
    PPO-Clip 算法实现
    
    支持连续和离散动作空间
    包含 GAE 优势估计、优势归一化、梯度裁剪等技巧
    """
    
    def __init__(self, 
                 env: gym.Env,
                 hidden_size: int = 64,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 clip_epsilon: float = 0.2,
                 c1: float = 0.5,  # 价值函数系数
                 c2: float = 0.01,  # 熵系数
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 10,
                 mini_batch_size: int = 64,
                 continuous: Optional[bool] = None):
        """
        初始化 PPO 智能体
        
        参数:
            env: 环境
            hidden_size: 隐藏层大小
            lr: 学习率
            gamma: 折扣因子
            lam: GAE λ
            clip_epsilon: PPO 截断参数
            c1: 价值函数损失系数
            c2: 熵正则系数
            max_grad_norm: 最大梯度范数
            ppo_epochs: 每次采样的 PPO 更新轮数
            mini_batch_size: 小批量大小
            continuous: 是否连续动作空间（None 则自动检测）
        """
        self.env = env
        
        # 检测动作空间类型
        if continuous is None:
            self.continuous = isinstance(env.action_space, gym.spaces.Box)
        else:
            self.continuous = continuous
        
        self.state_dim = env.observation_space.shape[0]
        
        if self.continuous:
            self.action_dim = env.action_space.shape[0]
        else:
            self.action_dim = env.action_space.n
        
        # 超参数
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        # 创建网络和优化器
        self.policy = ActorCritic(
            self.state_dim, 
            self.action_dim, 
            hidden_size=hidden_size,
            continuous=self.continuous
        )
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'kl_divergences': []
        }
    
    def select_action(self, state: np.ndarray, 
                      deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        选择动作
        
        参数:
            state: 状态
            deterministic: 是否确定性选择
        
        返回:
            action: 动作
            log_prob: 对数概率
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.policy.get_action(state_tensor, deterministic)
        return action[0], log_prob[0].item()
    
    def collect_experience(self, 
                          n_steps: int = 2048,
                          render: bool = False) -> dict:
        """
        收集经验轨迹
        
        参数:
            n_steps: 收集步数
            render: 是否渲染
        
        返回:
            包含 states, actions, rewards, dones, log_probs, values 的字典
        """
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for _ in range(n_steps):
            # 选择动作
            action, log_prob = self.select_action(state)
            value = self.policy.get_value(torch.FloatTensor(state).unsqueeze(0)).item()
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            if render:
                self.env.render()
            
            # 存储经验
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                state, _ = self.env.reset()
                self.training_stats['episode_rewards'].append(episode_reward)
                self.training_stats['episode_lengths'].append(episode_length)
                episode_reward = 0
                episode_length = 0
        
        # 添加最后状态的价值（用于 GAE）
        last_value = self.policy.get_value(torch.FloatTensor(state).unsqueeze(0)).item()
        values.append(last_value)
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'log_probs': np.array(log_probs),
            'values': np.array(values)
        }
    
    def update(self, batch: dict) -> dict:
        """
        PPO 更新
        
        参数:
            batch: 经验批次
        
        返回:
            训练统计信息
        """
        # 转换为张量
        states = torch.FloatTensor(batch['states'])
        actions = torch.FloatTensor(batch['actions'])
        old_log_probs = torch.FloatTensor(batch['log_probs'])
        advantages = torch.FloatTensor(batch['advantages'])
        returns = torch.FloatTensor(batch['returns'])
        
        # 优势归一化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 统计信息
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        n_updates = 0
        
        # 多轮 PPO 更新
        for _ in range(self.ppo_epochs):
            # 打乱数据
            indices = np.random.permutation(len(states))
            
            # 小批量更新
            for start in range(0, len(states), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                # 获取小批量数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 评估当前策略
                log_probs, values, entropies = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # 计算概率比
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # PPO 截断目标
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # 熵正则
                entropy_loss = -entropies.mean()
                
                # 总损失
                loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 统计 KL 散度
                kl = (batch_old_log_probs - log_probs).mean().item()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropies.mean().item()
                total_kl += kl
                n_updates += 1
        
        return {
            'total_loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'kl_divergence': total_kl / n_updates
        }
    
    def train(self, 
              total_timesteps: int = 100000,
              n_steps: int = 2048,
              log_interval: int = 10,
              render: bool = False) -> dict:
        """
        训练 PPO 智能体
        
        参数:
            total_timesteps: 总训练步数
            n_steps: 每次采样的步数
            log_interval: 日志间隔
            render: 是否渲染
        
        返回:
            训练统计
        """
        print(f"开始训练 PPO，总步数：{total_timesteps}")
        print(f"动作空间类型：{'连续' if self.continuous else '离散'}")
        
        timesteps_collected = 0
        
        while timesteps_collected < total_timesteps:
            # 收集经验
            batch = self.collect_experience(n_steps=n_steps, render=render)
            timesteps_collected += n_steps
            
            # 计算 GAE
            advantages, returns = compute_gae(
                batch['rewards'], 
                batch['values'], 
                batch['dones'],
                gamma=self.gamma,
                lam=self.lam
            )
            
            batch['advantages'] = advantages
            batch['returns'] = returns
            
            # PPO 更新
            stats = self.update(batch)
            
            # 记录统计
            self.training_stats['losses'].append(stats['total_loss'])
            self.training_stats['kl_divergences'].append(stats['kl_divergence'])
            
            # 日志
            if len(self.training_stats['episode_rewards']) > 0:
                avg_reward = np.mean(self.training_stats['episode_rewards'][-log_interval:])
            else:
                avg_reward = 0
            
            if (timesteps_collected // n_steps) % log_interval == 0:
                print(f"步数：{timesteps_collected}/{total_timesteps}, "
                      f"平均奖励：{avg_reward:.2f}, "
                      f"损失：{stats['total_loss']:.4f}, "
                      f"KL: {stats['kl_divergence']:.4f}")
        
        print("训练完成！")
        return self.training_stats
    
    def test(self, 
             n_episodes: int = 10, 
             render: bool = True,
             deterministic: bool = True) -> float:
        """
        测试智能体
        
        参数:
            n_episodes: 测试回合数
            render: 是否渲染
            deterministic: 是否使用确定性策略
        
        返回:
            平均奖励
        """
        rewards = []
        
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.select_action(state, deterministic=deterministic)
                
                if render:
                    self.env.render()
                
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            rewards.append(episode_reward)
            
            if render:
                import time
                time.sleep(0.5)
        
        avg_reward = np.mean(rewards)
        print(f"测试完成，平均奖励：{avg_reward:.2f} ± {np.std(rewards):.2f}")
        return avg_reward
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
        print(f"模型已保存到 {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        print(f"模型已从 {path} 加载")
    
    def plot_results(self, save_path: str = 'ppo_results.png'):
        """可视化训练结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 回合奖励
        if len(self.training_stats['episode_rewards']) > 0:
            axes[0, 0].plot(self.training_stats['episode_rewards'])
            axes[0, 0].set_title('回合奖励')
            axes[0, 0].set_xlabel('回合')
            axes[0, 0].set_ylabel('奖励')
            axes[0, 0].grid(True)
        
        # 回合长度
        if len(self.training_stats['episode_lengths']) > 0:
            axes[0, 1].plot(self.training_stats['episode_lengths'])
            axes[0, 1].set_title('回合长度')
            axes[0, 1].set_xlabel('回合')
            axes[0, 1].set_ylabel('步数')
            axes[0, 1].grid(True)
        
        # 损失
        if len(self.training_stats['losses']) > 0:
            axes[1, 0].plot(self.training_stats['losses'])
            axes[1, 0].set_title('训练损失')
            axes[1, 0].set_xlabel('更新次数')
            axes[1, 0].set_ylabel('损失')
            axes[1, 0].grid(True)
        
        # KL 散度
        if len(self.training_stats['kl_divergences']) > 0:
            axes[1, 1].plot(self.training_stats['kl_divergences'])
            axes[1, 1].set_title('KL 散度')
            axes[1, 1].set_xlabel('更新次数')
            axes[1, 1].set_ylabel('KL')
            axes[1, 1].axhline(y=0.02, color='r', linestyle='--', label='目标 KL')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"训练结果已保存到 {save_path}")


# ============================================================
# 4. 测试与示例
# ============================================================

def test_ppo_discrete():
    """测试 PPO 在离散动作空间（CartPole）"""
    print("=" * 60)
    print("PPO 测试 - 离散动作空间 (CartPole-v1)")
    print("=" * 60)
    
    env = gym.make('CartPole-v1')
    agent = PPO(env, hidden_size=64, lr=3e-4)
    
    # 训练
    agent.train(total_timesteps=50000, n_steps=2048, log_interval=5)
    
    # 测试
    agent.test(n_episodes=5, render=False)
    
    # 保存和可视化
    agent.save('ppo_cartpole.pth')
    agent.plot_results('ppo_cartpole_results.png')
    
    env.close()


def test_ppo_continuous():
    """测试 PPO 在连续动作空间（Pendulum）"""
    print("=" * 60)
    print("PPO 测试 - 连续动作空间 (Pendulum-v1)")
    print("=" * 60)
    
    env = gym.make('Pendulum-v1')
    agent = PPO(env, hidden_size=64, lr=3e-4)
    
    # 训练
    agent.train(total_timesteps=100000, n_steps=2048, log_interval=5)
    
    # 测试
    agent.test(n_episodes=5, render=False)
    
    # 保存和可视化
    agent.save('ppo_pendulum.pth')
    agent.plot_results('ppo_pendulum_results.png')
    
    env.close()


def demo_gae():
    """演示 GAE 计算"""
    print("=" * 60)
    print("GAE 优势估计演示")
    print("=" * 60)
    
    # 模拟轨迹
    T = 10
    rewards = np.array([1.0, 0.5, 0.8, 0.3, 1.2, 0.7, 0.9, 0.4, 1.1, 0.6])
    values = np.array([0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.5, 0.6, 0.4, 0.7, 0.0])
    dones = np.array([False] * T)
    
    gamma = 0.99
    lam = 0.95
    
    advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
    
    print(f"奖励：{rewards}")
    print(f"价值：{values[:-1]}")
    print(f"优势：{advantages}")
    print(f"回报：{returns}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='奖励')
    plt.plot(values[:-1], label='价值')
    plt.plot(advantages, label='优势 (GAE)')
    plt.plot(returns, label='回报')
    plt.legend()
    plt.grid(True)
    plt.title('GAE 优势估计')
    plt.savefig('gae_demo.png', dpi=150)
    print("GAE 演示图已保存到 gae_demo.png")


if __name__ == "__main__":
    # 运行测试
    
    # 1. GAE 演示
    demo_gae()
    
    print("\n")
    
    # 2. 离散动作空间测试
    try:
        test_ppo_discrete()
    except Exception as e:
        print(f"离散动作测试失败：{e}")
    
    print("\n")
    
    # 3. 连续动作空间测试
    try:
        test_ppo_continuous()
    except Exception as e:
        print(f"连续动作测试失败：{e}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
