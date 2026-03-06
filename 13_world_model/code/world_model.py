"""
world_model.py
第 13 章 世界模型基础 - 代码实现

包含：
1. 世界模型基础架构
2. 状态预测网络
3. 动作条件预测
4. 潜在空间可视化
5. 简单环境中的世界模型训练

作者：AI 前沿技术教程
日期：2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, List, Tuple, Optional
import gymnasium as gym


# ============================================================================
# 1. 世界模型基础架构
# ============================================================================

class Encoder(nn.Module):
    """
    编码器：将高维观察（如图像）压缩为低维潜在表示
    
    支持两种模式：
    - MLP：用于向量观察
    - CNN：用于图像观察
    """
    def __init__(self, obs_shape: Tuple, latent_dim: int, hidden_dim: int = 256, 
                 use_cnn: bool = False):
        super().__init__()
        self.use_cnn = use_cnn
        
        if use_cnn:
            # CNN 编码器用于图像
            self.network = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, hidden_dim),  # 假设输入 64x64
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            )
        else:
            # MLP 编码器用于向量观察
            obs_dim = int(np.prod(obs_shape))
            self.network = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        编码观察为潜在状态
        
        Args:
            obs: 观察张量 [batch_size, *obs_shape]
        
        Returns:
            latent: 潜在状态 [batch_size, latent_dim]
        """
        return self.network(obs)


class Decoder(nn.Module):
    """
    解码器：从潜在状态重建观察
    """
    def __init__(self, latent_dim: int, obs_shape: Tuple, hidden_dim: int = 256,
                 use_cnn: bool = False):
        super().__init__()
        self.use_cnn = use_cnn
        self.obs_shape = obs_shape
        
        if use_cnn:
            # CNN 解码器用于图像
            self.network = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 128 * 4 * 4),
                nn.ReLU(),
                nn.Unflatten(1, (128, 4, 4)),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, obs_shape[0], kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()  # 假设像素值归一化到 [0, 1]
            )
        else:
            # MLP 解码器用于向量观察
            obs_dim = int(np.prod(obs_shape))
            self.network = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, obs_dim)
            )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        从潜在状态解码观察
        
        Args:
            latent: 潜在状态 [batch_size, latent_dim]
        
        Returns:
            recon: 重建的观察 [batch_size, *obs_shape]
        """
        recon = self.network(latent)
        if self.use_cnn:
            return recon
        else:
            return recon.view(-1, *self.obs_shape)


class RSSM(nn.Module):
    """
    Recurrent State Space Model (RSSM)
    
    Dreamer 系列的核心组件，结合：
    - 确定性 RNN（GRU）捕捉时间依赖
    - 随机潜在变量表示不确定性
    
    状态组成：
    - h_t: 确定性隐藏状态
    - z_t: 随机潜在状态
    - s_t = (h_t, z_t): 完整状态表示
    """
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 32,
                 hidden_dim: int = 200):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # 观察编码器：o_t → z_t 的后验分布
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean 和 logvar
        )
        
        # 状态转移 GRU：(h_{t-1}, z_{t-1}, a_{t-1}) → h_t
        self.gru = nn.GRUCell(hidden_dim + latent_dim + action_dim, hidden_dim)
        
        # 先验头：h_t → z_t 的先验分布
        self.prior_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # 解码器：(h_t, z_t) → ô_t
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    
    def encode_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将观察编码为后验分布的参数"""
        params = self.obs_encoder(obs)
        mean, logvar = torch.chunk(params, 2, dim=-1)
        logvar = torch.clamp(logvar, -20, 2)
        return mean, logvar
    
    def get_prior(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从隐藏状态获得先验分布"""
        params = self.prior_head(hidden_state)
        mean, logvar = torch.chunk(params, 2, dim=-1)
        logvar = torch.clamp(logvar, -20, 2)
        return mean, logvar
    
    def get_posterior(self, hidden_state: torch.Tensor, 
                      obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从隐藏状态和观察获得后验分布"""
        return self.encode_obs(obs)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor,
                prev_hidden: torch.Tensor, prev_stochastic: torch.Tensor,
                use_posterior: bool = True) -> Dict[str, torch.Tensor]:
        """
        RSSM 单步前向传播
        
        Args:
            obs: 当前观察 [batch, obs_dim]
            action: 当前动作 [batch, action_dim]
            prev_hidden: 上一时刻隐藏状态 [batch, hidden_dim]
            prev_stochastic: 上一时刻随机状态 [batch, latent_dim]
            use_posterior: 是否使用后验（训练时用）或先验（推理时用）
        
        Returns:
            包含所有中间结果的字典
        """
        # 1. 计算先验分布 p(z_t | h_t)
        prior_mean, prior_logvar = self.get_prior(prev_hidden)
        
        # 2. 计算后验分布 q(z_t | h_t, o_t)
        if use_posterior:
            post_mean, post_logvar = self.get_posterior(prev_hidden, obs)
            # 重参数化采样
            std = torch.exp(0.5 * post_logvar)
            noise = torch.randn_like(post_mean)
            stochastic = post_mean + std * noise
        else:
            # 推理时只用先验
            post_mean, post_logvar = prior_mean, prior_logvar
            std = torch.exp(0.5 * prior_logvar)
            noise = torch.randn_like(prior_mean)
            stochastic = prior_mean + std * noise
        
        # 3. 更新确定性隐藏状态
        gru_input = torch.cat([prev_hidden, prev_stochastic, action], dim=-1)
        hidden_state = self.gru(gru_input, prev_hidden)
        
        # 4. 解码观察
        state_rep = torch.cat([hidden_state, stochastic], dim=-1)
        recon_obs = self.decoder(state_rep)
        
        return {
            'hidden': hidden_state,
            'stochastic': stochastic,
            'stochastic_mean': post_mean,
            'stochastic_logvar': post_logvar,
            'prior_mean': prior_mean,
            'prior_logvar': prior_logvar,
            'recon_obs': recon_obs
        }
    
    def compute_kl_loss(self, post_mean: torch.Tensor, post_logvar: torch.Tensor,
                        prior_mean: torch.Tensor, prior_logvar: torch.Tensor,
                        free_bits: float = 1.0) -> torch.Tensor:
        """
        计算 KL 散度损失（带 free bits 技巧）
        
        free bits 防止后验过早坍塌到先验
        """
        posterior = Normal(post_mean, torch.exp(0.5 * post_logvar))
        prior = Normal(prior_mean, torch.exp(0.5 * prior_logvar))
        
        kl = kl_divergence(posterior, prior).sum(dim=-1)
        kl = torch.max(kl, torch.full_like(kl, free_bits))
        
        return kl.mean()
    
    def imagine(self, initial_hidden: torch.Tensor, 
                initial_stochastic: torch.Tensor,
                actions: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        在潜在空间中"想象"未来轨迹（没有观察输入）
        
        Args:
            initial_hidden: 初始隐藏状态
            initial_stochastic: 初始随机状态
            actions: 动作序列 [horizon, batch, action_dim]
        
        Returns:
            想象的状态序列
        """
        hidden = initial_hidden
        stochastic = initial_stochastic
        imagined = []
        
        for t in range(actions.shape[0]):
            action = actions[t]
            
            # 只使用先验（没有观察）
            prior_mean, prior_logvar = self.get_prior(hidden)
            std = torch.exp(0.5 * prior_logvar)
            stochastic = prior_mean + std * torch.randn_like(prior_mean)
            
            # 更新隐藏状态
            gru_input = torch.cat([hidden, stochastic, action], dim=-1)
            hidden = self.gru(gru_input, hidden)
            
            imagined.append({
                'hidden': hidden,
                'stochastic': stochastic,
                'action': action
            })
        
        return imagined


# ============================================================================
# 2. 状态预测网络
# ============================================================================

class DeterministicPredictor(nn.Module):
    """
    确定性状态预测器
    
    适用于确定性环境或作为基线模型
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        预测下一状态
        
        Args:
            state: 当前状态 [batch, state_dim]
            action: 动作 [batch, action_dim]
        
        Returns:
            next_state: 预测的下一状态 [batch, state_dim]
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class StochasticPredictor(nn.Module):
    """
    随机性状态预测器
    
    输出状态的概率分布（均值和方差）
    能够表示环境的不确定性
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, state_dim)
        self.logvar_head = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测下一状态的分布
        
        Returns:
            mean: 预测均值
            logvar: 预测对数方差
        """
        x = torch.cat([state, action], dim=-1)
        h = self.shared(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, -10, 10)
        return mean, logvar
    
    def sample(self, state: torch.Tensor, action: torch.Tensor, 
               num_samples: int = 1) -> torch.Tensor:
        """
        从预测分布中采样
        
        Args:
            num_samples: 采样数量
        
        Returns:
            samples: 采样的下一状态 [batch, num_samples, state_dim]
        """
        mean, logvar = self.forward(state, action)
        std = torch.exp(0.5 * logvar)
        
        if num_samples > 1:
            mean = mean.unsqueeze(1).expand(-1, num_samples, -1)
            std = std.unsqueeze(1).expand(-1, num_samples, -1)
        
        noise = torch.randn_like(mean)
        return mean + std * noise
    
    def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """返回预测均值（确定性预测）"""
        mean, _ = self.forward(state, action)
        return mean


class MultiStepPredictor(nn.Module):
    """
    多步状态预测器
    
    递归预测多个时间步的未来状态
    """
    def __init__(self, single_step_predictor: nn.Module):
        super().__init__()
        self.predictor = single_step_predictor
    
    def forward(self, initial_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        多步预测
        
        Args:
            initial_state: 初始状态 [batch, state_dim]
            actions: 动作序列 [horizon, batch, action_dim]
        
        Returns:
            predicted_states: 预测的状态序列 [horizon+1, batch, state_dim]
        """
        horizon = actions.shape[0]
        batch_size = initial_state.shape[0]
        state_dim = initial_state.shape[1]
        
        # 存储所有预测的状态
        states = [initial_state]
        current_state = initial_state
        
        for t in range(horizon):
            action = actions[t]
            next_state = self.predictor.predict(current_state, action)
            states.append(next_state)
            current_state = next_state
        
        return torch.stack(states, dim=0)


# ============================================================================
# 3. 动作条件预测
# ============================================================================

class ActionConditionedWorldModel(nn.Module):
    """
    动作条件世界模型
    
    整合编码器、RSSM、解码器和动作预测
    """
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 32,
                 hidden_dim: int = 200):
        super().__init__()
        self.rssm = RSSM(obs_dim, action_dim, latent_dim, hidden_dim)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # 奖励预测头
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 终止预测头（预测是否 episode 结束）
        self.done_head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor,
                reward_seq: torch.Tensor, done_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        处理序列数据
        
        Args:
            obs_seq: 观察序列 [T, B, obs_dim]
            action_seq: 动作序列 [T, B, action_dim]
            reward_seq: 奖励序列 [T, B, 1]
            done_seq: 终止序列 [T, B, 1]
        
        Returns:
            包含所有预测和损失的字典
        """
        T, B = obs_seq.shape[0], obs_seq.shape[1]
        
        # 初始化状态
        hidden = torch.zeros(B, self.hidden_dim, device=obs_seq.device)
        stochastic = torch.zeros(B, self.latent_dim, device=obs_seq.device)
        
        # 存储每一步的结果
        kl_losses = []
        recon_losses = []
        reward_losses = []
        done_losses = []
        
        priors = []
        posteriors = []
        recons = []
        
        for t in range(T):
            # RSSM 前向传播
            result = self.rssm(
                obs_seq[t], action_seq[t], hidden, stochastic, use_posterior=True
            )
            
            # 计算 KL 损失
            kl = self.rssm.compute_kl_loss(
                result['stochastic_mean'], result['stochastic_logvar'],
                result['prior_mean'], result['prior_logvar']
            )
            kl_losses.append(kl)
            
            # 计算重建损失
            recon_loss = F.mse_loss(result['recon_obs'], obs_seq[t])
            recon_losses.append(recon_loss)
            
            # 预测奖励
            state_rep = torch.cat([result['hidden'], result['stochastic']], dim=-1)
            reward_pred = self.reward_head(state_rep)
            reward_loss = F.mse_loss(reward_pred, reward_seq[t])
            reward_losses.append(reward_loss)
            
            # 预测终止
            done_pred = self.done_head(state_rep)
            done_loss = F.binary_cross_entropy_with_logits(done_pred, done_seq[t])
            done_losses.append(done_loss)
            
            # 存储结果
            priors.append({
                'mean': result['prior_mean'],
                'logvar': result['prior_logvar']
            })
            posteriors.append({
                'mean': result['stochastic_mean'],
                'logvar': result['stochastic_logvar']
            })
            recons.append(result['recon_obs'])
            
            # 更新状态
            hidden = result['hidden']
            stochastic = result['stochastic']
        
        return {
            'kl_loss': torch.stack(kl_losses).mean(),
            'recon_loss': torch.stack(recon_losses).mean(),
            'reward_loss': torch.stack(reward_losses).mean(),
            'done_loss': torch.stack(done_losses).mean(),
            'priors': priors,
            'posteriors': posteriors,
            'recons': recons,
            'final_hidden': hidden,
            'final_stochastic': stochastic
        }
    
    def imagine(self, initial_hidden: torch.Tensor, initial_stochastic: torch.Tensor,
                actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        在潜在空间中想象未来
        
        Args:
            initial_hidden: 初始隐藏状态 [B, hidden_dim]
            initial_stochastic: 初始随机状态 [B, latent_dim]
            actions: 动作序列 [T, B, action_dim]
        
        Returns:
            想象的轨迹（状态、奖励、终止）
        """
        imagined = self.rssm.imagine(initial_hidden, initial_stochastic, actions)
        
        rewards = []
        dones = []
        
        for step in imagined:
            state_rep = torch.cat([step['hidden'], step['stochastic']], dim=-1)
            reward = self.reward_head(state_rep)
            done = torch.sigmoid(self.done_head(state_rep))
            rewards.append(reward)
            dones.append(done)
        
        return {
            'states': imagined,
            'rewards': torch.stack(rewards, dim=0),
            'dones': torch.stack(dones, dim=0)
        }


# ============================================================================
# 4. 潜在空间可视化
# ============================================================================

class LatentSpaceVisualizer:
    """
    潜在空间可视化工具
    
    提供多种可视化方法：
    - PCA 降维
    - t-SNE 降维
    - 潜在空间插值
    - 潜在遍历（latent traversal）
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        model.eval()
    
    @torch.no_grad()
    def encode_dataset(self, data_loader) -> np.ndarray:
        """
        编码整个数据集为潜在表示
        
        Args:
            data_loader: 数据加载器
        
        Returns:
            latent_codes: 所有潜在编码 [N, latent_dim]
        """
        latent_codes = []
        
        for batch in data_loader:
            obs = batch['obs'].to(self.device)
            if hasattr(self.model, 'encode'):
                latent = self.model.encode(obs)
            elif hasattr(self.model, 'rssm'):
                latent = self.model.rssm.encode_obs(obs)[0]  # 取均值
            else:
                raise ValueError("模型没有编码方法")
            
            latent_codes.append(latent.cpu().numpy())
        
        return np.concatenate(latent_codes, axis=0)
    
    def plot_pca(self, latent_codes: np.ndarray, labels: Optional[np.ndarray] = None,
                 title: str = '潜在空间 PCA 可视化'):
        """
        使用 PCA 可视化潜在空间
        
        Args:
            latent_codes: 潜在编码 [N, latent_dim]
            labels: 可选的类别标签
        """
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_codes)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            scatter = ax.scatter(
                latent_2d[:, 0], latent_2d[:, 1],
                c=labels, cmap='viridis', alpha=0.6, s=10
            )
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=10)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return latent_2d, pca
    
    def plot_tsne(self, latent_codes: np.ndarray, labels: Optional[np.ndarray] = None,
                  perplexity: int = 30, title: str = '潜在空间 t-SNE 可视化'):
        """
        使用 t-SNE 可视化潜在空间
        """
        from sklearn.manifold import TSNE
        
        # t-SNE 计算量大，限制样本数
        n_samples = min(1000, len(latent_codes))
        indices = np.random.choice(len(latent_codes), n_samples, replace=False)
        latent_subset = latent_codes[indices]
        labels_subset = labels[indices] if labels is not None else None
        
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
        latent_2d = tsne.fit_transform(latent_subset)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels_subset is not None:
            scatter = ax.scatter(
                latent_2d[:, 0], latent_2d[:, 1],
                c=labels_subset, cmap='viridis', alpha=0.6, s=10
            )
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=10)
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(title)
        
        plt.tight_layout()
        plt.show()
        
        return latent_2d
    
    @torch.no_grad()
    def interpolate(self, obs1: torch.Tensor, obs2: torch.Tensor,
                    num_steps: int = 10) -> List[torch.Tensor]:
        """
        在潜在空间中插值
        
        Args:
            obs1: 起始观察
            obs2: 结束观察
            num_steps: 插值步数
        
        Returns:
            插值生成的重建观察序列
        """
        self.model.eval()
        
        # 编码
        if hasattr(self.model, 'encode'):
            z1 = self.model.encode(obs1.unsqueeze(0))
            z2 = self.model.encode(obs2.unsqueeze(0))
        elif hasattr(self.model, 'rssm'):
            z1 = self.model.rssm.encode_obs(obs1.unsqueeze(0))[0]
            z2 = self.model.rssm.encode_obs(obs2.unsqueeze(0))[0]
        
        # 插值
        reconstructions = []
        for alpha in torch.linspace(0, 1, num_steps):
            z = (1 - alpha) * z1 + alpha * z2
            
            if hasattr(self.model, 'decode'):
                recon = self.model.decode(z)
            elif hasattr(self.model, 'rssm'):
                # 对于 RSSM，需要创建虚拟隐藏状态
                hidden = torch.zeros(1, self.model.rssm.hidden_dim, device=z.device)
                state_rep = torch.cat([hidden, z], dim=-1)
                recon = self.model.rssm.decoder(state_rep)
            
            reconstructions.append(recon.squeeze(0).cpu())
        
        return reconstructions
    
    def plot_interpolation(self, obs1: torch.Tensor, obs2: torch.Tensor,
                           num_steps: int = 10, obs_shape: Tuple = None):
        """
        可视化潜在空间插值
        """
        reconstructions = self.interpolate(obs1, obs2, num_steps)
        
        fig, axes = plt.subplots(2, num_steps, figsize=(20, 4))
        
        for i, recon in enumerate(reconstructions):
            if len(recon.shape) == 1 or obs_shape is None:
                # 向量观察
                axes[0, i].plot(recon.numpy())
            else:
                # 图像观察
                recon_img = recon.numpy()
                if len(recon_img.shape) == 3:
                    recon_img = np.transpose(recon_img, (1, 2, 0))
                axes[0, i].imshow(recon_img)
            
            axes[0, i].set_title(f'α={i/(num_steps-1):.2f}')
            axes[0, i].axis('off')
        
        # 显示原始图像
        obs1_np = obs1.cpu().numpy()
        obs2_np = obs2.cpu().numpy()
        
        if len(obs1_np.shape) == 3:
            obs1_np = np.transpose(obs1_np, (1, 2, 0))
            obs2_np = np.transpose(obs2_np, (1, 2, 0))
        
        axes[1, 0].imshow(obs1_np)
        axes[1, 0].set_title('原始 1')
        axes[1, 0].axis('off')
        axes[1, -1].imshow(obs2_np)
        axes[1, -1].set_title('原始 2')
        axes[1, -1].axis('off')
        
        # 隐藏中间的
        for i in range(1, num_steps - 1):
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# 5. 简单环境中的世界模型训练
# ============================================================================

class WorldModelTrainer:
    """
    世界模型训练器
    
    支持多种环境和训练配置
    """
    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # 创建模型
        self.model = ActionConditionedWorldModel(
            obs_dim=config['obs_dim'],
            action_dim=config['action_dim'],
            latent_dim=config.get('latent_dim', 32),
            hidden_dim=config.get('hidden_dim', 200)
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3)
        )
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=config.get('buffer_size', 100000))
        
        # 训练统计
        self.stats = {
            'total_steps': 0,
            'total_episodes': 0,
            'training_steps': 0
        }
    
    def add_experience(self, obs: np.ndarray, action: np.ndarray,
                       next_obs: np.ndarray, reward: float, done: bool):
        """
        添加单步经验到回放缓冲区
        """
        self.replay_buffer.append({
            'obs': obs,
            'action': action,
            'next_obs': next_obs,
            'reward': reward,
            'done': done
        })
    
    def add_episode(self, trajectory: List[Dict]):
        """
        添加完整轨迹到回放缓冲区
        
        Args:
            trajectory: 轨迹列表，每项包含 {obs, action, next_obs, reward, done}
        """
        for step in trajectory:
            self.add_experience(
                step['obs'], step['action'],
                step['next_obs'], step['reward'], step['done']
            )
        self.stats['total_episodes'] += 1
    
    def sample_batch(self, batch_size: int, sequence_length: int = 50) -> Dict:
        """
        采样序列批次用于训练
        
        Args:
            batch_size: 批次大小
            sequence_length: 序列长度
        
        Returns:
            批次字典，包含序列张量
        """
        if len(self.replay_buffer) < sequence_length:
            return None
        
        # 随机选择起始点
        max_start = len(self.replay_buffer) - sequence_length
        if max_start <= 0:
            return None
        
        starts = np.random.randint(0, max_start, batch_size)
        
        # 构建序列
        obs_seq = []
        action_seq = []
        reward_seq = []
        done_seq = []
        
        for start in starts:
            segment = list(self.replay_buffer)[start:start + sequence_length]
            
            obs_seq.append([step['obs'] for step in segment])
            action_seq.append([step['action'] for step in segment])
            reward_seq.append([step['reward'] for step in segment])
            done_seq.append([step['done'] for step in segment])
        
        # 转换为张量
        batch = {
            'obs': torch.FloatTensor(obs_seq).transpose(0, 1).to(self.device),
            'action': torch.FloatTensor(action_seq).transpose(0, 1).to(self.device),
            'reward': torch.FloatTensor(reward_seq).unsqueeze(-1).transpose(0, 1).to(self.device),
            'done': torch.FloatTensor(done_seq).unsqueeze(-1).transpose(0, 1).to(self.device)
        }
        
        return batch
    
    def train_step(self, batch_size: int = 32, sequence_length: int = 50) -> Dict:
        """
        单步训练
        
        Returns:
            训练指标字典
        """
        batch = self.sample_batch(batch_size, sequence_length)
        if batch is None:
            return None
        
        self.optimizer.zero_grad()
        
        # 前向传播
        result = self.model(
            batch['obs'], batch['action'],
            batch['reward'], batch['done']
        )
        
        # 计算总损失
        kl_weight = self.config.get('kl_weight', 1.0)
        total_loss = (
            result['recon_loss'] +
            result['reward_loss'] +
            result['done_loss'] +
            kl_weight * result['kl_loss']
        )
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        max_grad_norm = self.config.get('max_grad_norm', 100)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        self.optimizer.step()
        
        self.stats['training_steps'] += 1
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': result['recon_loss'].item(),
            'reward_loss': result['reward_loss'].item(),
            'done_loss': result['done_loss'].item(),
            'kl_loss': result['kl_loss'].item()
        }
    
    def train(self, env: gym.Env, num_episodes: int = 100,
              train_every: int = 1, batch_size: int = 32,
              log_interval: int = 10) -> List[Dict]:
        """
        完整训练循环
        
        Args:
            env: Gym 环境
            num_episodes: 训练回合数
            train_every: 每多少回合训练一次
            batch_size: 训练批次大小
            log_interval: 日志间隔
        
        Returns:
            训练历史
        """
        history = []
        
        for episode in range(num_episodes):
            # 收集数据
            obs, _ = env.reset()
            episode_reward = 0
            trajectory = []
            
            while True:
                # 随机动作（训练初期）
                action = env.action_space.sample()
                
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 存储经验
                self.add_experience(obs, action, next_obs, reward, done)
                trajectory.append({
                    'obs': obs, 'action': action,
                    'next_obs': next_obs, 'reward': reward, 'done': done
                })
                
                obs = next_obs
                episode_reward += reward
                self.stats['total_steps'] += 1
                
                if done:
                    break
            
            # 训练模型
            if episode % train_every == 0 and len(self.replay_buffer) >= batch_size:
                metrics = self.train_step(batch_size=batch_size)
                if metrics:
                    metrics['episode'] = episode
                    metrics['episode_reward'] = episode_reward
                    history.append(metrics)
                    
                    if episode % log_interval == 0:
                        print(f"Episode {episode}: "
                              f"Reward={episode_reward:.2f}, "
                              f"Loss={metrics['total_loss']:.4f}")
        
        return history
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats,
            'config': self.config
        }, path)
        print(f"模型已保存到 {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.stats = checkpoint['stats']
        print(f"模型已从 {path} 加载")


# ============================================================================
# 主程序示例
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("世界模型基础 - 代码示例")
    print("=" * 60)
    
    # 配置
    config = {
        'obs_dim': 4,  # CartPole 观察维度
        'action_dim': 2,  # CartPole 动作维度
        'latent_dim': 32,
        'hidden_dim': 200,
        'learning_rate': 1e-3,
        'kl_weight': 0.1,
        'buffer_size': 10000,
        'max_grad_norm': 100,
        'device': 'cpu'
    }
    
    # 创建环境和训练器
    env = gym.make('CartPole-v1')
    trainer = WorldModelTrainer(config)
    
    print(f"\n开始训练...")
    print(f"环境：CartPole-v1")
    print(f"配置：潜在维度={config['latent_dim']}, 隐藏维度={config['hidden_dim']}")
    print()
    
    # 训练
    history = trainer.train(
        env,
        num_episodes=50,
        train_every=1,
        batch_size=32,
        log_interval=10
    )
    
    # 可视化训练历史
    if history:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        losses = [h['total_loss'] for h in history]
        plt.plot(losses)
        plt.xlabel('训练步数')
        plt.ylabel('总损失')
        plt.title('训练损失曲线')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        rewards = [h['episode_reward'] for h in history]
        plt.plot(rewards)
        plt.xlabel('回合')
        plt.ylabel('回合奖励')
        plt.title('回合奖励曲线')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 保存模型
    trainer.save('world_model_checkpoint.pth')
    
    print("\n训练完成！")
    print(f"总步数：{trainer.stats['total_steps']}")
    print(f"总回合：{trainer.stats['total_episodes']}")
    print(f"训练步数：{trainer.stats['training_steps']}")
