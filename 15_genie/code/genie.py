#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genie 基础架构实现
=================
DeepMind 生成式交互环境 (Generative Interactive Environment) 的简化实现

本代码演示 Genie 的核心组件：
1. 潜在空间编码器 (Latent Space Encoder)
2. 动作条件生成器 (Action-Conditioned Generator)
3. 自回归解码器 (Autoregressive Decoder)
4. 智能体训练示例
5. 视频生成示例

注意：这是教学实现，简化了原始 Genie 的复杂度，但保留了核心思想。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt


# ============================================================
# 1. 潜在空间编码器 (Latent Space Encoder)
# ============================================================

class LatentEncoder(nn.Module):
    """
    将视频帧编码到潜在空间
    
    架构：CNN 编码器 → 潜在向量
    输入：(batch, channels, height, width) 视频帧
    输出：(batch, latent_dim) 潜在表示
    """
    
    def __init__(self, latent_dim: int = 64, input_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        
        # CNN 编码器：逐步降采样
        self.encoder = nn.Sequential(
            # 输入：3x64x64 → 输出：32x32x32
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 32x32x32 → 64x16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 64x16x16 → 128x8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 128x8x8 → 256x4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 展平 + 全连接
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码视频帧到潜在空间
        
        Args:
            x: 输入帧，shape (batch, channels, height, width)
        
        Returns:
            z: 潜在向量，shape (batch, latent_dim)
        """
        return self.encoder(x)


# ============================================================
# 2. 动作推断网络 (Action Inference Network)
# ============================================================

class ActionInference(nn.Module):
    """
    从连续帧的潜在表示推断隐式动作
    
    核心思想：z_t 和 z_{t+1} 之间的差异由动作 a_t 解释
    """
    
    def __init__(self, latent_dim: int = 64, action_dim: int = 8):
        super().__init__()
        self.action_dim = action_dim
        
        # 输入是两个潜在向量的拼接
        self.network = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """
        从状态变化推断动作
        
        Args:
            z_t: 当前帧的潜在表示
            z_next: 下一帧的潜在表示
        
        Returns:
            a: 推断的动作向量
        """
        # 拼接两个潜在向量
        combined = torch.cat([z_t, z_next], dim=-1)
        return self.network(combined)


# ============================================================
# 3. 动作条件生成器 (Action-Conditioned Generator)
# ============================================================

class ActionConditionedGenerator(nn.Module):
    """
    根据当前状态和动作生成下一状态的潜在表示
    
    这是 Genie 的核心：学习状态转移函数 z_{t+1} = f(z_t, a_t)
    """
    
    def __init__(self, latent_dim: int = 64, action_dim: int = 8, hidden_dim: int = 256):
        super().__init__()
        
        # 状态 - 动作融合网络
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 残差连接：保持信息流动
        self.residual = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """
        生成下一状态的潜在表示
        
        Args:
            z_t: 当前状态潜在向量
            a_t: 动作向量
        
        Returns:
            z_next: 预测的下一状态潜在向量
        """
        # 拼接状态和动作
        combined = torch.cat([z_t, a_t], dim=-1)
        
        # 通过融合网络
        delta = self.fusion(combined)
        
        # 残差连接
        z_next = self.residual(z_t) + delta
        
        return z_next


# ============================================================
# 4. 自回归解码器 (Autoregressive Decoder)
# ============================================================

class AutoregressiveDecoder(nn.Module):
    """
    将潜在表示解码为视频帧
    
    使用转置卷积进行上采样
    """
    
    def __init__(self, latent_dim: int = 64, output_channels: int = 3):
        super().__init__()
        
        # 从潜在向量到特征图
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(),
        )
        
        # 转置卷积上采样
        self.decoder = nn.Sequential(
            # 256x4x4 → 128x8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 128x8x8 → 64x16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 64x16x16 → 32x32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 32x32x32 → 3x64x64
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码潜在向量为视频帧
        
        Args:
            z: 潜在向量，shape (batch, latent_dim)
        
        Returns:
            frame: 生成的帧，shape (batch, channels, height, width)
        """
        # 初始投影
        x = self.initial(z)
        x = x.view(-1, 256, 4, 4)
        
        # 上采样解码
        frame = self.decoder(x)
        
        return frame


# ============================================================
# 5. Genie 完整模型
# ============================================================

class Genie(nn.Module):
    """
    Genie 生成式交互环境完整模型
    
    整合编码器、动作推断、生成器和解码器
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 8,
        input_channels: int = 3,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # 核心组件
        self.encoder = LatentEncoder(latent_dim, input_channels).to(device)
        self.action_inference = ActionInference(latent_dim, action_dim).to(device)
        self.generator = ActionConditionedGenerator(latent_dim, action_dim).to(device)
        self.decoder = AutoregressiveDecoder(latent_dim, input_channels).to(device)
        
        # 策略网络（用于智能体训练）
        self.policy = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # 动作范围 [-1, 1]
        ).to(device)
    
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """编码帧到潜在空间"""
        return self.encoder(frames)
    
    def infer_action(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """从状态变化推断动作"""
        return self.action_inference(z_t, z_next)
    
    def predict_next(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """预测下一状态"""
        return self.generator(z_t, a_t)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码潜在状态为帧"""
        return self.decoder(z)
    
    def get_action(self, z: torch.Tensor) -> torch.Tensor:
        """策略网络：从状态输出动作"""
        return self.policy(z)
    
    def forward_step(
        self,
        frame_t: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        单步前向传播
        
        Args:
            frame_t: 当前帧
            action: 可选的外部动作（如果为 None，使用策略网络）
        
        Returns:
            frame_next: 生成的下一帧
            z_next: 下一状态的潜在表示
            a_t: 使用的动作
        """
        # 编码当前帧
        z_t = self.encode(frame_t)
        
        # 获取动作
        if action is None:
            a_t = self.get_action(z_t)
        else:
            a_t = action
        
        # 预测下一状态
        z_next = self.predict_next(z_t, a_t)
        
        # 解码为帧
        frame_next = self.decode(z_next)
        
        return frame_next, z_next, a_t
    
    def generate_sequence(
        self,
        init_frame: torch.Tensor,
        num_steps: int,
        actions: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        自回归生成视频序列
        
        Args:
            init_frame: 初始帧
            num_steps: 生成步数
            actions: 可选的动作序列
        
        Returns:
            video: 生成的视频序列 (num_steps+1, channels, height, width)
        """
        frames = [init_frame]
        current_frame = init_frame
        
        for t in range(num_steps):
            # 获取动作
            if actions is not None and t < len(actions):
                action = actions[t]
            else:
                action = None
            
            # 生成下一帧
            next_frame, _, _ = self.forward_step(current_frame, action)
            frames.append(next_frame)
            current_frame = next_frame
        
        return torch.stack(frames, dim=0)


# ============================================================
# 6. 训练循环
# ============================================================

class GenieTrainer:
    """
    Genie 模型训练器
    
    训练目标：
    1. 重建损失：解码的帧应该接近原始帧
    2. 动作一致性：推断的动作应该能预测状态变化
    3. 预测损失：生成的下一帧应该接近真实下一帧
    """
    
    def __init__(self, model: Genie, lr: float = 1e-4, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def train_step(self, frame_batch: torch.Tensor) -> dict:
        """
        单步训练
        
        Args:
            frame_batch: 视频批次，shape (batch, seq_len, channels, height, width)
        
        Returns:
            losses: 各项损失值
        """
        self.optimizer.zero_grad()
        
        batch_size, seq_len = frame_batch.shape[:2]
        total_loss = 0
        
        # 遍历时间步
        for t in range(seq_len - 1):
            frame_t = frame_batch[:, t].to(self.device)
            frame_next = frame_batch[:, t + 1].to(self.device)
            
            # 编码
            z_t = self.model.encode(frame_t)
            z_next_true = self.model.encode(frame_next)
            
            # 推断动作
            a_inferred = self.model.infer_action(z_t, z_next_true)
            
            # 预测下一状态
            z_next_pred = self.model.predict_next(z_t, a_inferred)
            
            # 解码
            frame_next_pred = self.model.decode(z_next_pred)
            
            # 重建损失
            reconstruction_loss = self.mse_loss(frame_next_pred, frame_next)
            
            # 潜在空间预测损失
            latent_loss = self.mse_loss(z_next_pred, z_next_true)
            
            # 动作正则化（鼓励小动作）
            action_reg = torch.mean(a_inferred ** 2)
            
            # 总损失
            step_loss = reconstruction_loss + 0.5 * latent_loss + 0.1 * action_reg
            total_loss = total_loss + step_loss
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'latent_loss': latent_loss.item(),
            'action_reg': action_reg.item()
        }
    
    def train(
        self,
        dataloader,
        num_epochs: int,
        log_interval: int = 10
    ) -> List[dict]:
        """
        完整训练循环
        
        Args:
            dataloader: 数据加载器
            num_epochs: 训练轮数
            log_interval: 日志间隔
        
        Returns:
            history: 训练历史记录
        """
        history = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                
                losses = self.train_step(batch)
                epoch_losses.append(losses)
            
            # 计算平均损失
            avg_losses = {
                k: np.mean([l[k] for l in epoch_losses])
                for k in epoch_losses[0].keys()
            }
            avg_losses['epoch'] = epoch
            
            history.append(avg_losses)
            
            if epoch % log_interval == 0:
                print(f"Epoch {epoch}: Total Loss = {avg_losses['total_loss']:.4f}")
        
        return history


# ============================================================
# 7. 简单环境示例
# ============================================================

class SimpleGridEnvironment:
    """
    简单的网格环境用于演示
    
    智能体在网格中移动，生成对应的视觉帧
    """
    
    def __init__(self, grid_size: int = 8, num_objects: int = 3):
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.agent_pos = np.array([grid_size // 2, grid_size // 2])
        
        # 随机放置物体
        self.objects = []
        for _ in range(num_objects):
            obj_pos = np.random.randint(0, grid_size, size=2)
            obj_color = np.random.rand(3)
            self.objects.append({'pos': obj_pos, 'color': obj_color})
    
    def reset(self) -> np.ndarray:
        """重置环境并返回初始帧"""
        self.agent_pos = np.array([self.grid_size // 2, self.grid_size // 2])
        return self.render()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        执行动作
        
        Args:
            action: 动作向量 [dx, dy]，范围 [-1, 1]
        
        Returns:
            frame: 新帧
            reward: 奖励
            done: 是否结束
        """
        # 将动作转换为网格移动
        dx = int(np.round(action[0] * 2))
        dy = int(np.round(action[1] * 2))
        
        # 更新位置（带边界检查）
        self.agent_pos[0] = np.clip(self.agent_pos[0] + dx, 0, self.grid_size - 1)
        self.agent_pos[1] = np.clip(self.agent_pos[1] + dy, 0, self.grid_size - 1)
        
        # 检查是否碰到物体
        reward = 0
        for obj in self.objects:
            if np.array_equal(self.agent_pos, obj['pos']):
                reward = 1.0
        
        return self.render(), reward, False
    
    def render(self) -> np.ndarray:
        """渲染当前状态为图像"""
        # 创建空白画布
        img = np.ones((self.grid_size, self.grid_size, 3), dtype=np.float32) * 0.9
        
        # 绘制物体
        for obj in self.objects:
            x, y = obj['pos']
            img[x, y] = obj['color']
        
        # 绘制智能体（红色）
        x, y = self.agent_pos
        img[x, y] = [1.0, 0.3, 0.3]
        
        # 调整大小到 64x64
        from scipy.ndimage import zoom
        img = zoom(img, (64 / self.grid_size, 64 / self.grid_size, 1), order=0)
        
        return np.clip(img, 0, 1)


# ============================================================
# 8. 演示函数
# ============================================================

def generate_demo_video(model: Genie, num_frames: int = 16) -> np.ndarray:
    """
    生成演示视频
    
    Args:
        model: 训练好的 Genie 模型
        num_frames: 生成帧数
    
    Returns:
        video: 视频序列
    """
    model.eval()
    
    # 创建随机初始帧
    init_frame = torch.randn(1, 3, 64, 64).to(model.device) * 0.5 + 0.5
    init_frame = torch.clamp(init_frame, 0, 1)
    
    # 生成序列
    with torch.no_grad():
        video = model.generate_sequence(init_frame, num_frames)
    
    return video.cpu().numpy()


def train_agent_in_genie(
    model: Genie,
    env: SimpleGridEnvironment,
    num_episodes: int = 100,
    max_steps: int = 50
) -> List[float]:
    """
    在 Genie 生成的环境中训练智能体
    
    使用简单的 REINFORCE 算法
    """
    rewards_history = []
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    
    for episode in range(num_episodes):
        # 重置环境
        frame = env.reset()
        frame_tensor = torch.FloatTensor(frame).permute(2, 0, 1).unsqueeze(0).to(model.device)
        
        episode_rewards = []
        log_probs = []
        
        for step in range(max_steps):
            # 编码当前状态
            with torch.no_grad():
                z = model.encode(frame_tensor)
            
            # 选择动作
            action = model.get_action(z)
            log_prob = torch.log(torch.clamp(1 - action.abs(), 1e-6))
            log_probs.append(log_prob)
            
            # 执行动作
            action_np = action[0].cpu().numpy()
            next_frame, reward, done = env.step(action_np)
            episode_rewards.append(reward)
            
            # 更新
            frame_tensor = torch.FloatTensor(next_frame).permute(2, 0, 1).unsqueeze(0).to(model.device)
            
            if done:
                break
        
        # 计算回报
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(model.device)
        
        # 策略梯度更新
        if len(log_probs) > 0 and len(returns) > 0:
            policy_loss = []
            for lp, ret in zip(log_probs, returns):
                policy_loss.append(-lp * ret)
            
            optimizer.zero_grad()
            loss = torch.cat(policy_loss).mean()
            loss.backward()
            optimizer.step()
        
        total_reward = sum(episode_rewards)
        rewards_history.append(total_reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}")
    
    return rewards_history


# ============================================================
# 9. 主函数 - 完整演示
# ============================================================

def main():
    """
    主函数：演示 Genie 的完整流程
    """
    print("=" * 60)
    print("Genie 生成式交互环境 - 教学演示")
    print("=" * 60)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备：{device}")
    
    # 1. 创建模型
    print("\n[1] 创建 Genie 模型...")
    model = Genie(
        latent_dim=64,
        action_dim=8,
        input_channels=3,
        device=device
    )
    
    # 打印模型结构
    print(f"    - 潜在空间维度：{model.latent_dim}")
    print(f"    - 动作空间维度：{model.action_dim}")
    print(f"    - 参数量：{sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 创建训练数据（模拟视频序列）
    print("\n[2] 生成训练数据...")
    batch_size = 4
    seq_len = 8
    
    # 模拟视频批次：(batch, seq_len, channels, height, width)
    video_batch = torch.randn(batch_size, seq_len, 3, 64, 64).to(device) * 0.3 + 0.5
    video_batch = torch.clamp(video_batch, 0, 1)
    print(f"    - 批次大小：{batch_size}")
    print(f"    - 序列长度：{seq_len}")
    
    # 3. 创建训练器并训练
    print("\n[3] 训练模型...")
    trainer = GenieTrainer(model, lr=1e-4, device=device)
    
    # 模拟数据加载器
    class MockDataLoader:
        def __init__(self, data, num_batches=5):
            self.data = data
            self.num_batches = num_batches
        
        def __iter__(self):
            for _ in range(self.num_batches):
                yield self.data
    
    dataloader = MockDataLoader(video_batch)
    history = trainer.train(dataloader, num_epochs=20, log_interval=5)
    
    # 4. 生成演示视频
    print("\n[4] 生成演示视频...")
    with torch.no_grad():
        demo_video = generate_demo_video(model, num_frames=16)
    print(f"    - 生成视频形状：{demo_video.shape}")
    print(f"    - 帧数：{demo_video.shape[0]}")
    
    # 5. 在简单环境中训练智能体
    print("\n[5] 在简单环境中训练智能体...")
    env = SimpleGridEnvironment(grid_size=8, num_objects=3)
    rewards = train_agent_in_genie(model, env, num_episodes=50, max_steps=30)
    
    print(f"\n    - 最终平均奖励：{np.mean(rewards[-10:]):.2f}")
    
    # 6. 可视化（如果可能）
    print("\n[6] 训练完成！")
    print("=" * 60)
    print("提示：在实际使用中，可以：")
    print("  - 使用真实视频数据训练")
    print("  - 保存和加载模型权重")
    print("  - 可视化生成的视频")
    print("  - 在更复杂的环境中训练智能体")
    print("=" * 60)
    
    return model, history, demo_video, rewards


# ============================================================
# 10. 单元测试
# ============================================================

def test_components():
    """测试各个组件的基本功能"""
    print("\n运行组件测试...")
    
    device = 'cpu'
    
    # 测试编码器
    encoder = LatentEncoder(latent_dim=64).to(device)
    test_input = torch.randn(2, 3, 64, 64).to(device)
    z = encoder(test_input)
    assert z.shape == (2, 64), f"编码器输出形状错误：{z.shape}"
    print("  ✓ 编码器测试通过")
    
    # 测试动作推断
    action_inf = ActionInference(latent_dim=64, action_dim=8).to(device)
    z1 = torch.randn(2, 64).to(device)
    z2 = torch.randn(2, 64).to(device)
    a = action_inf(z1, z2)
    assert a.shape == (2, 8), f"动作推断输出形状错误：{a.shape}"
    print("  ✓ 动作推断测试通过")
    
    # 测试生成器
    generator = ActionConditionedGenerator(latent_dim=64, action_dim=8).to(device)
    z = torch.randn(2, 64).to(device)
    a = torch.randn(2, 8).to(device)
    z_next = generator(z, a)
    assert z_next.shape == (2, 64), f"生成器输出形状错误：{z_next.shape}"
    print("  ✓ 生成器测试通过")
    
    # 测试解码器
    decoder = AutoregressiveDecoder(latent_dim=64).to(device)
    z = torch.randn(2, 64).to(device)
    frame = decoder(z)
    assert frame.shape == (2, 3, 64, 64), f"解码器输出形状错误：{frame.shape}"
    print("  ✓ 解码器测试通过")
    
    # 测试完整模型
    model = Genie(latent_dim=64, action_dim=8, device=device)
    frame = torch.randn(2, 3, 64, 64).to(device)
    frame_next, z_next, action = model.forward_step(frame)
    assert frame_next.shape == (2, 3, 64, 64)
    assert z_next.shape == (2, 64)
    assert action.shape == (2, 8)
    print("  ✓ 完整模型测试通过")
    
    print("所有测试通过！✓")


if __name__ == "__main__":
    # 首先运行测试
    test_components()
    
    # 然后运行主演示
    model, history, video, rewards = main()
    
    # 保存训练历史（可选）
    import json
    with open('training_history.json', 'w', encoding='utf-8') as f:
        # 转换 numpy 类型为 Python 原生类型
        serializable_history = []
        for h in history:
            serializable_history.append({
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in h.items()
            })
        json.dump(serializable_history, f, indent=2, ensure_ascii=False)
    
    print("\n训练历史已保存到 training_history.json")
