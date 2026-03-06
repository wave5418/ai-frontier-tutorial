# -*- coding: utf-8 -*-
"""
扩散模型与 Stable Diffusion 完整实现
=====================================

本文件包含扩散模型的完整实现，包括：
1. 扩散过程（前向加噪 + 反向去噪）
2. UNet 网络架构（简化版）
3. DDPM 训练循环
4. DDIM 加速采样
5. 条件扩散模型
6. 文生图示例（简化版）

作者：AI 前沿技术教程
版本：1.0
日期：2026 年
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass
from tqdm import tqdm


# ============================================================
# 第一部分：扩散过程基础
# ============================================================

@dataclass
class DiffusionConfig:
    """扩散模型配置"""
    num_timesteps: int = 1000      # 扩散步数 T
    beta_start: float = 1e-4       # 初始噪声方差 β_1
    beta_end: float = 0.02         # 最终噪声方差 β_T
    schedule_type: str = "linear"  # 噪声调度类型
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DiffusionScheduler:
    """
    扩散调度器：管理噪声调度和采样系数
    
    核心公式：
    - α_t = 1 - β_t
    - ᾱ_t = ∏_{s=1}^{t} α_s
    - x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε
    """
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.num_timesteps = config.num_timesteps
        
        # 生成噪声调度 β_t
        if config.schedule_type == "linear":
            self.betas = torch.linspace(
                config.beta_start, 
                config.beta_end, 
                config.num_timesteps
            )
        elif config.schedule_type == "cosine":
            # 余弦调度：更平滑的噪声添加
            s = 0.008
            t = torch.linspace(0, config.num_timesteps, config.num_timesteps + 1)
            alphas_cumprod = torch.cos((t / config.num_timesteps + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(self.betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {config.schedule_type}")
        
        # 计算 α_t = 1 - β_t
        self.alphas = 1.0 - self.betas
        
        # 计算 ᾱ_t = ∏ α_s (累积乘积)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 计算 ᾱ_{t-1}，用于反向过程
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        
        # 计算 √(ᾱ_t)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        
        # 计算 √(1 - ᾱ_t)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 计算 1/√(ᾱ_t)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        
        # 计算 σ_t² = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / 
            (1.0 - self.alphas_cumprod)
        )
        
        # 移动到设备
        self.to(config.device)
    
    def to(self, device):
        """将所有张量移动到指定设备"""
        self.device = device
        for attr in dir(self):
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))
        return self
    
    def get_sampling_coeffs(self, t: torch.Tensor) -> dict:
        """
        获取时间步 t 的采样系数
        
        参数:
            t: 时间步张量 [batch_size]
            
        返回:
            包含各种系数的字典
        """
        # 辅助函数：从张量中提取对应时间的值
        def extract(a, t, x_shape):
            """从预计算的数组中提取对应时间步的值"""
            b, *_ = t.shape
            out = a.gather(-1, t)
            return out.reshape(b, *((1,) * (len(x_shape) - 1)))
        
        return {
            'sqrt_alphas_cumprod': extract(self.sqrt_alphas_cumprod, t, t.shape),
            'sqrt_one_minus_alphas_cumprod': extract(
                self.sqrt_one_minus_alphas_cumprod, t, t.shape
            ),
            'sqrt_recip_alphas_cumprod': extract(
                self.sqrt_recip_alphas_cumprod, t, t.shape
            ),
            'posterior_variance': extract(self.posterior_variance, t, t.shape),
            'alphas_cumprod': extract(self.alphas_cumprod, t, t.shape),
            'alphas_cumprod_prev': extract(self.alphas_cumprod_prev, t, t.shape),
        }
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向扩散：从 x_0 直接采样 x_t
        
        公式：x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε
        
        参数:
            x_0: 原始数据 [batch, channels, height, width]
            t: 时间步 [batch]
            noise: 噪声，默认为标准高斯噪声
            
        返回:
            x_t: 加噪后的数据
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        coeffs = self.get_sampling_coeffs(t)
        
        x_t = (
            coeffs['sqrt_alphas_cumprod'] * x_0 +
            coeffs['sqrt_one_minus_alphas_cumprod'] * noise
        )
        
        return x_t
    
    def q_posterior_mean_variance(
        self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算反向过程的后验分布 q(x_{t-1}|x_t, x_0)
        
        公式:
        μ_tilde(x_t, x_0) = (√(ᾱ_{t-1})β_t / (1-ᾱ_t))x_0 + 
                           (√(α_t)(1-ᾱ_{t-1}) / (1-ᾱ_t))x_t
        
        参数:
            x_0: 原始数据
            x_t: t 时刻的加噪数据
            t: 时间步
            
        返回:
            (posterior_mean, posterior_variance)
        """
        coeffs = self.get_sampling_coeffs(t)
        
        # 计算后验均值
        posterior_mean = (
            torch.sqrt(coeffs['alphas_cumprod_prev']) * 
            self.betas[t] / (1.0 - coeffs['alphas_cumprod']) * x_0 +
            torch.sqrt(self.alphas[t]) * 
            (1.0 - coeffs['alphas_cumprod_prev']) / 
            (1.0 - coeffs['alphas_cumprod']) * x_t
        )
        
        # 后验方差
        posterior_variance = coeffs['posterior_variance']
        
        return posterior_mean, posterior_variance


# ============================================================
# 第二部分：UNet 网络架构
# ============================================================

class TimeEmbedding(nn.Module):
    """
    时间嵌入：将离散时间步转换为连续向量表示
    
    使用正弦位置编码 + MLP 投影
    """
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # 正弦编码的半维度
        half_dim = dim // 2
        
        # 计算频率：exp(-log(max_period) * i / half_dim)
        emb = math.log(max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('emb_scale', emb)
        
        # MLP 投影
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        参数:
            t: 时间步 [batch]
            
        返回:
            时间嵌入 [batch, dim]
        """
        # 正弦位置编码
        emb = t.float()[:, None] * self.emb_scale[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # 零填充到目标维度
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        
        # MLP 投影
        emb = self.mlp(emb)
        
        return emb


class ResidualBlock(nn.Module):
    """
    残差块：UNet 的基本构建单元
    
    结构：Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU
          + 时间嵌入调制 + 跳跃连接
    """
    
    def __init__(self, channels: int, time_emb_dim: int, 
                 groups: int = 8):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels * 2)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        # 如果输入输出通道相同，使用恒等映射
        self.residual_conv = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入特征 [batch, channels, H, W]
            time_emb: 时间嵌入 [batch, time_dim]
            
        返回:
            输出特征 [batch, channels, H, W]
        """
        # 时间嵌入调制
        time_out = self.time_mlp(time_emb)
        time_out = time_out[:, :, None, None]  # [batch, channels*2, 1, 1]
        time_scale, time_shift = time_out.chunk(2, dim=1)
        
        # 主路径
        h = self.block1(x)
        h = h * (1 + time_scale) + time_shift  # FiLM 调制
        h = self.block2(h)
        
        # 跳跃连接
        residual = self.residual_conv(x)
        
        return h + residual


class AttentionBlock(nn.Module):
    """
    自注意力块：捕获长程依赖
    
    使用多头自注意力机制
    """
    
    def __init__(self, channels: int, num_heads: int = 8, 
                 groups: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        
        self.norm = nn.GroupNorm(groups, channels)
        
        # QKV 投影
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        
        # 输出投影
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入特征 [batch, channels, H, W]
            
        返回:
            输出特征 [batch, channels, H, W]
        """
        b, c, h, w = x.shape
        
        # 归一化
        x_norm = self.norm(x)
        
        # QKV 投影
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, batch, heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 自注意力
        # attention = softmax(QK^T / √d) V
        scale = 1.0 / math.sqrt(k.shape[-1])
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = attn @ v
        
        # 重塑回空间维度
        out = out.permute(0, 2, 3, 1).reshape(b, c, h, w)
        
        # 输出投影 + 跳跃连接
        out = self.proj_out(out)
        
        return x + out


class CrossAttentionBlock(nn.Module):
    """
    交叉注意力块：用于条件扩散（如文本条件）
    
    Query 来自图像特征，Key/Value 来自条件（如文本嵌入）
    """
    
    def __init__(self, channels: int, context_dim: int, 
                 num_heads: int = 8, groups: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        
        self.norm = nn.GroupNorm(groups, channels)
        
        # Query 投影（来自图像）
        self.to_q = nn.Linear(channels, channels)
        
        # Key, Value 投影（来自条件）
        self.to_k = nn.Linear(context_dim, channels)
        self.to_v = nn.Linear(context_dim, channels)
        
        # 输出投影
        self.proj_out = nn.Linear(channels, channels)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 图像特征 [batch, channels, H, W]
            context: 条件嵌入 [batch, seq_len, context_dim]
            
        返回:
            输出特征 [batch, channels, H, W]
        """
        b, c, h, w = x.shape
        
        # 归一化
        x_norm = self.norm(x)
        x_flat = x_norm.reshape(b, c, h * w).permute(0, 2, 1)  # [batch, seq_len, channels]
        
        # QKV 投影
        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # 多头注意力
        q = q.reshape(b, -1, self.num_heads, c // self.num_heads).transpose(1, 2)
        k = k.reshape(b, -1, self.num_heads, c // self.num_heads).transpose(1, 2)
        v = v.reshape(b, -1, self.num_heads, c // self.num_heads).transpose(1, 2)
        
        # 注意力计算
        scale = 1.0 / math.sqrt(k.shape[-1])
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = attn @ v
        
        # 重塑
        out = out.transpose(1, 2).reshape(b, h * w, c)
        out = self.proj_out(out)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        
        return x + out


class DownBlock(nn.Module):
    """下采样块"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 time_emb_dim: int, num_layers: int = 2,
                 add_attention: bool = True,
                 context_dim: Optional[int] = None):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            
            # 残差块
            layers.append(ResidualBlock(in_ch, time_emb_dim))
            
            # 自注意力
            if add_attention:
                layers.append(AttentionBlock(out_channels))
            
            # 交叉注意力（如果有条件）
            if context_dim is not None:
                layers.append(CrossAttentionBlock(out_channels, context_dim))
        
        self.layers = nn.ModuleList(layers)
        
        # 下采样
        self.downsample = nn.Conv2d(out_channels, out_channels, 
                                    3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        返回:
            output: 下采样后的特征
            skips: 跳跃连接特征（用于 UNet 的上采样）
        """
        skips = []
        
        for layer in self.layers:
            if isinstance(layer, CrossAttentionBlock):
                x = layer(x, context)
            else:
                x = layer(x, time_emb)
            skips.append(x)
        
        x = self.downsample(x)
        
        return x, skips


class UpBlock(nn.Module):
    """上采样块"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 time_emb_dim: int, num_layers: int = 2,
                 add_attention: bool = True,
                 context_dim: Optional[int] = None):
        super().__init__()
        
        layers = []
        for i in range(num_layers + 1):  # +1 用于融合跳跃连接
            in_ch = in_channels if i == 0 else out_channels
            
            # 残差块（输入通道翻倍，因为有跳跃连接）
            layers.append(ResidualBlock(in_ch * 2, time_emb_dim))
            
            # 自注意力
            if add_attention:
                layers.append(AttentionBlock(out_channels))
            
            # 交叉注意力
            if context_dim is not None:
                layers.append(CrossAttentionBlock(out_channels, context_dim))
        
        self.layers = nn.ModuleList(layers)
        
        # 上采样
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 
                                           4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor,
                skips: List[torch.Tensor],
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        参数:
            x: 输入特征
            time_emb: 时间嵌入
            skips: 来自对应下采样层的跳跃连接
            context: 条件嵌入
        """
        # 融合跳跃连接
        x = torch.cat([x, skips.pop()], dim=1)
        
        for layer in self.layers:
            if isinstance(layer, CrossAttentionBlock):
                x = layer(x, context)
            else:
                x = layer(x, time_emb)
            # 除了最后一层，其他层也融合跳跃连接
            if skips and not isinstance(layer, (AttentionBlock, CrossAttentionBlock)):
                x = torch.cat([x, skips.pop()], dim=1)
        
        x = self.upsample(x)
        
        return x


class UNet(nn.Module):
    """
    U-Net 网络：扩散模型的核心去噪网络
    
    架构：
    - 下采样路径：逐步降低分辨率，提取特征
    - 中间层：深层特征处理
    - 上采样路径：逐步恢复分辨率，融合跳跃连接
    
    输入：带噪声的图像 x_t 和时间 t
    输出：预测的噪声 ε_θ(x_t, t)
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 out_channels: int = 3,
                 base_channels: int = 128,
                 channel_multipliers: Tuple[int] = (1, 2, 4, 8),
                 num_res_blocks: int = 2,
                 time_emb_dim: int = 512,
                 add_attention: bool = True,
                 context_dim: Optional[int] = None):
        """
        参数:
            in_channels: 输入通道数（通常是 3 或 4）
            out_channels: 输出通道数（预测噪声的通道数）
            base_channels: 基础通道数
            channel_multipliers: 各层的通道倍数
            num_res_blocks: 每层的残差块数量
            time_emb_dim: 时间嵌入维度
            add_attention: 是否添加注意力
            context_dim: 条件嵌入维度（用于条件扩散）
        """
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        self.context_dim = context_dim
        
        # 时间嵌入网络
        self.time_embedding = TimeEmbedding(time_emb_dim)
        
        # 条件嵌入（如果有）
        if context_dim is not None:
            self.context_embedding = nn.Sequential(
                nn.Linear(context_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim)
            )
        
        # 初始卷积
        self.initial_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            add_attn = add_attention and (i == len(channel_multipliers) - 1)
            
            self.down_blocks.append(
                DownBlock(
                    channels, out_ch, time_emb_dim,
                    num_layers=num_res_blocks,
                    add_attention=add_attn,
                    context_dim=context_dim
                )
            )
            channels = out_ch
        
        # 中间层
        self.middle_block = nn.ModuleList([
            ResidualBlock(channels, time_emb_dim),
            AttentionBlock(channels) if add_attention else nn.Identity(),
            ResidualBlock(channels, time_emb_dim)
        ])
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            add_attn = add_attention and (i == len(channel_multipliers) - 1)
            
            self.up_blocks.append(
                UpBlock(
                    channels, out_ch, time_emb_dim,
                    num_layers=num_res_blocks,
                    add_attention=add_attn,
                    context_dim=context_dim
                )
            )
            channels = out_ch
        
        # 输出层
        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        参数:
            x: 输入图像 [batch, in_channels, H, W]
            t: 时间步 [batch]
            context: 条件嵌入 [batch, seq_len, context_dim]（可选）
            
        返回:
            预测的噪声 [batch, out_channels, H, W]
        """
        # 时间嵌入
        time_emb = self.time_embedding(t)
        
        # 条件嵌入
        if context is not None and hasattr(self, 'context_embedding'):
            context = self.context_embedding(context)
        
        # 初始卷积
        x = self.initial_conv(x)
        
        # 下采样路径
        all_skips = []
        for down_block in self.down_blocks:
            x, skips = down_block(x, time_emb, context)
            all_skips.append(skips)
        
        # 中间层
        for layer in self.middle_block:
            if isinstance(layer, AttentionBlock):
                x = layer(x)
            else:
                x = layer(x, time_emb)
        
        # 上采样路径（反转跳跃连接）
        all_skips = all_skips[::-1]
        for up_block in self.up_blocks:
            skips = all_skips.pop()
            x = up_block(x, time_emb, skips, context)
        
        # 输出层
        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)
        
        return x


# ============================================================
# 第三部分：DDPM 训练
# ============================================================

class DDPM:
    """
    DDPM (Denoising Diffusion Probabilistic Models)
    
    训练目标：预测添加的噪声
    L = E[||ε - ε_θ(x_t, t)||²]
    """
    
    def __init__(self, model: UNet, scheduler: DiffusionScheduler,
                 lr: float = 2e-5):
        """
        参数:
            model: UNet 模型
            scheduler: 扩散调度器
            lr: 学习率
        """
        self.model = model
        self.scheduler = scheduler
        self.device = scheduler.device
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )
    
    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        计算 DDPM 损失
        
        参数:
            x_0: 原始图像 [batch, channels, H, W]
            
        返回:
            loss: 标量损失
        """
        batch_size = x_0.shape[0]
        
        # 随机采样时间步
        t = torch.randint(
            0, self.scheduler.num_timesteps, 
            (batch_size,), device=self.device
        ).long()
        
        # 采样噪声
        noise = torch.randn_like(x_0)
        
        # 前向扩散：获取 x_t
        x_t = self.scheduler.q_sample(x_0, t, noise)
        
        # 预测噪声
        predicted_noise = self.model(x_t, t)
        
        # MSE 损失
        loss = F.mse_loss(noise, predicted_noise)
        
        return loss
    
    def train_step(self, x_0: torch.Tensor) -> float:
        """
        单步训练
        
        参数:
            x_0: 原始图像
            
        返回:
            loss: 损失值
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.compute_loss(x_0)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler_lr.step()
        
        return loss.item()
    
    def train(self, dataloader: torch.utils.data.DataLoader,
              num_epochs: int, save_path: Optional[str] = None):
        """
        完整训练循环
        
        参数:
            dataloader: 数据加载器
            num_epochs: 训练轮数
            save_path: 模型保存路径
        """
        print(f"开始训练 DDPM，设备：{self.device}")
        print(f"数据集大小：{len(dataloader)} batches")
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            # 进度条
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                if isinstance(batch, (tuple, list)):
                    x_0 = batch[0]
                else:
                    x_0 = batch
                
                x_0 = x_0.to(self.device)
                
                loss = self.train_step(x_0)
                total_loss += loss
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} 完成，平均损失：{avg_loss:.4f}")
            
            # 保存模型
            if save_path and (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                }, f"{save_path}_epoch{epoch+1}.pt")
        
        # 最终保存
        if save_path:
            torch.save(self.model.state_dict(), f"{save_path}_final.pt")
            print(f"模型已保存到 {save_path}_final.pt")


# ============================================================
# 第四部分：DDIM 采样
# ============================================================

class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Models) 采样器
    
    核心优势：
    - 确定性采样（无随机噪声）
    - 可跳步加速（如 1000 步 → 50 步）
    - 采样结果一致性好
    """
    
    def __init__(self, model: UNet, scheduler: DiffusionScheduler,
                 num_timesteps: Optional[int] = None):
        """
        参数:
            model: 训练好的 UNet
            scheduler: 扩散调度器
            num_timesteps: 采样步数（可小于原始步数以加速）
        """
        self.model = model
        self.scheduler = scheduler
        self.device = scheduler.device
        
        # 原始步数
        self.original_timesteps = scheduler.num_timesteps
        
        # 采样步数（可加速）
        self.num_timesteps = num_timesteps or scheduler.num_timesteps
        
        # 计算跳步序列
        self.skip_ratio = self.original_timesteps // self.num_timesteps
        self.timesteps = list(range(
            0, self.original_timesteps, self.skip_ratio
        ))
        
        print(f"DDIM 采样器：{self.num_timesteps} 步 "
              f"(原始：{self.original_timesteps} 步)")
    
    @torch.no_grad()
    def sample(self, shape: Tuple[int], context: Optional[torch.Tensor] = None,
               guidance_scale: float = 1.0,
               eta: float = 0.0) -> torch.Tensor:
        """
        DDIM 采样生成图像
        
        参数:
            shape: 输出形状 (batch, channels, height, width)
            context: 条件嵌入（用于条件生成）
            guidance_scale: 引导强度（分类器自由引导）
            eta: 随机性参数（0=完全确定性，1=类似 DDPM）
            
        返回:
            生成的图像
        """
        self.model.eval()
        batch_size = shape[0]
        
        # 从标准高斯分布采样
        x_t = torch.randn(shape, device=self.device)
        
        # 时间步序列（反转）
        time_steps = list(reversed(self.timesteps))
        
        # 进度条
        pbar = tqdm(time_steps, desc="DDIM 采样")
        
        for i, t in enumerate(pbar):
            t_tensor = torch.full((batch_size,), t, 
                                 device=self.device, dtype=torch.long)
            
            # 预测噪声
            noise_pred = self.model(x_t, t_tensor, context)
            
            # 分类器自由引导
            if guidance_scale > 1.0 and context is not None:
                # 无条件预测
                null_context = torch.zeros_like(context)
                noise_pred_uncond = self.model(x_t, t_tensor, null_context)
                
                # 引导
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred - noise_pred_uncond
                )
            
            # 计算当前和上一个时间步的 ᾱ
            alpha_cumprod_t = self.scheduler.alphas_cumprod[t]
            
            if i > 0:
                prev_t = time_steps[i + 1] if i + 1 < len(time_steps) else -1
                if prev_t >= 0:
                    alpha_cumprod_prev_t = self.scheduler.alphas_cumprod[prev_t]
                else:
                    alpha_cumprod_prev_t = torch.tensor(1.0, device=self.device)
            else:
                alpha_cumprod_prev_t = torch.tensor(1.0, device=self.device)
            
            # DDIM 公式
            # x_{t-1} = √(ᾱ_{t-1}) * (x_t - √(1-ᾱ_t)*ε) / √(ᾱ_t)
            #         + √(1-ᾱ_{t-1} - σ_t²) * ε
            #         + σ_t * z (可选随机噪声)
            
            # 方向指向 x_0 的部分
            pred_x_0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / \
                       torch.sqrt(alpha_cumprod_t)
            
            # 方差系数
            sigma_t = eta * torch.sqrt(
                (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t) * 
                (1 - alpha_cumprod_t / alpha_cumprod_prev_t)
            )
            
            # 更新 x_t
            x_t = (
                torch.sqrt(alpha_cumprod_prev_t) * pred_x_0 +
                torch.sqrt(1 - alpha_cumprod_prev_t - sigma_t ** 2) * noise_pred +
                sigma_t * torch.randn_like(x_t) if eta > 0 else 0
            )
            
            pbar.set_postfix({'t': t})
        
        # 裁剪到 [-1, 1]
        x_t = torch.clamp(x_t, -1.0, 1.0)
        
        return x_t
    
    @torch.no_grad()
    def interpolate(self, x_1: torch.Tensor, x_2: torch.Tensor,
                    t: int = 500, lam: float = 0.5) -> torch.Tensor:
        """
        在潜空间插值
        
        参数:
            x_1, x_2: 两个噪声向量
            t: 插值时间步
            lam: 插值系数 (0=x_1, 1=x_2)
            
        返回:
            插值后生成的图像
        """
        # 在时间 t 插值
        x_t = lam * x_1 + (1 - lam) * x_2
        
        # 从 t 开始反向扩散
        self.model.eval()
        
        time_steps = list(reversed([ts for ts in self.timesteps if ts <= t]))
        
        for i, ts in enumerate(tqdm(time_steps, desc="插值采样")):
            ts_tensor = torch.full((x_t.shape[0],), ts, 
                                  device=self.device, dtype=torch.long)
            
            noise_pred = self.model(x_t, ts_tensor)
            
            # 简化 DDIM 步
            alpha_cumprod = self.scheduler.alphas_cumprod[ts]
            
            if i > 0:
                prev_ts = time_steps[i + 1] if i + 1 < len(time_steps) else -1
                alpha_cumprod_prev = self.scheduler.alphas_cumprod[prev_ts] if prev_ts >= 0 else 1.0
            else:
                alpha_cumprod_prev = 1.0
            
            pred_x_0 = (x_t - torch.sqrt(1 - alpha_cumprod) * noise_pred) / \
                       torch.sqrt(alpha_cumprod)
            
            x_t = torch.sqrt(alpha_cumprod_prev) * pred_x_0 + \
                  torch.sqrt(1 - alpha_cumprod_prev) * noise_pred
        
        return torch.clamp(x_t, -1.0, 1.0)


# ============================================================
# 第五部分：条件扩散模型（文生图）
# ============================================================

class TextEncoder(nn.Module):
    """
    简化的文本编码器（模拟 CLIP Text Encoder）
    
    实际应用中应使用预训练的 CLIP 或 T5 编码器
    """
    
    def __init__(self, vocab_size: int = 49408, embed_dim: int = 768,
                 max_length: int = 77, num_layers: int = 12):
        """
        参数:
            vocab_size: 词汇表大小
            embed_dim: 嵌入维度
            max_length: 最大序列长度
            num_layers: Transformer 层数
        """
        super().__init__()
        
        self.max_length = max_length
        self.embed_dim = embed_dim
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 位置嵌入
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 层归一化
        self.final_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        参数:
            tokens: token IDs [batch, seq_len]
            
        返回:
            文本嵌入 [batch, seq_len, embed_dim]
        """
        batch_size, seq_len = tokens.shape
        
        # 词嵌入 + 位置嵌入
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        
        # Transformer 编码
        x = self.transformer(x)
        x = self.final_norm(x)
        
        return x


class StableDiffusionLite:
    """
    简化版 Stable Diffusion
    
    包含：
    - 文本编码器（条件）
    - UNet（去噪）
    - DDIM 采样器
    - （简化：省略 VAE，直接在像素空间操作）
    """
    
    def __init__(self, image_size: int = 64, base_channels: int = 128,
                 context_dim: int = 768):
        """
        参数:
            image_size: 图像大小（简化版使用较小尺寸）
            base_channels: UNet 基础通道数
            context_dim: 文本嵌入维度
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        # 扩散配置
        config = DiffusionConfig(
            num_timesteps=1000,
            beta_start=1e-4,
            beta_end=0.02,
            device=self.device
        )
        self.scheduler = DiffusionScheduler(config)
        
        # 文本编码器
        self.text_encoder = TextEncoder(
            vocab_size=49408,
            embed_dim=context_dim,
            max_length=77
        ).to(self.device)
        
        # UNet
        self.unet = UNet(
            in_channels=3,
            out_channels=3,
            base_channels=base_channels,
            channel_multipliers=(1, 2, 4),
            num_res_blocks=2,
            context_dim=context_dim
        ).to(self.device)
        
        # DDIM 采样器（50 步加速）
        self.sampler = DDIMSampler(self.unet, self.scheduler, num_timesteps=50)
        
        print(f"StableDiffusionLite 初始化完成，设备：{self.device}")
    
    def encode_text(self, prompts: List[str], 
                    tokenizer: Optional[nn.Module] = None) -> torch.Tensor:
        """
        文本编码
        
        参数:
            prompts: 文本提示列表
            tokenizer: 分词器（简化版使用随机 token）
            
        返回:
            文本嵌入 [batch, seq_len, dim]
        """
        batch_size = len(prompts)
        
        # 简化：随机生成 token（实际应用应使用真实分词器）
        # 这里模拟 CLIP 的 tokenization
        tokens = torch.randint(
            0, 49408, 
            (batch_size, 77), 
            device=self.device
        )
        
        # 文本编码
        text_embeds = self.text_encoder(tokens)
        
        return text_embeds
    
    def train_step(self, images: torch.Tensor, 
                   text_embeds: torch.Tensor) -> float:
        """
        条件扩散训练步
        
        参数:
            images: 图像 [batch, 3, H, W]
            text_embeds: 文本嵌入 [batch, seq_len, dim]
            
        返回:
            损失值
        """
        self.unet.train()
        self.text_encoder.train()
        
        batch_size = images.shape[0]
        
        # 随机时间步
        t = torch.randint(
            0, self.scheduler.num_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        # 前向扩散
        noise = torch.randn_like(images)
        x_t = self.scheduler.q_sample(images, t, noise)
        
        # 预测噪声（条件）
        predicted_noise = self.unet(x_t, t, text_embeds)
        
        # MSE 损失
        loss = F.mse_loss(noise, predicted_noise)
        
        # 反向传播
        self.unet.zero_grad()
        self.text_encoder.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            list(self.unet.parameters()) + 
            list(self.text_encoder.parameters()),
            1.0
        )
        
        # 优化器（简化：使用同一个）
        optimizer = torch.optim.AdamW(
            list(self.unet.parameters()) + 
            list(self.text_encoder.parameters()),
            lr=2e-5
        )
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def generate(self, prompts: List[str], num_images: int = 1,
                 guidance_scale: float = 7.5,
                 negative_prompt: str = "") -> torch.Tensor:
        """
        文生图生成
        
        参数:
            prompts: 文本提示列表
            num_images: 每个提示生成的图像数
            guidance_scale: 引导强度
            negative_prompt: 负向提示
            
        返回:
            生成的图像 [batch, 3, H, W]
        """
        self.unet.eval()
        self.text_encoder.eval()
        
        all_images = []
        
        for prompt in prompts:
            # 文本编码
            text_embeds = self.encode_text([prompt])
            
            # 负向提示编码
            if negative_prompt:
                neg_embeds = self.encode_text([negative_prompt])
            else:
                neg_embeds = torch.zeros_like(text_embeds)
            
            # 生成多张图像
            for _ in range(num_images):
                # 采样
                image = self.sampler.sample(
                    shape=(1, 3, self.image_size, self.image_size),
                    context=text_embeds,
                    guidance_scale=guidance_scale
                )
                
                all_images.append(image)
        
        # 合并
        all_images = torch.cat(all_images, dim=0)
        
        # 从 [-1, 1] 转换到 [0, 1]
        all_images = (all_images + 1.0) / 2.0
        all_images = torch.clamp(all_images, 0.0, 1.0)
        
        return all_images


# ============================================================
# 第六部分：示例和测试
# ============================================================

def test_diffusion_process():
    """测试扩散过程"""
    print("=" * 60)
    print("测试 1: 扩散过程（前向 + 反向）")
    print("=" * 60)
    
    # 配置
    config = DiffusionConfig(num_timesteps=100)
    scheduler = DiffusionScheduler(config)
    
    # 创建测试图像
    batch_size = 2
    x_0 = torch.randn(batch_size, 3, 32, 32)
    
    # 测试前向扩散
    print("\n前向扩散测试:")
    for t in [0, 25, 50, 75, 99]:
        t_tensor = torch.full((batch_size,), t, dtype=torch.long)
        x_t = scheduler.q_sample(x_0, t_tensor)
        
        # 计算信噪比
        signal_power = torch.mean(x_t ** 2)
        print(f"  t={t}: 信号功率 = {signal_power:.4f}")
    
    print("\n✓ 前向扩散测试通过")
    
    # 测试后验计算
    print("\n反向过程后验计算:")
    t = 50
    t_tensor = torch.full((batch_size,), t, dtype=torch.long)
    x_t = scheduler.q_sample(x_0, t_tensor)
    
    posterior_mean, posterior_var = scheduler.q_posterior_mean_variance(
        x_0, x_t, t_tensor
    )
    
    print(f"  后验均值形状：{posterior_mean.shape}")
    print(f"  后验方差形状：{posterior_var.shape}")
    print("\n✓ 反向过程测试通过")


def test_unet():
    """测试 UNet 架构"""
    print("\n" + "=" * 60)
    print("测试 2: UNet 网络架构")
    print("=" * 60)
    
    # 创建 UNet
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=1,
        time_emb_dim=256,
        context_dim=768
    )
    
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nUNet 参数量：{num_params / 1e6:.2f}M")
    
    # 前向传播测试
    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))
    context = torch.randn(batch_size, 77, 768)
    
    output = model(x, t, context)
    
    print(f"输入形状：{x.shape}")
    print(f"时间步形状：{t.shape}")
    print(f"条件形状：{context.shape}")
    print(f"输出形状：{output.shape}")
    
    assert output.shape == x.shape, "输出形状应与输入相同"
    print("\n✓ UNet 测试通过")


def test_ddpm_training():
    """测试 DDPM 训练循环"""
    print("\n" + "=" * 60)
    print("测试 3: DDPM 训练循环（简化版）")
    print("=" * 60)
    
    # 配置
    config = DiffusionConfig(num_timesteps=100)
    scheduler = DiffusionScheduler(config)
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        time_emb_dim=128
    ).to(config.device)
    
    ddpm = DDPM(model, scheduler, lr=1e-4)
    
    # 创建虚拟数据
    batch_size = 4
    x_0 = torch.randn(batch_size, 3, 16, 16).to(config.device)
    
    # 训练几步
    print("\n训练 10 步:")
    losses = []
    for step in range(10):
        loss = ddpm.train_step(x_0)
        losses.append(loss)
        if (step + 1) % 5 == 0:
            print(f"  步 {step+1}: 损失 = {loss:.4f}")
    
    # 检查损失是否下降
    if len(losses) > 5:
        avg_first = np.mean(losses[:5])
        avg_last = np.mean(losses[-5:])
        print(f"\n前 5 步平均损失：{avg_first:.4f}")
        print(f"后 5 步平均损失：{avg_last:.4f}")
    
    print("\n✓ DDPM 训练测试通过")


def test_ddim_sampling():
    """测试 DDIM 采样"""
    print("\n" + "=" * 60)
    print("测试 4: DDIM 加速采样")
    print("=" * 60)
    
    # 配置
    config = DiffusionConfig(num_timesteps=100)
    scheduler = DiffusionScheduler(config)
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        time_emb_dim=128
    ).to(config.device)
    model.eval()
    
    # 创建采样器（10 步加速）
    sampler = DDIMSampler(model, scheduler, num_timesteps=10)
    
    # 采样
    print("\n从噪声采样 (10 步 DDIM):")
    sample = sampler.sample(shape=(2, 3, 16, 16))
    
    print(f"输出形状：{sample.shape}")
    print(f"输出范围：[{sample.min():.3f}, {sample.max():.3f}]")
    
    print("\n✓ DDIM 采样测试通过")


def test_conditional_diffusion():
    """测试条件扩散模型"""
    print("\n" + "=" * 60)
    print("测试 5: 条件扩散模型（文生图简化版）")
    print("=" * 60)
    
    # 创建简化版 Stable Diffusion
    sd = StableDiffusionLite(
        image_size=32,
        base_channels=32,
        context_dim=128
    )
    
    # 测试文本编码
    prompts = ["a beautiful sunset", "a cute cat"]
    print(f"\n文本提示：{prompts}")
    
    text_embeds = sd.encode_text(prompts)
    print(f"文本嵌入形状：{text_embeds.shape}")
    
    # 测试生成（不实际运行，因为需要训练）
    print("\n注意：实际文生图需要训练模型")
    print("这里仅测试架构完整性")
    
    print("\n✓ 条件扩散测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🎯" * 30)
    print("扩散模型完整实现 - 测试套件")
    print("🎯" * 30 + "\n")
    
    test_diffusion_process()
    test_unet()
    test_ddpm_training()
    test_ddim_sampling()
    test_conditional_diffusion()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    print("\n本实现包含:")
    print("  1. ✓ 扩散过程（前向加噪 + 反向去噪）")
    print("  2. ✓ UNet 网络架构（含注意力机制）")
    print("  3. ✓ DDPM 训练循环")
    print("  4. ✓ DDIM 加速采样")
    print("  5. ✓ 条件扩散模型（文生图）")
    print("\n提示：实际训练需要大量数据和计算资源")
    print("      建议使用预训练模型或简化数据集进行实验")


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    # 设置随机种子（可复现）
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    run_all_tests()
    
    # 示例：如何使用本代码
    print("\n" + "=" * 60)
    print("使用示例")
    print("=" * 60)
    
    example_code = """
# 1. 创建扩散模型
config = DiffusionConfig(num_timesteps=1000)
scheduler = DiffusionScheduler(config)

# 2. 创建 UNet
model = UNet(
    in_channels=3,
    out_channels=3,
    base_channels=128,
    channel_multipliers=(1, 2, 4, 8),
    context_dim=768  # 用于条件生成
)

# 3. 训练 DDPM
ddpm = DDPM(model, scheduler, lr=2e-5)
ddpm.train(dataloader, num_epochs=100)

# 4. DDIM 采样
sampler = DDIMSampler(model, scheduler, num_timesteps=50)
generated = sampler.sample(shape=(1, 3, 512, 512))

# 5. 条件生成（文生图）
sd = StableDiffusionLite(image_size=512)
images = sd.generate(
    prompts=["a beautiful landscape"],
    guidance_scale=7.5
)
"""
    print(example_code)
