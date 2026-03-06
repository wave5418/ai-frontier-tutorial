"""
video_pred.py
第 14 章 视频预测 - 代码实现

包含：
1. 视频预测模型实现
2. 时空注意力机制
3. 多帧预测
4. 视频生成示例

作者：AI 前沿技术教程
日期：2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
# 1. 视频预测模型实现
# ============================================================================

class ConvLSTMCell(nn.Module):
    """
    ConvLSTM 单元
    
    将卷积操作引入 LSTM，用于处理时空数据
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3,
                 bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        
        # 卷积门控：同时处理输入和隐藏状态
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # i, f, o, g 四个门
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
    
    def forward(self, input_tensor: torch.Tensor, 
                cur_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple]:
        """
        ConvLSTM 单步前向传播
        
        Args:
            input_tensor: 输入 [B, C, H, W]
            cur_state: (h_prev, c_prev) 上一时刻的隐藏状态和细胞状态
        
        Returns:
            h_cur: 新的隐藏状态
            (h_cur, c_cur): 新的状态元组
        """
        h_prev, c_prev = cur_state
        
        # 拼接输入和上一时刻隐藏状态
        combined = torch.cat([input_tensor, h_prev], dim=1)
        
        # 卷积操作
        combined_conv = self.conv(combined)
        
        # 分割为四个门
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # 门控计算
        i = torch.sigmoid(cc_i)  # 输入门
        f = torch.sigmoid(cc_f)  # 遗忘门
        o = torch.sigmoid(cc_o)  # 输出门
        g = torch.tanh(cc_g)     # 候选细胞状态
        
        # 更新细胞状态和隐藏状态
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        
        return h_cur, (h_cur, c_cur)
    
    def init_hidden(self, batch_size: int, height: int, width: int, 
                    device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化隐藏状态"""
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return (h, c)


class ConvLSTM(nn.Module):
    """
    多层 ConvLSTM
    
    堆叠多个 ConvLSTM 层以学习更复杂的时空特征
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 kernel_size: int = 3, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        
        # 构建多层 ConvLSTM
        cells = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dims[i-1]
            out_dim = hidden_dims[i]
            cells.append(ConvLSTMCell(in_dim, out_dim, kernel_size))
        
        self.cells = nn.ModuleList(cells)
    
    def forward(self, x: torch.Tensor, 
                hidden_states: Optional[List[Tuple]] = None) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        处理序列数据
        
        Args:
            x: 输入序列 [B, T, C, H, W]
            hidden_states: 可选的初始隐藏状态列表
        
        Returns:
            output: 输出序列 [B, T, C_last, H, W]
            hidden_states: 最终隐藏状态列表
        """
        B, T, C, H, W = x.shape
        
        # 初始化隐藏状态
        if hidden_states is None:
            hidden_states = [
                cell.init_hidden(B, H, W, x.device) 
                for cell in self.cells
            ]
        
        # 存储所有时间步的输出
        layer_output = []
        
        for t in range(T):
            input_t = x[:, t]
            cur_layer_input = input_t
            new_hidden_states = []
            
            # 逐层传递
            for layer_idx, cell in enumerate(self.cells):
                h, new_state = cell(cur_layer_input, hidden_states[layer_idx])
                new_hidden_states.append(new_state)
                cur_layer_input = h  # 上一层的输出作为下一层的输入
            
            layer_output.append(cur_layer_input)
            hidden_states = new_hidden_states
        
        # 堆叠为 [B, T, C, H, W]
        output = torch.stack(layer_output, dim=1)
        return output, hidden_states


class VideoPredictor(nn.Module):
    """
    基于 ConvLSTM 的视频预测器
    
    架构：
    编码器 → ConvLSTM → 解码器 → 预测帧
    """
    def __init__(self, input_channels: int = 3, hidden_channels: List[int] = [64, 128, 256],
                 kernel_size: int = 3, prediction_horizon: int = 10):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        
        # 编码器：提取每帧的空间特征
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels[0], 3, padding=1),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(),
            nn.Conv2d(hidden_channels[0], hidden_channels[0], 3, padding=1),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU()
        )
        
        # ConvLSTM：建模时间动态
        self.conv_lstm = ConvLSTM(
            input_dim=hidden_channels[0],
            hidden_dims=hidden_channels,
            kernel_size=kernel_size,
            num_layers=len(hidden_channels)
        )
        
        # 解码器：从特征重建帧
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], hidden_channels[-2], 3, padding=1),
            nn.BatchNorm2d(hidden_channels[-2]),
            nn.ReLU(),
            nn.Conv2d(hidden_channels[-2], input_channels, 3, padding=1),
            nn.Sigmoid()  # 输出归一化到 [0, 1]
        )
    
    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        预测未来视频帧
        
        Args:
            input_sequence: 输入帧序列 [B, T_in, C, H, W]
        
        Returns:
            predictions: 预测的未来帧 [B, T_out, C, H, W]
        """
        B, T_in, C, H, W = input_sequence.shape
        
        # 1. 编码输入序列的每一帧
        encoded = []
        for t in range(T_in):
            enc = self.encoder(input_sequence[:, t])
            encoded.append(enc)
        encoded = torch.stack(encoded, dim=1)  # [B, T_in, C_enc, H, W]
        
        # 2. ConvLSTM 处理序列
        _, hidden_states = self.conv_lstm(encoded)
        
        # 3. 自回归预测未来帧
        predictions = []
        cur_input = encoded[:, -1]  # 使用最后一帧的编码作为初始输入
        
        for t in range(self.prediction_horizon):
            # 用 ConvLSTM 更新状态
            _, hidden_states = self.conv_lstm(
                cur_input.unsqueeze(1), hidden_states
            )
            
            # 从最后一层隐藏状态解码
            h = hidden_states[-1][0]  # 取最后一层的 h
            pred_frame = self.decoder(h)
            predictions.append(pred_frame)
            
            # 将预测帧编码作为下一帧的输入（自回归）
            cur_input = self.encoder(pred_frame)
        
        predictions = torch.stack(predictions, dim=1)
        return predictions


# ============================================================================
# 2. 时空注意力机制
# ============================================================================

class SpatialAttention(nn.Module):
    """
    空间注意力模块
    
    关注帧内的重要空间区域
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
        """
        attn_map = self.attention(x)
        return x * attn_map


class TemporalAttention(nn.Module):
    """
    时间注意力模块
    
    关注时间序列中的重要帧
    """
    def __init__(self, in_channels: int, num_frames: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(num_frames, num_frames // 2),
            nn.ReLU(),
            nn.Linear(num_frames // 2, num_frames),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入序列 [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape
        
        # 全局池化获取时间序列表示
        x_pool = x.mean(dim=(3, 4))  # [B, T, C]
        x_pool = x_pool.mean(dim=2, keepdim=True)  # [B, T, 1]
        
        # 计算时间注意力权重
        attn_weights = self.attention(x_pool.squeeze(2))  # [B, T]
        
        # 应用注意力
        attn_weights = attn_weights.view(B, T, 1, 1, 1)
        x_weighted = x * attn_weights
        
        return x_weighted


class SpatioTemporalAttentionBlock(nn.Module):
    """
    时空注意力块
    
    结合空间和时间注意力
    """
    def __init__(self, channels: int, num_frames: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # QKV 投影
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
        
        # 归一化
        self.norm = nn.LayerNorm(channels)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入 [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape
        N = T * H * W  # 总 token 数
        
        # 展平时空维度
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, N, C)
        x_flat = self.norm(x_flat)
        
        # 计算 Q, K, V
        x_3d = x_flat.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        qkv = self.qkv(x_3d)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 加权求和
        x_attn = (attn @ v).transpose(2, 3).reshape(B, N, C)
        x_attn = self.proj(x_attn.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3))
        x_attn = x_attn.permute(0, 2, 3, 4, 1).reshape(B, N, C)
        
        # 残差连接
        x_out = x_flat + x_attn
        x_out = x_out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        
        return x_out


class VideoTransformer(nn.Module):
    """
    基于 Transformer 的视频预测模型
    
    使用时空注意力机制
    """
    def __init__(self, input_channels: int = 3, embed_dim: int = 192,
                 num_heads: int = 6, num_layers: int = 6,
                 patch_size: int = 4, prediction_horizon: int = 10):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Patch 嵌入
        self.patch_embed = nn.Conv2d(
            input_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, embed_dim))
        self.temp_embed = nn.Parameter(torch.zeros(1, 20, embed_dim))
        
        # Transformer 编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 预测头
        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, patch_size * patch_size * input_channels)
        )
    
    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_sequence: [B, T_in, C, H, W]
        """
        B, T_in, C, H, W = input_sequence.shape
        
        # 1. Patch 嵌入
        x = input_sequence.view(B * T_in, C, H, W)
        x = self.patch_embed(x)  # [B*T, D, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B*T, N, D]
        
        # 添加位置编码
        x = x + self.pos_embed[:, :x.shape[1]]
        
        # 恢复时间维度
        x = x.view(B, T_in, -1, self.embed_dim)
        x = x + self.temp_embed[:, :T_in]
        
        # 展平为序列
        x = x.view(B, T_in * x.shape[2], self.embed_dim)
        
        # 2. Transformer 编码
        x = self.transformer(x)
        
        # 3. 自回归预测未来帧
        predictions = []
        
        # 使用最后的表示作为初始状态
        cur_repr = x[:, -self.embed_dim:]
        
        for t in range(self.prediction_horizon):
            # 添加时间步编码
            time_emb = self.temp_embed[:, T_in + t:T_in + t + 1]
            cur_repr = cur_repr + time_emb.expand(-1, cur_repr.shape[1], -1)
            
            # Transformer 处理
            cur_repr = self.transformer(cur_repr)
            
            # 预测下一帧的 patches
            pred_patches = self.pred_head(cur_repr[:, -1])
            
            # 重组为帧
            pred_frame = pred_patches.view(
                B, self.embed_dim // (self.patch_size ** 2 * C),
                H // self.patch_size, W // self.patch_size
            )
            pred_frame = self.patch_embed.inverse(pred_frame) if hasattr(self.patch_embed, 'inverse') else pred_frame
            
            predictions.append(pred_frame)
        
        predictions = torch.stack(predictions, dim=1)
        return predictions


# ============================================================================
# 3. 多帧预测
# ============================================================================

class MultiFramePredictor(nn.Module):
    """
    多帧预测器
    
    支持多种预测策略：
    1. 自回归：逐帧预测，使用预测结果作为下一帧输入
    2. 直接：一次性预测所有未来帧
    3. 混合：结合自回归和直接预测
    """
    def __init__(self, input_channels: int = 3, hidden_dim: int = 256,
                 prediction_horizon: int = 10, strategy: str = 'autoregressive'):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.strategy = strategy
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 动态模型
        self.dynamics = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        if strategy == 'direct':
            # 直接预测所有帧
            self.decoder = nn.Linear(hidden_dim, prediction_horizon * input_channels * 64 * 64)
        else:
            # 自回归预测
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.ReLU(),
                nn.Linear(512, input_channels * 64 * 64)
            )
    
    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_sequence: [B, T_in, C, H, W]
        """
        B, T_in, C, H, W = input_sequence.shape
        
        # 编码输入序列
        encoded = []
        for t in range(T_in):
            enc = self.encoder(input_sequence[:, t])
            encoded.append(enc.squeeze(-1).squeeze(-1))
        encoded = torch.stack(encoded, dim=1)  # [B, T_in, D]
        
        if self.strategy == 'direct':
            # 直接预测
            _, (h_n, _) = self.dynamics(encoded)
            final_state = h_n[-1]
            all_frames = self.decoder(final_state)
            predictions = all_frames.view(B, self.prediction_horizon, C, 64, 64)
        else:
            # 自回归预测
            _, (h_n, c_n) = self.dynamics(encoded)
            
            predictions = []
            cur_state = h_n[-1]
            
            for t in range(self.prediction_horizon):
                # 解码当前帧
                frame = self.decoder(cur_state)
                frame = frame.view(B, C, 64, 64)
                predictions.append(frame)
                
                # 编码预测帧并更新状态
                if t < self.prediction_horizon - 1:
                    enc = self.encoder(frame)
                    enc = enc.squeeze(-1).squeeze(-1)
                    _, (h_n, c_n) = self.dynamics(enc.unsqueeze(1), (h_n, c_n))
                    cur_state = h_n[-1]
            
            predictions = torch.stack(predictions, dim=1)
        
        return predictions


class HierarchicalPredictor(nn.Module):
    """
    层次化预测器
    
    在不同时间尺度上预测：
    - 低分辨率：长期趋势
    - 高分辨率：短期细节
    """
    def __init__(self, input_channels: int = 3, prediction_horizon: int = 10):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        
        # 低分辨率路径（长期预测）
        self.low_res_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.low_res_lstm = nn.LSTM(64, 128, batch_first=True)
        self.low_res_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 高分辨率路径（短期细节）
        self.high_res_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.high_res_lstm = nn.LSTM(64 + 128, 128, batch_first=True)  # +128 来自低分辨率
        self.high_res_decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_sequence: [B, T_in, C, H, W]
        """
        B, T_in, C, H, W = input_sequence.shape
        
        # 低分辨率处理
        low_res = []
        for t in range(T_in):
            lr = F.interpolate(input_sequence[:, t], scale_factor=0.5)
            lr_enc = self.low_res_encoder(lr)
            low_res.append(lr_enc.flatten(2).mean(2))  # 全局池化
        
        low_res = torch.stack(low_res, dim=1)
        low_res_out, (low_res_h, _) = self.low_res_lstm(low_res)
        
        # 高分辨率处理
        high_res = []
        high_res_h = low_res_h  # 从低分辨率初始化
        high_res_c = torch.zeros_like(low_res_h)
        
        for t in range(T_in):
            hr_enc = self.high_res_encoder(input_sequence[:, t])
            hr_enc_flat = hr_enc.flatten(2).mean(2)
            
            # 融合低分辨率信息
            lr_info = low_res_out[:, t].unsqueeze(-1).expand(-1, -1, hr_enc.shape[2] * hr_enc.shape[3])
            lr_info = lr_info.view(B, -1, hr_enc.shape[2], hr_enc.shape[3])
            combined = torch.cat([hr_enc, lr_info], dim=1)
            combined_flat = combined.flatten(2).mean(2)
            
            _, (high_res_h, high_res_c) = self.high_res_lstm(
                combined_flat.unsqueeze(1), (high_res_h, high_res_c)
            )
            high_res.append(high_res_h[-1])
        
        # 预测未来
        predictions = []
        for t in range(self.prediction_horizon):
            # 低分辨率预测
            _, (low_res_h, _) = self.low_res_lstm(
                low_res_out[:, -1:].mean(1, keepdim=True), (low_res_h, _)
            )
            lr_pred = self.low_res_decoder(low_res_h[-1].view(B, 128, 8, 8))
            lr_pred = F.interpolate(lr_pred, scale_factor=2)
            
            # 高分辨率预测
            _, (high_res_h, high_res_c) = self.high_res_lstm(
                high_res[-1].unsqueeze(1), (high_res_h, high_res_c)
            )
            hr_pred = self.high_res_decoder(high_res_h[-1].view(B, 128, 8, 8))
            
            # 融合
            pred = 0.5 * lr_pred + 0.5 * hr_pred
            predictions.append(pred)
            
            high_res.append(high_res_h[-1])
        
        return torch.stack(predictions, dim=1)


# ============================================================================
# 4. 视频生成示例
# ============================================================================

class VideoGenerator:
    """
    视频生成器
    
    提供视频预测和可视化功能
    """
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, input_frames: np.ndarray, num_future: int = 10) -> np.ndarray:
        """
        预测未来帧
        
        Args:
            input_frames: 输入帧序列 [T, H, W, C]，值域 [0, 1]
            num_future: 预测的未来帧数
        
        Returns:
            predicted_frames: 预测帧 [T_in + T_out, H, W, C]
        """
        # 转换为张量
        input_tensor = torch.FloatTensor(input_frames).permute(0, 3, 1, 2).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # 预测
        predictions = self.model(input_tensor)
        
        # 合并输入和预测
        all_frames = torch.cat([input_tensor[0].cpu(), predictions[0].cpu()], dim=0)
        all_frames = all_frames.permute(0, 2, 3, 1).numpy()
        
        return all_frames
    
    def visualize(self, frames: np.ndarray, save_path: Optional[str] = None):
        """
        可视化视频帧序列
        
        Args:
            frames: 帧序列 [T, H, W, C]
            save_path: 可选的保存路径
        """
        T = len(frames)
        fig, axes = plt.subplots(2, (T + 1) // 2, figsize=(20, 5))
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            if i < T:
                ax.imshow(frames[i])
                if i == frames.shape[0] - 10:
                    ax.set_title('← 输入结束 | 预测开始 →', fontsize=10)
                elif i < frames.shape[0] - 10:
                    ax.set_title(f'输入 {i+1}', fontsize=8)
                else:
                    ax.set_title(f'预测 {i - (frames.shape[0] - 10) + 1}', fontsize=8)
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"可视化已保存到 {save_path}")
        else:
            plt.show()
    
    def create_comparison_video(self, input_frames: np.ndarray, 
                                 target_frames: np.ndarray,
                                 predicted_frames: np.ndarray,
                                 save_path: str = 'comparison.mp4'):
        """
        创建对比视频（输入 | 真实 | 预测）
        """
        try:
            import cv2
        except ImportError:
            print("需要安装 opencv-python 来创建视频")
            return
        
        T = min(len(target_frames), len(predicted_frames))
        H, W = input_frames.shape[1:3]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 10, (W * 3, H))
        
        # 输入帧（重复最后一帧）
        last_input = input_frames[-1]
        
        for t in range(T):
            # 合并三帧
            combined = np.hstack([
                (last_input * 255).astype(np.uint8),
                (target_frames[t] * 255).astype(np.uint8),
                (predicted_frames[t] * 255).astype(np.uint8)
            ])
            combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            out.write(combined)
        
        out.release()
        print(f"对比视频已保存到 {save_path}")


class TrajectoryVisualizer:
    """
    轨迹可视化工具
    
    用于可视化视频预测中的物体运动轨迹
    """
    def __init__(self):
        pass
    
    def extract_trajectory(self, frames: np.ndarray, object_mask: np.ndarray = None) -> np.ndarray:
        """
        从视频帧中提取物体轨迹
        
        Args:
            frames: 视频帧 [T, H, W, C]
            object_mask: 可选的物体掩码
        
        Returns:
            trajectory: 轨迹点 [T, 2] (x, y 坐标)
        """
        T, H, W, C = frames.shape
        trajectory = []
        
        for t in range(T):
            frame = frames[t]
            
            if object_mask is not None:
                # 使用提供的掩码
                mask = object_mask[t]
            else:
                # 简单的光流法估计运动
                if t > 0:
                    diff = np.abs(frame - frames[t-1]).mean(axis=2)
                    coords = np.argwhere(diff > diff.mean() + diff.std())
                    if len(coords) > 0:
                        center = coords.mean(axis=0)
                        trajectory.append([center[1], center[0]])  # x, y
                        continue
            
            trajectory.append([W // 2, H // 2])  # 默认中心
        
        return np.array(trajectory)
    
    def plot_trajectory(self, trajectory: np.ndarray, frame_shape: Tuple[int, int],
                        title: str = '物体运动轨迹'):
        """
        绘制运动轨迹
        
        Args:
            trajectory: 轨迹点 [T, 2]
            frame_shape: 帧形状 (H, W)
        """
        H, W = frame_shape
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制轨迹
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='轨迹')
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, label='起点', zorder=5)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, label='终点', zorder=5)
        
        # 设置范围
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # y 轴向下
        ax.set_aspect('equal')
        
        ax.set_xlabel('X 像素')
        ax.set_ylabel('Y 像素')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# 评估指标
# ============================================================================

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """计算 PSNR"""
    mse = ((pred - target) ** 2).mean()
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr.item()


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, 
                   window_size: int = 11, max_val: float = 1.0) -> float:
    """计算 SSIM"""
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    def gaussian_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.outer(g).unsqueeze(0).unsqueeze(0)
    
    window = gaussian_window(window_size, 1.5).to(pred.device)
    window = window.expand(pred.shape[1], -1, -1, -1)
    
    mu_x = F.conv2d(pred, window, padding=window_size//2, groups=pred.shape[1])
    mu_y = F.conv2d(target, window, padding=window_size//2, groups=target.shape[1])
    
    sigma_x2 = F.conv2d(pred**2, window, padding=window_size//2, groups=pred.shape[1]) - mu_x**2
    sigma_y2 = F.conv2d(target**2, window, padding=window_size//2, groups=target.shape[1]) - mu_y**2
    sigma_xy = F.conv2d(pred*target, window, padding=window_size//2, groups=pred.shape[1]) - mu_x*mu_y
    
    ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2))
    
    return ssim_map.mean().item()


class VideoPredictionEvaluator:
    """视频预测评估器"""
    def __init__(self):
        pass
    
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
        """
        评估预测质量
        
        Args:
            predictions: [B, T, C, H, W]
            targets: [B, T, C, H, W]
        """
        B, T, C, H, W = predictions.shape
        
        psnrs = []
        ssims = []
        
        for t in range(T):
            for b in range(B):
                pred_t = predictions[b, t]
                target_t = targets[b, t]
                
                psnr = calculate_psnr(pred_t, target_t)
                if psnr != float('inf'):
                    psnrs.append(psnr)
                
                ssim = calculate_ssim(pred_t, target_t)
                ssims.append(ssim)
        
        return {
            'psnr_mean': sum(psnrs) / len(psnrs) if psnrs else 0,
            'ssim_mean': sum(ssims) / len(ssims) if ssims else 0,
            'psnr_per_frame': [sum(psnrs[i::T])/len(psnrs[i::T]) for i in range(T)] if psnrs else [],
            'ssim_per_frame': [sum(ssims[i::T])/len(ssims[i::T]) for i in range(T)] if ssims else []
        }


# ============================================================================
# 主程序示例
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("视频预测 - 代码示例")
    print("=" * 60)
    
    # 配置
    config = {
        'input_channels': 3,
        'hidden_channels': [64, 128, 256],
        'prediction_horizon': 10,
        'input_sequence_length': 10,
        'img_size': 64,
        'batch_size': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"\n设备：{config['device']}")
    print(f"输入序列长度：{config['input_sequence_length']}")
    print(f"预测未来帧数：{config['prediction_horizon']}")
    
    # 创建模型
    model = VideoPredictor(
        input_channels=config['input_channels'],
        hidden_channels=config['hidden_channels'],
        prediction_horizon=config['prediction_horizon']
    ).to(config['device'])
    
    print(f"\n模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    
    # 创建模拟输入
    B = config['batch_size']
    T_in = config['input_sequence_length']
    C, H, W = config['input_channels'], config['img_size'], config['img_size']
    
    input_sequence = torch.rand(B, T_in, C, H, W).to(config['device'])
    
    print(f"\n输入形状：{input_sequence.shape}")
    
    # 前向传播
    print("进行前向传播...")
    with torch.no_grad():
        predictions = model(input_sequence)
    
    print(f"预测输出形状：{predictions.shape}")
    
    # 评估
    # 创建模拟目标
    target_sequence = torch.rand(B, config['prediction_horizon'], C, H, W).to(config['device'])
    
    evaluator = VideoPredictionEvaluator()
    metrics = evaluator.evaluate(predictions, target_sequence)
    
    print(f"\n评估指标（随机预测）:")
    print(f"  PSNR: {metrics['psnr_mean']:.2f} dB")
    print(f"  SSIM: {metrics['ssim_mean']:.4f}")
    
    # 可视化示例
    print("\n创建可视化...")
    generator = VideoGenerator(model, config['device'])
    
    # 单样本可视化
    sample_input = input_sequence[0].cpu().permute(0, 2, 3, 1).numpy()
    sample_pred = predictions[0].cpu().permute(0, 2, 3, 1).numpy()
    all_frames = np.concatenate([sample_input, sample_pred], axis=0)
    
    generator.visualize(all_frames, save_path='video_prediction_example.png')
    
    print("\n示例完成！")
    print("输出文件：video_prediction_example.png")
