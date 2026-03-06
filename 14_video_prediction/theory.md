# 第 14 章 视频预测

## 14.1 视频预测任务定义

### 什么是视频预测？

**视频预测（Video Prediction）** 是指给定过去的视频帧序列，预测未来视频帧的任务。这是计算机视觉和机器学习中的重要问题，具有广泛的应用价值。

```
输入：[Frame_{t-T}, ..., Frame_{t-1}, Frame_t]
输出：[Frame_{t+1}, Frame_{t+2}, ..., Frame_{t+K}]
```

### 数学形式化

给定历史帧序列 $X_{1:t} = \{x_1, x_2, ..., x_t\}$，视频预测模型学习条件概率分布：

$$P(x_{t+1:t+K} | x_{1:t})$$

其中 $K$ 是预测的未来帧数。

### 视频预测的挑战

1. **高维数据**：视频帧包含大量像素，直接建模计算量大
2. **时间依赖性**：需要捕捉长程时间依赖关系
3. **不确定性**：未来可能有多种合理结果（多模态）
4. **运动复杂性**：物体运动、形变、遮挡等
5. **外观变化**：光照变化、纹理细节等

### 应用场景

| 应用领域 | 具体应用 |
|---------|---------|
| **自动驾驶** | 预测其他车辆和行人的运动轨迹 |
| **机器人** | 预测动作执行后的视觉结果 |
| **视频压缩** | 预测下一帧以减少编码比特 |
| **异常检测** | 预测与实际的差异检测异常 |
| **视频生成** | 生成连续的视频内容 |
| **游戏 AI** | 预测游戏画面的变化 |

```python
# 视频预测的基本接口
class VideoPredictor:
    def predict(self, frames: torch.Tensor, num_future: int) -> torch.Tensor:
        """
        预测未来视频帧
        
        Args:
            frames: 输入帧序列 [B, T, C, H, W]
            num_future: 预测的未来帧数
        
        Returns:
            predicted_frames: 预测的未来帧 [B, num_future, C, H, W]
        """
        pass
    
    def predict_step(self, frames: torch.Tensor) -> torch.Tensor:
        """单步预测"""
        pass
```

## 14.2 时空建模方法

### 时空数据的特性

视频是典型的时空数据：
- **空间维度**：每帧内的像素排列（高度 × 宽度）
- **时间维度**：帧与帧之间的变化
- **通道维度**：颜色通道（RGB）

### 基本建模范式

**范式 1：分离式建模**
```
空间特征提取（CNN） → 时间建模（RNN/Transformer） → 解码
```

**范式 2：联合式建模**
```
3D CNN / 时空 Transformer → 直接输出预测
```

**范式 3：潜在空间建模**
```
编码 → 潜在动态模型 → 解码
```

### 3D 卷积

3D 卷积同时在空间和时间维度上进行卷积：

```python
import torch
import torch.nn as nn

class Conv3DBlock(nn.Module):
    """3D 卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: 输入张量 [B, C, T, H, W]
        """
        return self.relu(self.bn(self.conv(x)))


# 3D 卷积 vs 2D 卷积 + RNN
class Comparison(nn.Module):
    """对比 3D 卷积和 2D+RNN 方法"""
    def __init__(self, input_shape):
        super().__init__()
        C, T, H, W = input_shape
        
        # 方法 1: 3D 卷积
        self.conv3d = nn.Sequential(
            nn.Conv3d(C, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 方法 2: 2D CNN + RNN
        self.conv2d = nn.Conv2d(C, 64, kernel_size=3, padding=1)
        self.rnn = nn.ConvLSTM2d(64, 64, kernel_size=3, padding=1)
    
    def forward_3d(self, x):
        """3D 卷积方法"""
        # x: [B, C, T, H, W]
        return self.conv3d(x)
    
    def forward_2d_rnn(self, x):
        """2D CNN + RNN 方法"""
        B, C, T, H, W = x.shape
        
        # 对每帧应用 2D CNN
        x = x.view(B * T, C, H, W)
        x = self.conv2d(x)
        x = x.view(B, T, 64, H, W)
        
        # 沿时间维度应用 RNN
        outputs = []
        h, c = None, None
        for t in range(T):
            h, c = self.rnn(x[:, t], (h, c))
            outputs.append(h)
        
        return torch.stack(outputs, dim=1)
```

### 时空注意力机制

注意力机制可以捕捉长程依赖：

```python
class SpatioTemporalAttention(nn.Module):
    """时空注意力模块"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        """
        Args:
            x: 输入 [B, T*H*W, C]（展平的时空位置）
        """
        B, N, C = x.shape
        
        # 计算 Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class SeparableSpatioTemporalAttention(nn.Module):
    """
    分离式时空注意力
    
    分别计算空间注意力和时间注意力，效率更高
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        """
        Args:
            x: 输入 [B, T, H*W, C]
        """
        B, T, N, C = x.shape
        
        # 空间注意力（在每个时间步内）
        x = x.view(B * T, N, C)
        x = self.norm1(x)
        x_spatial, _ = self.spatial_attn(x, x, x)
        x = x + x_spatial
        
        # 时间注意力（在每个空间位置）
        x = x.view(B, T, N, C).permute(2, 0, 1, 3)  # [N, B, T, C]
        x = x.reshape(N * B, T, C)
        x = self.norm2(x)
        x_temporal, _ = self.temporal_attn(x, x, x)
        x = x + x_temporal
        
        # 恢复形状
        x = x.view(N, B, T, C).permute(1, 2, 0, 3)
        x = x.view(B, T, N, C)
        
        return x
```

## 14.3 ConvLSTM / Trajectory LSTM

### ConvLSTM 原理

**ConvLSTM** 将卷积操作引入 LSTM，使其能够处理时空数据：

$$\begin{aligned}
i_t &= \sigma(W_{xi} * X_t + W_{hi} * H_{t-1} + b_i) \\
f_t &= \sigma(W_{xf} * X_t + W_{hf} * H_{t-1} + b_f) \\
o_t &= \sigma(W_{xo} * X_t + W_{ho} * H_{t-1} + b_o) \\
\tilde{C}_t &= \tanh(W_{x\tilde{C}} * X_t + W_{h\tilde{C}} * H_{t-1} + b_{\tilde{C}}) \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\
H_t &= o_t \odot \tanh(C_t)
\end{aligned}$$

其中 $*$ 表示卷积操作，$\odot$ 表示逐元素乘法。

```python
class ConvLSTMCell(nn.Module):
    """ConvLSTM 单元"""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        
        # 卷积门控
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # i, f, o, g
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
    
    def forward(self, input_tensor, cur_state):
        """
        Args:
            input_tensor: [B, C, H, W]
            cur_state: (h_prev, c_prev)
        """
        h_prev, c_prev = cur_state
        
        # 拼接输入和上一时刻隐藏状态
        combined = torch.cat([input_tensor, h_prev], dim=1)
        
        # 卷积
        combined_conv = self.conv(combined)
        
        # 分割为四个门
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # 门控计算
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        # 更新细胞状态和隐藏状态
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        
        return h_cur, (h_cur, c_cur)
    
    def init_hidden(self, batch_size, height, width, device):
        """初始化隐藏状态"""
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return (h, c)


class ConvLSTM(nn.Module):
    """多层 ConvLSTM"""
    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        # 构建多层 ConvLSTM
        cells = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dims[i-1]
            out_dim = hidden_dims[i]
            cells.append(ConvLSTMCell(in_dim, out_dim, kernel_size))
        
        self.cells = nn.ModuleList(cells)
    
    def forward(self, x, hidden_states=None):
        """
        Args:
            x: 输入序列 [B, T, C, H, W]
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
            
            # 逐层传递
            cur_layer_input = input_t
            new_hidden_states = []
            
            for layer_idx, cell in enumerate(self.cells):
                h, new_state = cell(cur_layer_input, hidden_states[layer_idx])
                new_hidden_states.append(new_state)
                cur_layer_input = h
            
            layer_output.append(cur_layer_input)
            hidden_states = new_hidden_states
        
        # [B, T, C, H, W]
        output = torch.stack(layer_output, dim=1)
        return output, hidden_states


# 使用 ConvLSTM 进行视频预测
class ConvLSTMPredictor(nn.Module):
    """基于 ConvLSTM 的视频预测器"""
    def __init__(self, input_channels=3, hidden_channels=[64, 128, 256],
                 kernel_size=3, prediction_horizon=10):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels[0], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels[0], hidden_channels[0], 3, padding=1),
            nn.ReLU()
        )
        
        # ConvLSTM
        self.conv_lstm = ConvLSTM(
            input_dim=hidden_channels[0],
            hidden_dims=hidden_channels,
            kernel_size=(kernel_size, kernel_size),
            num_layers=len(hidden_channels)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], hidden_channels[-2], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels[-2], input_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, input_sequence):
        """
        Args:
            input_sequence: [B, T_in, C, H, W]
        
        Returns:
            predictions: [B, T_out, C, H, W]
        """
        B, T_in, C, H, W = input_sequence.shape
        
        # 编码输入序列
        encoded = []
        for t in range(T_in):
            enc = self.encoder(input_sequence[:, t])
            encoded.append(enc)
        encoded = torch.stack(encoded, dim=1)  # [B, T_in, C_enc, H, W]
        
        # ConvLSTM 处理
        _, hidden_states = self.conv_lstm(encoded)
        
        # 自回归预测未来帧
        predictions = []
        cur_input = encoded[:, -1]  # 最后一帧的编码
        
        for t in range(self.prediction_horizon):
            # 更新 ConvLSTM 状态
            _, hidden_states = self.conv_lstm(
                cur_input.unsqueeze(1), hidden_states
            )
            
            # 从最后一层隐藏状态解码
            h = hidden_states[-1][0]  # 取最后一层的 h
            pred_frame = self.decoder(h)
            predictions.append(pred_frame)
            
            # 将预测作为下一帧的输入（自回归）
            cur_input = self.encoder(pred_frame)
        
        predictions = torch.stack(predictions, dim=1)
        return predictions
```

### Trajectory LSTM

**Trajectory LSTM** 是一种专门为轨迹预测设计的变体，特别适用于视频中的物体运动预测：

```python
class TrajectoryLSTM(nn.Module):
    """
    Trajectory LSTM
    
    专门用于预测物体轨迹的 LSTM 变体
    输入：物体位置序列
    输出：预测的未来位置
    """
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.1
        )
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, trajectory, future_steps=10):
        """
        Args:
            trajectory: 历史轨迹 [B, T, 2] (x, y 坐标)
            future_steps: 预测的未来步数
        
        Returns:
            predicted_trajectory: [B, future_steps, 2]
        """
        # LSTM 编码历史轨迹
        lstm_out, (h_n, c_n) = self.lstm(trajectory)
        
        # 自回归预测未来
        predictions = []
        cur_input = trajectory[:, -1:]  # 最后已知位置
        
        for _ in range(future_steps):
            out, (h_n, c_n) = self.lstm(cur_input, (h_n, c_n))
            pred = self.fc(out[:, -1])  # [B, 2]
            predictions.append(pred)
            cur_input = pred.unsqueeze(1)
        
        return torch.stack(predictions, dim=1)


class SpatioTemporalTrajectoryLSTM(nn.Module):
    """
    时空 Trajectory LSTM
    
    同时处理视觉特征和轨迹信息
    """
    def __init__(self, visual_dim=512, trajectory_dim=2, hidden_dim=256):
        super().__init__()
        # 视觉特征编码器
        self.visual_encoder = nn.Linear(visual_dim, hidden_dim)
        
        # 轨迹 LSTM
        self.trajectory_lstm = nn.LSTM(trajectory_dim, hidden_dim, batch_first=True)
        
        # 融合模块
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 预测头
        self.predictor = nn.Linear(hidden_dim, trajectory_dim)
    
    def forward(self, visual_features, trajectories, future_steps=10):
        """
        Args:
            visual_features: 视觉特征 [B, T, visual_dim]
            trajectories: 轨迹 [B, T, 2]
        """
        # 编码视觉特征
        vis_enc = self.visual_encoder(visual_features)
        
        # 编码轨迹
        traj_out, _ = self.trajectory_lstm(trajectories)
        
        # 融合
        combined = torch.cat([vis_enc, traj_out], dim=-1)
        fused = self.fusion(combined)
        
        # 预测未来轨迹
        predictions = []
        cur_traj = trajectories[:, -1:]
        cur_fused = fused[:, -1:]
        
        for _ in range(future_steps):
            pred = self.predictor(cur_fused[:, -1])
            predictions.append(pred)
            cur_traj = pred.unsqueeze(1)
        
        return torch.stack(predictions, dim=1)
```

## 14.4 Video GPT / VideoBERT

### Video GPT

**Video GPT** 将 GPT 的自回归生成范式应用于视频预测：

```
将视频帧序列化为 token 序列 → 使用 Transformer 建模 → 自回归生成未来 token
```

```python
class VideoGPT(nn.Module):
    """
    Video GPT 简化实现
    
    使用 VQ-VAE 离散化视频帧，然后用 GPT 预测 token 序列
    """
    def __init__(self, vocab_size, embed_dim=768, num_heads=12, 
                 num_layers=12, max_seq_len=1024):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # Token 嵌入
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
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
        self.head = nn.Linear(embed_dim, vocab_size)
        
        # 初始化位置编码
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, tokens, mask=None):
        """
        Args:
            tokens: 输入 token 序列 [B, T]
            mask: 注意力掩码
        
        Returns:
            logits: 预测的下一个 token 分布 [B, T, vocab_size]
        """
        B, T = tokens.shape
        
        # Token 嵌入 + 位置编码
        x = self.token_embed(tokens) + self.pos_embed[:, :T]
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # 预测头
        logits = self.head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成
        
        Args:
            tokens: 初始 token 序列 [B, T]
            max_new_tokens: 最多生成的新 token 数
            temperature: 采样温度
            top_k: top-k 采样
        """
        for _ in range(max_new_tokens):
            # 截断到最大序列长度
            tokens_cond = tokens[:, -self.max_seq_len:]
            
            # 前向传播
            logits = self.forward(tokens_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 追加到序列
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokens


class VQVAEForVideo(nn.Module):
    """
    用于视频的 VQ-VAE
    
    将视频帧离散化为 token，供 Video GPT 使用
    """
    def __init__(self, input_channels=3, hidden_dim=256, 
                 num_embeddings=512, embedding_dim=64):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, embedding_dim, 1)
        )
        
        # 向量量化
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, input_channels, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """编码并量化"""
        z_e = self.encoder(x)
        
        # 向量量化
        z_e_flat = z_e.flatten(1).unsqueeze(1)  # [B, 1, H*W*C]
        codebook = self.codebook.weight.unsqueeze(0)  # [1, N, D]
        
        # 计算距离
        distances = (
            (z_e_flat ** 2).sum(2, keepdim=True)
            - 2 * z_e_flat @ codebook.transpose(1, 2)
            + (codebook ** 2).sum(2).unsqueeze(0)
        )
        
        # 找到最近的码本向量
        encoding_indices = torch.argmin(distances, dim=2)
        z_q = self.codebook(encoding_indices).view_as(z_e)
        
        return z_q, encoding_indices
    
    def decode(self, z_q):
        """从量化表示解码"""
        return self.decoder(z_q)
    
    def forward(self, x):
        """前向传播"""
        z_q, encoding_indices = self.encode(x)
        recon = self.decode(z_q)
        return recon, encoding_indices


# Video GPT 训练示例
class VideoGPTTrainer:
    """Video GPT 训练器"""
    def __init__(self, vq_vae, video_gpt, config):
        self.vq_vae = vq_vae
        self.video_gpt = video_gpt
        self.config = config
        
        self.vq_optimizer = torch.optim.Adam(vq_vae.parameters(), lr=1e-4)
        self.gpt_optimizer = torch.optim.Adam(video_gpt.parameters(), lr=3e-4)
    
    def tokenize_video(self, video_frames):
        """
        将视频帧序列转换为 token 序列
        
        Args:
            video_frames: [B, T, C, H, W]
        
        Returns:
            tokens: [B, T * H' * W']
        """
        B, T, C, H, W = video_frames.shape
        all_tokens = []
        
        for t in range(T):
            _, tokens = self.vq_vae.encode(video_frames[:, t])
            # tokens: [B, H' * W']
            all_tokens.append(tokens)
        
        return torch.cat(all_tokens, dim=1)  # [B, T * H' * W']
    
    def train_gpt(self, video_batch):
        """
        训练 Video GPT
        
        Args:
            video_batch: [B, T, C, H, W]
        """
        # 转换为 token
        tokens = self.tokenize_video(video_batch)
        B, L = tokens.shape
        
        # 创建输入和目标（自回归）
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        
        # 前向传播
        self.gpt_optimizer.zero_grad()
        logits = self.video_gpt(input_tokens)
        
        # 计算损失
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
        
        # 反向传播
        loss.backward()
        self.gpt_optimizer.step()
        
        return loss.item()
```

### VideoBERT

**VideoBERT** 是另一个将 BERT 范式应用于视频的工作：

```python
class VideoBERT(nn.Module):
    """
    VideoBERT 简化实现
    
    使用掩码语言建模预训练视频表示
    """
    def __init__(self, vocab_size, embed_dim=768, num_heads=12, num_layers=12):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 用于掩码预测的头
        self.mlm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size)
        )
    
    def forward(self, tokens, mask=None, mask_indices=None):
        """
        Args:
            tokens: 输入 token [B, T]
            mask: 注意力掩码
            mask_indices: 需要预测的位置 [B, num_masked]
        """
        B, T = tokens.shape
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # 位置编码
        x = self.token_embed(tokens) + self.pos_embed[:, :T+1]
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # 如果需要 MLM 预测
        if mask_indices is not None:
            # 提取被掩码位置的表示
            B, num_masked = mask_indices.shape
            batch_indices = torch.arange(B).unsqueeze(1).expand(-1, num_masked)
            masked_repr = x[batch_indices, mask_indices + 1]  # +1 因为 CLS
            logits = self.mlm_head(masked_repr)
            return logits
        
        return x
```

## 14.5 JeSS 方法

### JeSS 概述

**JeSS（Joint Embedding Self-Supervised）** 是一种自监督视频表示学习方法，通过对比学习捕捉时空一致性。

### 核心思想

1. **时空增强**：对视频进行时空裁剪和变换
2. **联合嵌入**：将不同视图映射到共享表示空间
3. **对比损失**：最大化正样本对的相似度，最小化负样本对

```python
class JeSS(nn.Module):
    """
    JeSS 自监督视频表示学习
    """
    def __init__(self, encoder_dim=2048, proj_dim=256, temp=0.07):
        super().__init__()
        self.temp = temp
        
        # 视频编码器（可以使用 3D CNN 或 Video Transformer）
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # ... 更多层
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 4, encoder_dim)
        )
        
        # 投影头（用于对比学习）
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, proj_dim)
        )
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
    
    def encode(self, video):
        """编码视频"""
        return self.encoder(video)
    
    def project(self, repr):
        """投影到对比学习空间"""
        return self.projector(repr)
    
    def forward(self, video1, video2):
        """
        前向传播（两个增强视图）
        
        Args:
            video1, video2: 同一视频的两个增强视图 [B, C, T, H, W]
        
        Returns:
            loss: 对比损失
        """
        # 编码
        repr1 = self.encode(video1)
        repr2 = self.encode(video2)
        
        # 投影
        proj1 = self.project(repr1)
        proj2 = self.project(repr2)
        
        # 预测
        pred1 = self.predictor(proj1)
        pred2 = self.predictor(proj2)
        
        # 对比损失（InfoNCE）
        loss = self.info_nce_loss(pred1, proj2) + self.info_nce_loss(pred2, proj1)
        
        return loss
    
    def info_nce_loss(self, pred, target):
        """
        InfoNCE 损失
        """
        pred = F.normalize(pred, dim=1)
        target = F.normalize(target, dim=1)
        
        # 计算相似度
        pos_sim = (pred * target).sum(dim=1) / self.temp
        
        # 负样本相似度（与其他样本的 target）
        neg_sim = torch.matmul(pred, target.T) / self.temp
        
        # InfoNCE 损失
        loss = -pos_sim + torch.logsumexp(neg_sim, dim=1)
        
        return loss.mean()


class JeSSForPrediction(nn.Module):
    """
    使用 JeSS 预训练的视频预测模型
    """
    def __init__(self, jess_encoder, prediction_horizon=10):
        super().__init__()
        self.encoder = jess_encoder.encoder
        self.prediction_horizon = prediction_horizon
        
        # 动态预测头
        self.dynamics_predictor = nn.LSTM(
            input_size=2048,
            hidden_size=512,
            num_layers=2,
            batch_first=True
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * 64 * 64),  # 假设输出 64x64 RGB
        )
    
    def forward(self, video_sequence):
        """
        Args:
            video_sequence: [B, T, C, H, W]
        """
        B, T, C, H, W = video_sequence.shape
        
        # 编码每一帧
        representations = []
        for t in range(T):
            # 需要调整形状以适应 3D CNN
            frame = video_sequence[:, t:t+1].transpose(1, 2)  # [B, C, 1, H, W]
            repr = self.encoder(frame).squeeze(2)  # [B, D]
            representations.append(repr)
        
        repr_seq = torch.stack(representations, dim=1)  # [B, T, D]
        
        # 预测未来表示
        pred_repr, _ = self.dynamics_predictor(repr_seq)
        
        # 解码为视频帧
        predictions = []
        for t in range(self.prediction_horizon):
            pred = self.decoder(pred_repr[:, -1])
            pred = pred.view(B, 3, 64, 64)
            predictions.append(pred)
        
        return torch.stack(predictions, dim=1)
```

## 14.6 评估指标

### PSNR（峰值信噪比）

**PSNR** 衡量重建质量，值越高越好：

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$$

```python
def calculate_psnr(pred, target, max_val=1.0):
    """
    计算 PSNR
    
    Args:
        pred: 预测帧 [B, C, H, W] 或 [C, H, W]
        target: 目标帧
        max_val: 像素最大值
    
    Returns:
        psnr: PSNR 值（dB）
    """
    mse = ((pred - target) ** 2).mean()
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr.item()


def calculate_psnr_batch(preds, targets):
    """批量计算 PSNR"""
    psnrs = []
    for pred, target in zip(preds, targets):
        psnr = calculate_psnr(pred, target)
        if psnr != float('inf'):
            psnrs.append(psnr)
    return sum(psnrs) / len(psnrs) if psnrs else float('inf')
```

### SSIM（结构相似性）

**SSIM** 衡量结构相似性，考虑亮度、对比度和结构：

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

```python
def calculate_ssim(pred, target, window_size=11, max_val=1.0):
    """
    计算 SSIM
    
    使用高斯窗口计算局部统计量
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # 创建高斯窗口
    def gaussian_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.outer(g).unsqueeze(0).unsqueeze(0)
    
    window = gaussian_window(window_size, 1.5).to(pred.device)
    window = window.expand(pred.shape[1], -1, -1, -1)
    
    # 计算局部均值
    mu_x = F.conv2d(pred, window, padding=window_size//2, groups=pred.shape[1])
    mu_y = F.conv2d(target, window, padding=window_size//2, groups=target.shape[1])
    
    # 计算局部方差和协方差
    sigma_x2 = F.conv2d(pred**2, window, padding=window_size//2, groups=pred.shape[1]) - mu_x**2
    sigma_y2 = F.conv2d(target**2, window, padding=window_size//2, groups=target.shape[1]) - mu_y**2
    sigma_xy = F.conv2d(pred*target, window, padding=window_size//2, groups=pred.shape[1]) - mu_x*mu_y
    
    # SSIM
    ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2))
    
    return ssim_map.mean().item()
```

### LPIPS（学习感知图像块相似度）

**LPIPS** 使用预训练网络的特征计算感知相似度：

```python
import lpips

class LPIPSEvaluator:
    """LPIPS 评估器"""
    def __init__(self, net_type='alex'):
        """
        Args:
            net_type: 'alex', 'vgg', 或 'squeeze'
        """
        self.loss_fn = lpips.LPIPS(net=net_type)
        self.loss_fn.eval()
    
    def calculate(self, pred, target):
        """
        计算 LPIPS 距离（越低越好）
        
        Args:
            pred: 预测图像 [B, C, H, W]，值域 [-1, 1]
            target: 目标图像
        """
        # LPIPS 期望输入在 [-1, 1] 范围
        with torch.no_grad():
            lpips_score = self.loss_fn(pred, target)
        return lpips_score.mean().item()


# 综合评估器
class VideoPredictionEvaluator:
    """视频预测综合评估器"""
    def __init__(self):
        self.lpips_evaluator = LPIPSEvaluator()
    
    def evaluate(self, predictions, targets):
        """
        全面评估预测质量
        
        Args:
            predictions: 预测视频 [B, T, C, H, W]
            targets: 真实视频
        
        Returns:
            metrics: 各项指标字典
        """
        B, T, C, H, W = predictions.shape
        
        psnrs = []
        ssims = []
        lpips_scores = []
        
        for t in range(T):
            pred_t = predictions[:, t]
            target_t = targets[:, t]
            
            # PSNR
            psnr = calculate_psnr_batch(pred_t, target_t)
            psnrs.append(psnr)
            
            # SSIM
            ssim = calculate_ssim(pred_t, target_t)
            ssims.append(ssim)
            
            # LPIPS
            # 归一化到 [-1, 1]
            pred_norm = pred_t * 2 - 1
            target_norm = target_t * 2 - 1
            lpips = self.lpips_evaluator.calculate(pred_norm, target_norm)
            lpips_scores.append(lpips)
        
        return {
            'psnr': psnrs,
            'ssim': ssims,
            'lpips': lpips_scores,
            'psnr_mean': sum(psnrs) / len(psnrs),
            'ssim_mean': sum(ssims) / len(ssims),
            'lpips_mean': sum(lpips_scores) / len(lpips_scores)
        }
```

## 14.7 代码实现示例

### 完整的视频预测训练流程

```python
"""
video_prediction_training.py
完整的视频预测训练示例
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import cv2


class VideoDataset(Dataset):
    """
    视频数据集
    
    从文件夹加载视频序列
    """
    def __init__(self, video_dir, sequence_length=10, prediction_horizon=5,
                 img_size=64, transform=None):
        self.video_dir = Path(video_dir)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.img_size = img_size
        self.transform = transform
        
        # 收集所有视频文件
        self.video_files = list(self.video_dir.glob('*.mp4'))
        
        # 提取帧序列
        self.sequences = []
        for video_file in self.video_files:
            frames = self.extract_frames(video_file)
            # 创建重叠序列
            for i in range(len(frames) - sequence_length - prediction_horizon):
                seq = frames[i:i + sequence_length + prediction_horizon]
                self.sequences.append(seq)
    
    def extract_frames(self, video_path, max_frames=100):
        """从视频中提取帧"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换为 RGB 并调整大小
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frame = frame.astype(np.float32) / 255.0  # 归一化到 [0, 1]
            
            frames.append(frame)
        
        cap.release()
        return frames
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        sequence = np.stack(sequence, axis=0)  # [T, H, W, C]
        sequence = np.transpose(sequence, (0, 3, 1, 2))  # [T, C, H, W]
        
        input_seq = sequence[:self.sequence_length]
        target_seq = sequence[self.sequence_length:]
        
        return {
            'input': torch.FloatTensor(input_seq),
            'target': torch.FloatTensor(target_seq)
        }


class SimpleVideoPredictor(nn.Module):
    """
    简化的视频预测模型
    
    使用 ConvLSTM 进行时空建模
    """
    def __init__(self, input_channels=3, hidden_channels=64,
                 prediction_horizon=5):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU()
        )
        
        # ConvLSTM
        self.conv_lstm = ConvLSTM(
            input_dim=hidden_channels,
            hidden_dims=[hidden_channels, hidden_channels * 2],
            kernel_size=(3, 3),
            num_layers=2
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, input_sequence):
        """
        Args:
            input_sequence: [B, T_in, C, H, W]
        """
        B, T_in, C, H, W = input_sequence.shape
        
        # 编码
        encoded = []
        for t in range(T_in):
            enc = self.encoder(input_sequence[:, t])
            encoded.append(enc)
        encoded = torch.stack(encoded, dim=1)
        
        # ConvLSTM
        _, hidden_states = self.conv_lstm(encoded)
        
        # 自回归预测
        predictions = []
        cur_input = encoded[:, -1]
        
        for t in range(self.prediction_horizon):
            _, hidden_states = self.conv_lstm(
                cur_input.unsqueeze(1), hidden_states
            )
            
            h = hidden_states[-1][0]
            pred = self.decoder(h)
            predictions.append(pred)
            
            cur_input = self.encoder(pred)
        
        return torch.stack(predictions, dim=1)


class VideoPredictionTrainer:
    """视频预测训练器"""
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = SimpleVideoPredictor(
            input_channels=config.get('input_channels', 3),
            hidden_channels=config.get('hidden_channels', 64),
            prediction_horizon=config.get('prediction_horizon', 5)
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4)
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 评估器
        self.evaluator = VideoPredictionEvaluator()
        
        # 训练统计
        self.best_loss = float('inf')
        self.history = []
    
    def compute_loss(self, predictions, targets):
        """
        计算组合损失
        
        包括：重建损失 + 感知损失 + 时序一致性损失
        """
        B, T, C, H, W = predictions.shape
        
        # 1. 重建损失（MSE）
        recon_loss = F.mse_loss(predictions, targets)
        
        # 2. 感知损失（使用 VGG 特征）
        # perceptual_loss = self.compute_perceptual_loss(predictions, targets)
        
        # 3. 时序一致性损失（相邻帧之间的平滑性）
        # temporal_loss = self.compute_temporal_loss(predictions)
        
        # 总损失
        total_loss = recon_loss  # + 0.1 * perceptual_loss + 0.01 * temporal_loss
        
        return total_loss, {'recon': recon_loss.item()}
    
    def train_epoch(self, data_loader):
        """训练一个 epoch"""
        self.model.train()
        epoch_losses = []
        
        for batch in data_loader:
            input_seq = batch['input'].to(self.device)
            target_seq = batch['target'].to(self.device)
            
            # 前向传播
            predictions = self.model(input_seq)
            
            # 计算损失
            loss, metrics = self.compute_loss(predictions, target_seq)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses)
    
    @torch.no_grad()
    def validate(self, data_loader):
        """验证"""
        self.model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []
        
        for batch in data_loader:
            input_seq = batch['input'].to(self.device)
            target_seq = batch['target'].to(self.device)
            
            predictions = self.model(input_seq)
            
            loss, _ = self.compute_loss(predictions, target_seq)
            val_losses.append(loss.item())
            
            all_predictions.append(predictions.cpu())
            all_targets.append(target_seq.cpu())
        
        # 计算评估指标
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.evaluator.evaluate(all_predictions, all_targets)
        
        return np.mean(val_losses), metrics
    
    def train(self, train_loader, val_loader, num_epochs):
        """完整训练循环"""
        print(f"开始在 {self.device} 上训练...")
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_metrics = self.validate(val_loader)
            
            # 记录历史
            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_psnr': val_metrics['psnr_mean'],
                'val_ssim': val_metrics['ssim_mean'],
                'val_lpips': val_metrics['lpips_mean']
            })
            
            # 打印日志
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  PSNR: {val_metrics['psnr_mean']:.2f} dB")
            print(f"  SSIM: {val_metrics['ssim_mean']:.4f}")
            print(f"  LPIPS: {val_metrics['lpips_mean']:.4f}")
            
            # 学习率调整
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save('best_model.pth')
                print(f"  ✓ 保存最佳模型")
        
        return self.history
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }, path)
        print(f"模型已保存到 {path}")
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', [])
        print(f"模型已从 {path} 加载")
    
    @torch.no_grad()
    def predict(self, input_sequence, num_future=None):
        """
        预测未来帧
        
        Args:
            input_sequence: 输入序列 [B, T, C, H, W] 或 [T, C, H, W]
        """
        self.model.eval()
        
        if len(input_sequence.shape) == 4:
            input_sequence = input_sequence.unsqueeze(0)
        
        input_sequence = input_sequence.to(self.device)
        predictions = self.model(input_sequence)
        
        return predictions.cpu()


# 使用示例
if __name__ == '__main__':
    # 配置
    config = {
        'input_channels': 3,
        'hidden_channels': 64,
        'prediction_horizon': 5,
        'learning_rate': 1e-4,
        'batch_size': 8,
        'num_epochs': 100,
        'device': 'cuda'
    }
    
    # 创建数据集
    # dataset = VideoDataset('path/to/videos', sequence_length=10)
    # train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 创建训练器
    trainer = VideoPredictionTrainer(config)
    
    # 训练
    # history = trainer.train(train_loader, val_loader, config['num_epochs'])
    
    print("视频预测训练框架已准备就绪！")
```

## 14.8 参考文献

### 核心论文

1. **ConvLSTM** (Shi et al., 2015)
   - "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
   - 首次将卷积引入 LSTM

2. **Video GPT** (Yan et al., 2021)
   - "Video GPT: Video Generation using VQ-VAE and Transformers"
   - 将 GPT 范式应用于视频生成

3. **VideoBERT** (Sun et al., 2019)
   - "VideoBERT: A Joint Model for Video and Language Representation Learning"
   - 视频的 BERT 式预训练

4. **JeSS** (相关自监督工作)
   - 自监督视频表示学习

### 视频预测方法

5. **PredNet** (Lotter et al., 2016)
   - "Deep Predictive Coding Networks for Video Prediction"
   - 预测编码框架

6. **SVG** (Denton & Fergus, 2018)
   - "Stochastic Video Generation with a Learned Prior"
   - 随机视频生成

7. **DVD** (Tulyan et al., 2020)
   - "Deep Video Dynamics"
   - 高质量视频预测

8. **PHD** (Gao et al., 2021)
   - "Probabilistic Hierarchical Dynamics"
   - 概率层次动态模型

### 评估指标

9. **LPIPS** (Zhang et al., 2018)
   - "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
   - 感知相似度指标

10. **SSIM** (Wang et al., 2004)
    - "Image Quality Assessment: From Error Visibility to Structural Similarity"
    - 结构相似性指标

### 综述

11. **"Deep Learning for Video Prediction"** (Wang et al., 2021)
    - 视频预测的全面综述

12. **"A Survey on Deep Learning Technique for Video Prediction"** (Liu et al., 2022)
    - 最新进展和技术分类

---

*第 14 章完。视频预测是理解动态视觉场景的关键技术，与世界模型密切相关。*
