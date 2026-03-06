"""
Transformer 架构完整实现
========================
基于论文 "Attention Is All You Need" (2017) 的 PyTorch 实现

作者：AI 前沿技术教程
适合：学习和理解 Transformer 核心组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 1. Self-Attention（自注意力机制）
# ============================================================================

class SelfAttention(nn.Module):
    """
    缩放点积注意力（Scaled Dot-Product Attention）
    
    公式：Attention(Q, K, V) = softmax(QK^T / √d_k) * V
    
    参数:
        d_model: 输入/输出维度
        dropout: Dropout 概率
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # 线性变换矩阵 W^Q, W^K, W^V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出线性变换
        self.W_o = nn.Linear(d_model, d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化"""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.W_k.bias)
        nn.init.zeros_(self.W_v.bias)
        nn.init.zeros_(self.W_o.bias)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状 (batch_size, seq_len, d_model)
            mask: 注意力掩码，形状 (batch_size, 1, seq_len) 或 (batch_size, seq_len, seq_len)
                  mask 值为 1 的位置会被遮蔽（注意力为 0）
        
        返回:
            输出张量，形状 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 步骤 1: 线性变换得到 Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)  # (batch, seq_len, d_model)
        V = self.W_v(x)  # (batch, seq_len, d_model)
        
        # 步骤 2: 计算注意力分数 QK^T / √d_k
        # 转置 K 以便矩阵乘法：(batch, d_model, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        # scores: (batch, seq_len, seq_len)
        
        # 步骤 3: 应用掩码（如果有）
        if mask is not None:
            # 将 mask 为 1 的位置设为负无穷，softmax 后变为 0
            scores = scores.masked_fill(mask == 1, float('-inf'))
        
        # 步骤 4: softmax 归一化
        attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights)
        
        # 步骤 5: 加权求和 V
        output = torch.matmul(attention_weights, V)  # (batch, seq_len, d_model)
        
        # 步骤 6: 输出线性变换
        output = self.W_o(output)
        
        return output


# ============================================================================
# 2. Multi-Head Attention（多头注意力）
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    将 d_model 维度分成 h 个头，每个头独立计算注意力，最后拼接
    
    公式：MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
    其中 head_i = Attention(Q*W_i^Q, K*W_i^K, V*W_i^V)
    
    参数:
        d_model: 模型维度（必须能被 n_heads 整除）
        n_heads: 注意力头数
        dropout: Dropout 概率
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 为每个头创建线性变换（可以合并为一个大的线性层）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化"""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.W_k.bias)
        nn.init.zeros_(self.W_v.bias)
        nn.init.zeros_(self.W_o.bias)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将最后一个维度拆分成多头
        
        输入：(batch, seq_len, d_model)
        输出：(batch, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # 交换维度，让 heads 在第二维
    
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将多头拼接回原始维度
        
        输入：(batch, n_heads, seq_len, d_k)
        输出：(batch, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 1, 3)  # 恢复维度顺序
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query: 查询张量，(batch, seq_len_q, d_model)
            key: 键张量，(batch, seq_len_k, d_model)
            value: 值张量，(batch, seq_len_v, d_model)
            mask: 注意力掩码，(batch, 1, seq_len_k) 或 (batch, seq_len_q, seq_len_k)
        
        返回:
            输出张量，(batch, seq_len_q, d_model)
        """
        batch_size = query.shape[0]
        
        # 步骤 1: 线性变换
        Q = self.W_q(query)  # (batch, seq_len_q, d_model)
        K = self.W_k(key)    # (batch, seq_len_k, d_model)
        V = self.W_v(value)  # (batch, seq_len_v, d_model)
        
        # 步骤 2: 拆分多头
        Q = self._split_heads(Q)  # (batch, n_heads, seq_len_q, d_k)
        K = self._split_heads(K)  # (batch, n_heads, seq_len_k, d_k)
        V = self._split_heads(V)  # (batch, n_heads, seq_len_v, d_k)
        
        # 步骤 3: 计算注意力分数
        # QK^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, n_heads, seq_len_q, seq_len_k)
        
        # 步骤 4: 应用掩码
        if mask is not None:
            # 扩展 mask 以匹配多头维度
            # mask 可能是 (batch, 1, seq_len_k) 或 (batch, seq_len_q, seq_len_k)
            # 需要扩展到 (batch, 1, seq_len_q, seq_len_k) 以便广播
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len) 或 (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 1, float('-inf'))
        
        # 步骤 5: softmax 和 dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 步骤 6: 加权求和
        output = torch.matmul(attention_weights, V)  # (batch, n_heads, seq_len_q, d_k)
        
        # 步骤 7: 拼接多头并线性变换
        output = self._combine_heads(output)  # (batch, seq_len_q, d_model)
        output = self.W_o(output)
        
        return output


# ============================================================================
# 3. Positional Encoding（位置编码）
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    
    使用不同频率的正弦和余弦函数为每个位置生成唯一编码
    
    公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    参数:
        d_model: 嵌入维度
        max_len: 最大序列长度
        dropout: Dropout 概率
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 位置序列 (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        
        # 计算频率分母：10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # div_term: (d_model/2,)
        
        # 计算正弦和余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos
        
        # 添加 batch 维度 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer（不参与梯度更新，但会随模型保存）
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将位置编码添加到输入嵌入
        
        参数:
            x: 输入嵌入，(batch, seq_len, d_model)
        
        返回:
            带位置编码的输出，(batch, seq_len, d_model)
        """
        # x: (batch, seq_len, d_model)
        # self.pe[:, :x.size(1), :]: (1, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# 4. Feed-Forward Network（前馈神经网络）
# ============================================================================

class FeedForward(nn.Module):
    """
    位置级前馈网络（Position-wise Feed-Forward Network）
    
    结构：Linear -> ReLU -> Linear -> Dropout
    
    参数:
        d_model: 输入/输出维度
        d_ff: 隐藏层维度（通常为 d_model 的 4 倍）
        dropout: Dropout 概率
    """
    
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化"""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，(batch, seq_len, d_model)
        
        返回:
            输出张量，(batch, seq_len, d_model)
        """
        # Linear -> ReLU -> Dropout -> Linear -> Dropout
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


# ============================================================================
# 5. Layer Normalization（层归一化）
# ============================================================================

class LayerNorm(nn.Module):
    """
    层归一化（Layer Normalization）
    
    注意：PyTorch 已有 nn.LayerNorm，这里展示手动实现
    
    参数:
        d_model: 特征维度
        eps: 数值稳定性常数
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 可学习的缩放和平移参数
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        层归一化
        
        参数:
            x: 输入张量，(batch, seq_len, d_model)
        
        返回:
            归一化输出，(batch, seq_len, d_model)
        """
        # 计算均值和标准差（在最后一个维度上）
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)    # (batch, seq_len, 1)
        
        # 归一化 + 缩放 + 平移
        x_norm = (x - mean) / (std + self.eps)
        return self.weight * x_norm + self.bias


# ============================================================================
# 6. Transformer Encoder Layer（编码器层）
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Transformer 编码器层
    
    结构（Pre-LN）：
        x → LayerNorm → MultiHeadAttention → Add → LayerNorm → FeedForward → Add → 输出
    
    参数:
        d_model: 模型维度
        n_heads: 注意力头数
        d_ff: 前馈网络隐藏层维度
        dropout: Dropout 概率
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # 多头自注意力
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 层归一化（Pre-LN 结构）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout（用于残差连接）
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，(batch, seq_len, d_model)
            mask: 注意力掩码，(batch, 1, seq_len) 或 (batch, seq_len, seq_len)
        
        返回:
            输出张量，(batch, seq_len, d_model)
        """
        seq_len = x.shape[1]
        
        # Pre-LN: 先归一化，再经过注意力
        x_norm = self.norm1(x)
        
        # 确保 mask 形状正确：(batch, 1, 1, seq_len) 用于广播
        if mask is not None and mask.dim() == 3:
            # (batch, 1, seq_len) -> (batch, 1, 1, seq_len)
            mask = mask.unsqueeze(2)
        
        attn_output = self.attention(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout(attn_output)  # 残差连接
        
        # Pre-LN: 先归一化，再经过前馈网络
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)  # 残差连接
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer 编码器（多层编码器层堆叠）
    
    参数:
        d_model: 模型维度
        n_heads: 注意力头数
        n_layers: 编码器层数
        d_ff: 前馈网络隐藏层维度
        dropout: Dropout 概率
        max_len: 最大序列长度
    """
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int = 6,
                 d_ff: int = 2048, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # 编码器层堆叠
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入嵌入，(batch, seq_len, d_model)
            mask: 注意力掩码（用于填充位置）
        
        返回:
            编码器输出，(batch, seq_len, d_model)
        """
        # 添加位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 逐层通过编码器
        for layer in self.layers:
            x = layer(x, mask)
        
        # 最终归一化
        x = self.norm(x)
        return x


# ============================================================================
# 7. Transformer Decoder Layer（解码器层）
# ============================================================================

class TransformerDecoderLayer(nn.Module):
    """
    Transformer 解码器层
    
    结构：
        1. Masked Multi-Head Attention（自注意力，带因果掩码）
        2. Cross Attention（关注编码器输出）
        3. Feed Forward
    
    参数:
        d_model: 模型维度
        n_heads: 注意力头数
        d_ff: 前馈网络隐藏层维度
        dropout: Dropout 概率
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # Masked 自注意力
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 交叉注意力（关注编码器输出）
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 层归一化（Pre-LN）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                causal_mask: torch.Tensor = None, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 解码器输入，(batch, tgt_len, d_model)
            encoder_output: 编码器输出，(batch, src_len, d_model)
            causal_mask: 因果掩码（防止看到未来位置）
            src_mask: 源序列掩码（填充位置）
        
        返回:
            解码器输出，(batch, tgt_len, d_model)
        """
        # 1. Masked 自注意力
        x_norm = self.norm1(x)
        
        # 处理因果掩码形状
        tgt_mask = causal_mask
        if tgt_mask is not None and tgt_mask.dim() == 3:
            tgt_mask = tgt_mask.unsqueeze(1)  # (batch, 1, tgt_len, tgt_len)
        
        attn_output = self.self_attention(x_norm, x_norm, x_norm, tgt_mask)
        x = x + self.dropout(attn_output)
        
        # 2. 交叉注意力
        x_norm = self.norm2(x)
        
        # 处理源序列掩码形状
        if src_mask is not None and src_mask.dim() == 3:
            src_mask = src_mask.unsqueeze(1)  # (batch, 1, 1, src_len)
        
        cross_output = self.cross_attention(x_norm, encoder_output, encoder_output, src_mask)
        x = x + self.dropout(cross_output)
        
        # 3. 前馈网络
        x_norm = self.norm3(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer 解码器（多层解码器层堆叠）
    
    参数:
        d_model: 模型维度
        n_heads: 注意力头数
        n_layers: 解码器层数
        d_ff: 前馈网络隐藏层维度
        dropout: Dropout 概率
        max_len: 最大序列长度
    """
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int = 6,
                 d_ff: int = 2048, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                causal_mask: torch.Tensor = None, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 目标序列嵌入，(batch, tgt_len, d_model)
            encoder_output: 编码器输出，(batch, src_len, d_model)
            causal_mask: 因果掩码
            src_mask: 源序列掩码
        
        返回:
            解码器输出，(batch, tgt_len, d_model)
        """
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask, src_mask)
        
        x = self.norm(x)
        return x


# ============================================================================
# 8. 完整 Transformer 模型
# ============================================================================

class Transformer(nn.Module):
    """
    完整的 Transformer 模型（Encoder-Decoder 架构）
    
    用于序列到序列任务（如机器翻译）
    
    参数:
        src_vocab_size: 源语言词表大小
        tgt_vocab_size: 目标语言词表大小
        d_model: 模型维度（默认 512）
        n_heads: 注意力头数（默认 8）
        n_layers: 编码器/解码器层数（默认 6）
        d_ff: 前馈网络隐藏层维度（默认 2048）
        dropout: Dropout 概率（默认 0.1）
        max_len: 最大序列长度（默认 5000）
    """
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, n_heads: int = 8, n_layers: int = 6,
                 d_ff: int = 2048, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        
        self.d_model = d_model
        
        # 输入嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 缩放因子（论文建议：嵌入 * √d_model）
        self.scale = math.sqrt(d_model)
        
        # 编码器
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers, d_ff, dropout, max_len)
        
        # 解码器
        self.decoder = TransformerDecoder(d_model, n_heads, n_layers, d_ff, dropout, max_len)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 权重绑定（可选）：输出层权重与目标嵌入共享
        # self.output_projection.weight = self.tgt_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化嵌入层"""
        nn.init.xavier_uniform_(self.src_embedding.weight)
        nn.init.xavier_uniform_(self.tgt_embedding.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        生成因果掩码（下三角矩阵，防止看到未来位置）
        
        返回:
            mask: (1, seq_len, seq_len)，1 表示遮蔽位置
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.unsqueeze(0)  # (1, seq_len, seq_len)
    
    def generate_padding_mask(self, x: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        """
        生成填充掩码
        
        参数:
            x: 输入序列，(batch, seq_len)
            pad_token_id: 填充 token 的 ID
        
        返回:
            mask: (batch, 1, seq_len)，1 表示填充位置
        """
        return (x == pad_token_id).unsqueeze(1)  # (batch, 1, seq_len)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_pad_mask: torch.Tensor = None, tgt_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播（训练时使用）
        
        参数:
            src: 源序列，(batch, src_len)
            tgt: 目标序列，(batch, tgt_len)
            src_pad_mask: 源序列填充掩码
            tgt_pad_mask: 目标序列填充掩码
        
        返回:
            输出 logits，(batch, tgt_len, tgt_vocab_size)
        """
        batch_size, src_len = src.shape
        tgt_len = tgt.shape[1]
        device = src.device
        
        # 生成因果掩码（防止解码器看到未来）
        causal_mask = self.generate_causal_mask(tgt_len, device)
        
        # 合并因果掩码和填充掩码
        if tgt_pad_mask is not None:
            # tgt_pad_mask: (batch, 1, tgt_len)
            # causal_mask: (1, tgt_len, tgt_len)
            # 扩展 causal_mask 到 batch 维度
            causal_mask_expanded = causal_mask.expand(batch_size, -1, -1)  # (batch, tgt_len, tgt_len)
            # 扩展 tgt_pad_mask 到 (batch, tgt_len, tgt_len)
            tgt_pad_expanded = tgt_pad_mask.expand(-1, tgt_len, -1)  # (batch, tgt_len, tgt_len)
            combined_mask = torch.max(causal_mask_expanded, tgt_pad_expanded)
        else:
            combined_mask = causal_mask.expand(batch_size, -1, -1)
        
        # 源序列填充掩码
        if src_pad_mask is None:
            src_pad_mask = self.generate_padding_mask(src)
        
        # 嵌入 + 缩放
        src_embedded = self.src_embedding(src) * self.scale  # (batch, src_len, d_model)
        tgt_embedded = self.tgt_embedding(tgt) * self.scale  # (batch, tgt_len, d_model)
        
        # 编码
        encoder_output = self.encoder(src_embedded, src_pad_mask)
        
        # 解码
        decoder_output = self.decoder(tgt_embedded, encoder_output, combined_mask, src_pad_mask)
        
        # 输出投影
        logits = self.output_projection(decoder_output)  # (batch, tgt_len, tgt_vocab_size)
        
        return logits
    
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        仅编码（用于推理时缓存编码器输出）
        
        参数:
            src: 源序列，(batch, src_len)
        
        返回:
            编码器输出，(batch, src_len, d_model)
        """
        src_pad_mask = self.generate_padding_mask(src)
        src_embedded = self.src_embedding(src) * self.scale
        return self.encoder(src_embedded, src_pad_mask)
    
    def decode_step(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
                    src_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        单步解码（用于自回归推理）
        
        参数:
            tgt: 当前已生成的目标序列，(batch, tgt_len)
            encoder_output: 缓存的编码器输出
            src_pad_mask: 源序列填充掩码
        
        返回:
            下一个 token 的 logits，(batch, 1, tgt_vocab_size)
        """
        tgt_len = tgt.shape[1]
        causal_mask = self.generate_causal_mask(tgt_len, tgt.device)
        
        tgt_embedded = self.tgt_embedding(tgt) * self.scale
        decoder_output = self.decoder(tgt_embedded, encoder_output, causal_mask, src_pad_mask)
        
        # 只返回最后一个位置的 logits
        logits = self.output_projection(decoder_output[:, -1:, :])
        return logits


# ============================================================================
# 9. 使用示例
# ============================================================================

def example_usage():
    """
    Transformer 使用示例
    """
    print("=" * 60)
    print("Transformer 使用示例")
    print("=" * 60)
    
    # 模型超参数
    src_vocab_size = 10000  # 源语言词表大小
    tgt_vocab_size = 10000  # 目标语言词表大小
    d_model = 512
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    batch_size = 32
    src_seq_len = 100
    tgt_seq_len = 80
    
    # 创建模型
    print(f"\n创建 Transformer 模型...")
    print(f"  - d_model: {d_model}")
    print(f"  - n_heads: {n_heads}")
    print(f"  - n_layers: {n_layers}")
    print(f"  - d_ff: {d_ff}")
    
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0.1
    )
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量:")
    print(f"  - 总参数：{total_params:,}")
    print(f"  - 可训练：{trainable_params:,}")
    
    # 创建示例输入
    print(f"\n创建示例输入...")
    print(f"  - batch_size: {batch_size}")
    print(f"  - src_seq_len: {src_seq_len}")
    print(f"  - tgt_seq_len: {tgt_seq_len}")
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))  # 0 保留给 PAD
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # 前向传播
    print(f"\n执行前向传播...")
    model.eval()
    with torch.no_grad():
        output = model(src, tgt)
    
    print(f"  - 输出形状：{output.shape}")
    print(f"    期望：(batch_size, tgt_seq_len, tgt_vocab_size) = ({batch_size}, {tgt_seq_len}, {tgt_vocab_size})")
    
    # 验证输出形状
    assert output.shape == (batch_size, tgt_seq_len, tgt_vocab_size), "输出形状错误!"
    print("  [OK] 输出形状正确!")
    
    # 测试编码 - 解码分离
    print(f"\n测试编码 - 解码分离...")
    encoder_output = model.encode(src)
    print(f"  - 编码器输出形状：{encoder_output.shape}")
    assert encoder_output.shape == (batch_size, src_seq_len, d_model)
    print("  [OK] 编码器输出形状正确!")
    
    # 测试单步解码
    tgt_prefix = tgt[:, :10]  # 取前 10 个 token 作为前缀
    next_logits = model.decode_step(tgt_prefix, encoder_output)
    print(f"  - 单步解码输出形状：{next_logits.shape}")
    assert next_logits.shape == (batch_size, 1, tgt_vocab_size)
    print("  [OK] 单步解码输出形状正确!")
    
    # 获取预测
    next_token = next_logits.argmax(dim=-1)
    print(f"  - 预测的下一个 token 形状：{next_token.shape}")
    
    print("\n" + "=" * 60)
    print("所有测试通过！[OK]")
    print("=" * 60)


def example_training():
    """
    简单的训练循环示例
    """
    print("\n" + "=" * 60)
    print("训练循环示例")
    print("=" * 60)
    
    # 创建模型和数据
    model = Transformer(src_vocab_size=1000, tgt_vocab_size=1000, d_model=256, n_heads=4, n_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 PAD
    
    # 模拟一个 batch
    src = torch.randint(1, 1000, (4, 20))
    tgt = torch.randint(1, 1000, (4, 15))
    tgt_input = tgt[:, :-1]  # 解码器输入（去掉最后一个）
    tgt_target = tgt[:, 1:]  # 目标输出（去掉第一个）
    
    # 前向传播
    model.train()
    output = model(src, tgt_input)
    
    # 计算损失
    # output: (batch, seq_len, vocab_size) -> (batch*seq_len, vocab_size)
    # tgt_target: (batch, seq_len) -> (batch*seq_len)
    loss = criterion(output.view(-1, output.shape[-1]), tgt_target.reshape(-1))
    
    print(f"  - 损失：{loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print("  [OK] 训练步骤完成!")


if __name__ == "__main__":
    # 运行示例
    example_usage()
    example_training()
    
    print("\n" + "=" * 60)
    print("代码执行完毕！")
    print("提示：可以根据需要调整超参数或添加自定义功能")
    print("=" * 60)
