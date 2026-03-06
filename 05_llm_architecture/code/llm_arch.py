"""
LLM 架构实现 - LLaMA 风格模型

本模块实现了 LLaMA 模型的核心组件：
- RMSNorm：均方根归一化
- RoPE：旋转位置编码
- SwiGLU：门控线性单元
- LLaMA Decoder 层
- 完整 LLM 模型
- KV Cache 推理示例

作者：AI 前沿技术教程
日期：2026-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


# ============================================================
# 1. RMSNorm - 均方根归一化
# ============================================================

class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)
    
    相比 LayerNorm，RMSNorm 省去了均值计算和偏置项，
    仅使用均方根进行归一化，速度更快且效果相当。
    
    公式：RMSNorm(x) = (x / RMS(x)) * γ
    其中：RMS(x) = sqrt(sum(x_i^2) / n + eps)
    
    参数:
        dim: 归一化的维度大小
        eps: 数值稳定性常数，防止除零
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 γ
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """计算 RMS 归一化"""
        # 计算均方根：sqrt(mean(x^2) + eps)
        # x.pow(2).mean(-1, keepdim=True) 计算每个样本的均方
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状 (batch, seq_len, dim)
        
        返回:
            归一化后的张量，形状同输入
        """
        # 转换为 float32 计算，保证数值稳定性
        output = self._norm(x.float()).type_as(x)
        # 应用可学习参数 γ
        return output * self.weight


# ============================================================
# 2. RoPE - 旋转位置编码
# ============================================================

class RotaryEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding) - 旋转位置编码
    
    核心思想：通过旋转矩阵将位置信息编码到查询和键向量中，
    使得注意力分数只依赖于相对位置，具有良好的外推性。
    
    数学原理：
    - 对于位置 m 的查询 q_m 和位置 n 的键 k_n
    - 应用旋转后：q_m^T * k_n = f(q, k, m-n)
    - 注意力分数仅依赖于相对位置 m-n
    
    参数:
        dim: 嵌入维度
        max_seq_len: 最大序列长度
        base: 旋转频率基数，默认 10000
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 计算旋转频率 theta
        # theta_i = base^(-2i/dim), i = 0, 1, ..., dim/2-1
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算所有位置的旋转角度
        # seq_len 维度用于支持不同长度的序列
        self._build_rotary_table(max_seq_len)
    
    def _build_rotary_table(self, max_seq_len: int):
        """构建旋转表"""
        # 位置索引：[0, 1, 2, ..., max_seq_len-1]
        seq = torch.arange(max_seq_len, device=self.inv_freq.device)
        
        # 计算每个位置的角度：outer product of seq and inv_freq
        # 形状：(max_seq_len, dim/2)
        freqs = torch.outer(seq, self.inv_freq)
        
        # 交错排列：[theta_0, theta_0, theta_1, theta_1, ...]
        # 用于后续的旋转操作
        self.register_buffer('freqs', freqs)
    
    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        """
        旋转向量的一半维度
        
        将向量分成两半，前半部分旋转 90 度：
        [x_0, x_1, ..., x_{d/2-1}, x_{d/2}, ..., x_{d-1}]
        → [-x_{d/2}, ..., -x_{d-1}, x_0, ..., x_{d/2-1}]
        
        参数:
            x: 输入张量，最后一维为偶数
        
        返回:
            旋转后的张量
        """
        # 将最后一维分成两半
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        # 旋转 90 度并拼接
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        应用旋转位置编码
        
        参数:
            x: 输入张量，形状 (batch, seq_len, dim) 或 (batch, heads, seq_len, dim)
            seq_len: 实际序列长度（用于动态长度）
        
        返回:
            应用 RoPE 后的张量
        """
        # 获取实际序列长度
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # 获取对应位置的频率
        # 形状：(seq_len, dim/2)
        freqs = self.freqs[:seq_len]
        
        # 计算旋转矩阵的 cos 和 sin
        # 形状：(seq_len, dim/2)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        # 扩展维度以匹配 x 的形状
        # x 形状：(batch, heads, seq_len, dim) 或 (batch, seq_len, dim)
        if x.dim() == 4:
            # (batch, heads, seq_len, dim/2)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            # (batch, seq_len, dim/2)
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        
        # 应用旋转：x_rot = x * cos + rotate(x) * sin
        x_rot = self._rotate(x)
        output = x * cos + x_rot * sin
        
        return output


# ============================================================
# 3. SwiGLU - 门控线性单元
# ============================================================

class SwiGLU(nn.Module):
    """
    SwiGLU (Swish Gated Linear Unit)
    
    SwiGLU 是 GLU 的变体，使用 Swish 激活函数：
    SwiGLU(x) = Swish(xW) ⊗ (xV) = (xW · σ(xW)) ⊗ (xV)
    
    其中：
    - W, V: 可学习权重矩阵
    - σ: Sigmoid 函数
    - ⊗: 逐元素乘法
    
    优势：
    - 比 ReLU/GeLU 表达力更强
    - 门控机制允许信息流动控制
    - 在 LLM 中表现优异
    
    参数:
        dim: 输入维度
        hidden_dim: 隐藏层维度（通常为 dim 的 2-4 倍）
    """
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # 门控路径：xW
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        # 值路径：xV
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        # 输出投影
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状 (batch, seq_len, dim)
        
        返回:
            输出张量，形状 (batch, seq_len, dim)
        """
        # 门控路径应用 Swish 激活
        # Swish(x) = x * sigmoid(x)
        gate = F.silu(self.gate_proj(x))
        
        # 值路径
        up = self.up_proj(x)
        
        # 逐元素相乘（门控）
        hidden = gate * up
        
        # 输出投影
        return self.down_proj(hidden)


# ============================================================
# 4. LLaMA 风格注意力机制
# ============================================================

class LlamaAttention(nn.Module):
    """
    LLaMA 风格的分组查询注意力（GQA）
    
    GQA 在 MHA（多头注意力）和 MQA（多查询注意力）之间取得平衡：
    - MHA: 每个头有独立的 K、V
    - MQA: 所有头共享一组 K、V
    - GQA: 将头分组，每组共享 K、V
    
    参数:
        dim: 模型维度
        n_heads: 查询头数量
        n_kv_heads: KV 头数量（GQA 的关键参数）
        max_seq_len: 最大序列长度
    """
    
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_seq_len: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        
        # 查询、键、值投影
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        # 旋转位置编码
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        
        # 缩放因子
        self.scale = math.sqrt(self.head_dim)
    
    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        重复 KV 头以匹配查询头数量
        
        例如：n_heads=8, n_kv_heads=2, n_rep=4
        将 2 个 KV 头重复 4 次得到 8 个
        
        参数:
            x: KV 张量，形状 (batch, seq_len, n_kv_heads, head_dim)
            n_rep: 重复次数
        
        返回:
            重复后的张量，形状 (batch, seq_len, n_heads, head_dim)
        """
        if n_rep == 1:
            return x
        
        batch, seq_len, n_kv_heads, head_dim = x.shape
        # 增加重复维度
        x = x.unsqueeze(3).expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
        # 合并头维度
        return x.reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        参数:
            x: 输入张量，形状 (batch, seq_len, dim)
            mask: 注意力掩码
            kv_cache: 历史 KV 缓存 (key_cache, value_cache)
            use_cache: 是否使用 KV Cache
        
        返回:
            output: 输出张量
            cache: 更新后的 KV 缓存（如果使用）
        """
        batch, seq_len, _ = x.shape
        
        # 投影到 Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头形状
        q = q.view(batch, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        
        # 应用 RoPE
        q = self.rope(q)
        k = self.rope(k)
        
        # 转置为 (batch, heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 处理 KV Cache
        if kv_cache is not None:
            key_cache, value_cache = kv_cache
            # 拼接历史 KV
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        
        # 更新 KV Cache
        new_cache = None
        if use_cache:
            new_cache = (k, v)
        
        # 重复 KV 头以匹配查询头
        n_rep = self.n_heads // self.n_kv_heads
        k = self._repeat_kv(k, n_rep)
        v = self._repeat_kv(v, n_rep)
        
        # 缩放点积注意力
        # scores: (batch, n_heads, seq_len_q, seq_len_kv)
        scores = torch.matmul(q, k.transpose(2, 3)) / self.scale
        
        # 应用掩码
        if mask is not None:
            scores = scores + mask
        
        # Softmax 注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力到值
        attn_output = torch.matmul(attn_weights, v)  # (batch, n_heads, seq_len, head_dim)
        
        # 转置回 (batch, seq_len, dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, self.dim)
        
        # 输出投影
        output = self.o_proj(attn_output)
        
        return output, new_cache


# ============================================================
# 5. LLaMA Decoder 层
# ============================================================

class LlamaDecoderLayer(nn.Module):
    """
    LLaMA Decoder 层
    
    结构：
    1. Pre-Norm (RMSNorm) → Attention → 残差
    2. Pre-Norm (RMSNorm) → SwiGLU FFN → 残差
    
    采用 Pre-Norm 结构，更稳定，适合深层网络。
    
    参数:
        dim: 模型维度
        n_heads: 注意力头数
        n_kv_heads: KV 头数（GQA）
        hidden_dim: FFN 隐藏层维度
        max_seq_len: 最大序列长度
    """
    
    def __init__(
        self, 
        dim: int, 
        n_heads: int, 
        n_kv_heads: int, 
        hidden_dim: int,
        max_seq_len: int
    ):
        super().__init__()
        # 注意力前的归一化
        self.attn_norm = RMSNorm(dim)
        # 注意力机制
        self.attn = LlamaAttention(dim, n_heads, n_kv_heads, max_seq_len)
        
        # FFN 前的归一化
        self.ffn_norm = RMSNorm(dim)
        # SwiGLU FFN
        self.ffn = SwiGLU(dim, hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        参数:
            x: 输入张量
            mask: 注意力掩码
            kv_cache: KV 缓存
            use_cache: 是否使用缓存
        
        返回:
            output: 输出张量
            cache: KV 缓存
        """
        # 自注意力子层（Pre-Norm + Attention + 残差）
        attn_input = self.attn_norm(x)
        attn_output, cache = self.attn(attn_input, mask, kv_cache, use_cache)
        x = x + attn_output  # 残差连接
        
        # FFN 子层（Pre-Norm + FFN + 残差）
        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_input)
        x = x + ffn_output  # 残差连接
        
        return x, cache


# ============================================================
# 6. 完整 LLM 模型
# ============================================================

class LlamaLLM(nn.Module):
    """
    完整的 LLaMA 风格语言模型
    
    结构：
    1. Token Embedding
    2. N 个 Decoder 层
    3. Final RMSNorm
    4. Output Projection (共享权重)
    
    参数:
        vocab_size: 词表大小
        dim: 模型维度
        n_layers: Decoder 层数
        n_heads: 注意力头数
        n_kv_heads: KV 头数
        hidden_dim: FFN 隐藏层维度
        max_seq_len: 最大序列长度
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: int = 8,  # GQA: 8 组 KV 头
        hidden_dim: int = 11008,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Token 嵌入
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # Decoder 层
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(dim, n_heads, n_kv_heads, hidden_dim, max_seq_len)
            for _ in range(n_layers)
        ])
        
        # 最终归一化
        self.norm = RMSNorm(dim)
        
        # 输出投影（与嵌入层共享权重）
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight  # 权重共享
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建因果注意力掩码
        
        确保每个位置只能看到之前的位置（自回归生成）
        
        参数:
            seq_len: 序列长度
            device: 设备
        
        返回:
            掩码张量，形状 (1, 1, seq_len, seq_len)
        """
        # 创建下三角矩阵
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        # 转换为 -inf 掩码
        mask = mask.masked_fill(mask == 1, float('-inf'))
        # 调整形状用于广播
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        前向传播
        
        参数:
            input_ids: 输入 token IDs，形状 (batch, seq_len)
            kv_caches: 每层的 KV 缓存列表
            use_cache: 是否使用 KV Cache
        
        返回:
            logits: 输出 logits，形状 (batch, seq_len, vocab_size)
            caches: 更新后的 KV 缓存列表
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # 创建因果掩码
        mask = self._create_causal_mask(seq_len, device)
        
        # Token 嵌入
        x = self.tok_embeddings(input_ids)  # (batch, seq_len, dim)
        
        # 存储 KV 缓存
        new_caches = [] if use_cache else None
        
        # 通过所有 Decoder 层
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            x, cache = layer(x, mask, kv_cache, use_cache)
            if use_cache:
                new_caches.append(cache)
        
        # 最终归一化
        x = self.norm(x)
        
        # 输出投影
        logits = self.output(x)  # (batch, seq_len, vocab_size)
        
        return logits, new_caches
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        自回归生成（使用 KV Cache）
        
        参数:
            input_ids: 输入 prompt 的 token IDs
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: 核采样参数
            pad_token_id: padding token ID
        
        返回:
            生成的完整序列（包含输入）
        """
        self.eval()
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # 初始化 KV Caches（每层一个）
        kv_caches = [None] * self.n_layers
        
        # 存储生成的 token
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # 前向传播（使用 KV Cache）
            logits, kv_caches = self.forward(generated, kv_caches, use_cache=True)
            
            # 获取最后一个 token 的 logits
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)
            
            # 温度采样
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Top-p 采样（核采样）
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除概率过低的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            # 检查是否全部生成结束符（简单处理）
            if (next_token == pad_token_id).all():
                break
            
            # 拼接到生成序列
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


# ============================================================
# 7. KV Cache 推理示例
# ============================================================

def demo_kv_cache():
    """
    KV Cache 推理示例
    
    演示如何使用 KV Cache 加速自回归生成。
    """
    print("=" * 60)
    print("KV Cache 推理示例")
    print("=" * 60)
    
    # 创建小型模型用于演示
    model = LlamaLLM(
        vocab_size=1000,
        dim=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,  # GQA
        hidden_dim=512,
        max_seq_len=128
    )
    
    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{total_params:,}")
    print(f"模型维度：{model.dim}")
    print(f"Decoder 层数：{model.n_layers}")
    print(f"注意力头数：{model.layers[0].attn.n_heads}")
    print(f"KV 头数：{model.layers[0].attn.n_kv_heads}")
    print()
    
    # 模拟输入
    batch_size = 2
    prompt_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, prompt_len))
    
    print(f"输入序列长度：{prompt_len}")
    print()
    
    # ========== 方式 1：不使用 KV Cache（慢） ==========
    print("--- 不使用 KV Cache ---")
    model.eval()
    
    import time
    start = time.time()
    
    with torch.no_grad():
        # 逐步生成，每次都重新计算所有历史
        generated_no_cache = input_ids.clone()
        for _ in range(20):
            logits, _ = model.forward(generated_no_cache, use_cache=False)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_no_cache = torch.cat([generated_no_cache, next_token], dim=1)
    
    time_no_cache = time.time() - start
    print(f"生成 20 tokens 耗时：{time_no_cache:.4f} 秒")
    print()
    
    # ========== 方式 2：使用 KV Cache（快） ==========
    print("--- 使用 KV Cache ---")
    
    start = time.time()
    
    with torch.no_grad():
        generated_with_cache = model.generate(input_ids, max_new_tokens=20)
    
    time_with_cache = time.time() - start
    print(f"生成 20 tokens 耗时：{time_with_cache:.4f} 秒")
    print()
    
    # ========== 对比 ==========
    print("=" * 60)
    print(f"加速比：{time_no_cache / time_with_cache:.2f}x")
    print(f"KV Cache 节省时间：{(1 - time_with_cache/time_no_cache) * 100:.1f}%")
    print("=" * 60)
    print()
    
    # ========== KV Cache 内存分析 ==========
    print("KV Cache 内存分析:")
    print("-" * 40)
    
    seq_len = 100  # 假设生成了 100 个 token
    head_dim = model.dim // model.layers[0].attn.n_heads
    n_kv_heads = model.layers[0].attn.n_kv_heads
    n_layers = model.n_layers
    
    # 每个 token 的 KV Cache 大小（以 float16 计）
    bytes_per_token = 2 * n_layers * 2 * n_kv_heads * head_dim * 2  # K+V, float16=2bytes
    
    print(f"序列长度：{seq_len}")
    print(f"每层 KV 头数：{n_kv_heads}")
    print(f"每头维度：{head_dim}")
    print(f"层数：{n_layers}")
    print(f"每 token KV Cache 大小：{bytes_per_token} bytes ({bytes_per_token/1024:.2f} KB)")
    print(f"总 KV Cache 大小：{bytes_per_token * seq_len} bytes ({bytes_per_token * seq_len / 1024 / 1024:.2f} MB)")
    print()


def demo_components():
    """
    组件功能演示
    """
    print("=" * 60)
    print("LLM 组件功能演示")
    print("=" * 60)
    print()
    
    # 1. RMSNorm 演示
    print("1. RMSNorm 演示")
    print("-" * 40)
    rmsnorm = RMSNorm(dim=128)
    x = torch.randn(2, 10, 128)
    output = rmsnorm(x)
    print(f"输入形状：{x.shape}")
    print(f"输出形状：{output.shape}")
    print(f"输入 RMS: {x.pow(2).mean(-1).sqrt().mean().item():.4f}")
    print(f"输出 RMS: {output.pow(2).mean(-1).sqrt().mean().item():.4f} (应接近 1.0)")
    print()
    
    # 2. RoPE 演示
    print("2. RoPE 旋转位置编码演示")
    print("-" * 40)
    rope = RotaryEmbedding(dim=64, max_seq_len=512)
    x = torch.randn(2, 8, 20, 64)  # (batch, heads, seq_len, dim)
    output = rope(x)
    print(f"输入形状：{x.shape}")
    print(f"输出形状：{output.shape}")
    print(f"RoPE 保持了输入形状，但编码了位置信息")
    print()
    
    # 3. SwiGLU 演示
    print("3. SwiGLU 激活函数演示")
    print("-" * 40)
    swiglu = SwiGLU(dim=128, hidden_dim=256)
    x = torch.randn(2, 10, 128)
    output = swiglu(x)
    print(f"输入形状：{x.shape}")
    print(f"输出形状：{output.shape}")
    print(f"SwiGLU 先扩展维度再投影回原维度")
    print()
    
    # 4. 注意力演示
    print("4. LLaMA 注意力（GQA）演示")
    print("-" * 40)
    attn = LlamaAttention(dim=256, n_heads=8, n_kv_heads=2, max_seq_len=128)
    x = torch.randn(2, 10, 256)
    output, cache = attn(x, use_cache=True)
    print(f"输入形状：{x.shape}")
    print(f"输出形状：{output.shape}")
    print(f"KV Cache: K={cache[0].shape}, V={cache[1].shape}")
    print(f"GQA: {attn.n_heads} 个查询头，{attn.n_kv_heads} 个 KV 头")
    print()
    
    print("=" * 60)


if __name__ == "__main__":
    # 设置随机种子以便复现
    torch.manual_seed(42)
    
    # 运行组件演示
    demo_components()
    
    print("\n")
    
    # 运行 KV Cache 推理示例
    demo_kv_cache()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
