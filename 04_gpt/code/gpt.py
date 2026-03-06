#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT 与自回归生成 - 完整实现

本文件包含：
- Causal Self-Attention（因果自注意力）
- GPT Block（GPT 解码器块）
- 完整 GPT 模型
- 文本生成函数（支持 temperature、top-k、top-p）
- 训练示例
- 文本生成示例

作者：AI 前沿技术教程
日期：2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# ============================================================
# 1. Causal Self-Attention（因果自注意力）
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    因果自注意力层
    
    特点：
    - 使用因果掩码，确保每个位置只能关注自身及之前的位置
    - 多头注意力机制
    - 包含输出投影层
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        初始化因果自注意力层
        
        参数:
            d_model: 模型维度（隐藏层大小）
            n_heads: 注意力头数量
            dropout: Dropout 概率
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # 每个头的维度
        
        # Q, K, V 投影矩阵
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 输出投影矩阵
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子（防止点积过大）
        self.scale = math.sqrt(self.d_head)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状 (batch_size, seq_len, d_model)
            mask: 因果掩码，形状 (seq_len, seq_len) 或 (batch_size, 1, seq_len, seq_len)
        
        返回:
            输出张量，形状 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q, K, V
        # 形状：(batch_size, seq_len, d_model)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
        # 多头拆分：(batch_size, n_heads, seq_len, d_head)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # 计算注意力分数：Q @ K^T / sqrt(d_head)
        # 形状：(batch_size, n_heads, seq_len, seq_len)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # 应用因果掩码
        if mask is not None:
            attention_scores = attention_scores + mask
        
        # Softmax 归一化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和：attention_weights @ V
        # 形状：(batch_size, n_heads, seq_len, d_head)
        attention_output = torch.matmul(attention_weights, v)
        
        # 多头合并：(batch_size, seq_len, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 输出投影
        output = self.w_o(attention_output)
        
        return output


# ============================================================
# 2. GPT Block（GPT 解码器块）
# ============================================================

class GPTBlock(nn.Module):
    """
    GPT 解码器块
    
    结构：
    输入 → LayerNorm → MultiHeadAttention → 残差连接 → LayerNorm → FFN → 残差连接 → 输出
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        初始化 GPT Block
        
        参数:
            d_model: 模型维度
            n_heads: 注意力头数量
            d_ff: 前馈网络隐藏层维度（通常为 4 * d_model）
            dropout: Dropout 概率
        """
        super().__init__()
        
        # 层归一化 1（注意力前）
        self.ln1 = nn.LayerNorm(d_model)
        
        # 因果自注意力
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        
        # Dropout 1（注意力后）
        self.dropout1 = nn.Dropout(dropout)
        
        # 层归一化 2（FFN 前）
        self.ln2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU 激活函数
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Dropout 2（FFN 后）
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状 (batch_size, seq_len, d_model)
            mask: 因果掩码
        
        返回:
            输出张量，形状 (batch_size, seq_len, d_model)
        """
        # 注意力子层（预归一化 + 残差连接）
        attention_output = self.attention(self.ln1(x), mask)
        x = x + self.dropout1(attention_output)
        
        # 前馈子层（预归一化 + 残差连接）
        ffn_output = self.ffn(self.ln2(x))
        x = x + self.dropout2(ffn_output)
        
        return x


# ============================================================
# 3. 完整 GPT 模型
# ============================================================

class GPT(nn.Module):
    """
    完整 GPT 模型
    
    架构：
    Token Embedding + Position Embedding → [GPT Block × N] → LayerNorm → 输出投影
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        """
        初始化 GPT 模型
        
        参数:
            vocab_size: 词汇表大小
            max_seq_len: 最大序列长度
            d_model: 模型维度（默认 768，GPT-1/2-small 配置）
            n_layers: Transformer 层数（默认 12）
            n_heads: 注意力头数量（默认 12）
            d_ff: 前馈网络维度（默认 3072 = 4 * 768）
            dropout: Dropout 概率
            pad_token_id: Padding token 的 ID
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Token 嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # 位置嵌入（可学习）
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # GPT Block 堆叠
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 最终层归一化
        self.ln_final = nn.LayerNorm(d_model)
        
        # 输出投影（不共享权重）
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化（遵循 GPT 论文）"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建因果掩码
        
        参数:
            seq_len: 序列长度
            device: 设备（CPU/GPU）
        
        返回:
            因果掩码，形状 (1, 1, seq_len, seq_len)
        """
        # 创建上三角掩码（对角线以上为 1，以下为 0）
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        # 转换为 -inf（禁止关注的位置）
        mask = mask.masked_fill(mask == 1, float('-inf'))
        # 调整形状以匹配注意力分数
        mask = mask.view(1, 1, seq_len, seq_len)
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        参数:
            input_ids: 输入 token IDs，形状 (batch_size, seq_len)
            targets: 目标 token IDs（用于训练），形状 (batch_size, seq_len)
        
        返回:
            logits: 输出 logits，形状 (batch_size, seq_len, vocab_size)
            loss: 交叉熵损失（如果提供了 targets）
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 创建因果掩码
        mask = self.create_causal_mask(seq_len, device)
        
        # Token 嵌入 + 位置嵌入
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # 通过所有 GPT Block
        for block in self.blocks:
            x = block(x, mask)
        
        # 最终层归一化
        x = self.ln_final(x)
        
        # 输出投影
        logits = self.head(x)
        
        # 计算损失（如果提供了 targets）
        loss = None
        if targets is not None:
            # 调整形状以计算交叉熵
            logits_flat = logits.view(-1, self.vocab_size)
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.pad_token_id)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        自回归文本生成
        
        参数:
            input_ids: 输入 prompt 的 token IDs，形状 (batch_size, seq_len)
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数（控制随机性）
            top_k: Top-k 采样（如果为 None 则不使用）
            top_p: Top-p 采样（如果为 None 则不使用）
            eos_token_id: 结束 token 的 ID
        
        返回:
            生成的完整序列（包括输入），形状 (batch_size, seq_len + new_tokens)
        """
        model.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 迭代生成
        for _ in range(max_new_tokens):
            # 截取最后 max_seq_len 个 token（防止超出位置嵌入范围）
            input_cond = input_ids[:, -self.max_seq_len:]
            
            # 前向传播
            logits, _ = self.forward(input_cond)
            
            # 取最后一个位置的 logits
            logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # 应用温度
            if temperature != 1.0:
                logits = logits / temperature
            
            # Top-k 采样
            if top_k is not None:
                logits = self._apply_top_k(logits, top_k)
            
            # Top-p 采样
            if top_p is not None:
                logits = self._apply_top_p(logits, top_p)
            
            # 转换为概率分布
            probs = F.softmax(logits, dim=-1)
            
            # 采样下一个 token
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            
            # 拼接到输入序列
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 检查是否生成结束 token
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break
        
        return input_ids
    
    def _apply_top_k(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """应用 Top-k 采样"""
        top_k_values, _ = torch.topk(logits, k, dim=-1)
        min_val = top_k_values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_val, torch.tensor(float('-inf'), device=logits.device), logits)
        return logits
    
    def _apply_top_p(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """应用 Top-p (Nucleus) 采样"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 找到截断点
        mask = cumsum_probs > p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False
        
        # 将截断后的 logits 设为 -inf
        sorted_logits = sorted_logits.masked_fill(mask, float('-inf'))
        
        # 还原原始顺序
        scattered_logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)
        return scattered_logits


# ============================================================
# 4. 文本生成函数（封装）
# ============================================================

def generate_text(
    model: GPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> str:
    """
    文本生成封装函数
    
    参数:
        model: GPT 模型
        tokenizer: Tokenizer（需支持 encode/decode）
        prompt: 输入提示文本
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数
        top_k: Top-k 采样
        top_p: Top-p 采样
    
    返回:
        生成的文本
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # 获取结束 token ID
    eos_token_id = getattr(tokenizer, 'eos_token_id', None)
    
    # 生成
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_token_id
    )
    
    # 解码输出
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text


# ============================================================
# 5. 训练示例
# ============================================================

def train_example():
    """
    训练示例
    
    演示如何训练一个小型 GPT 模型
    """
    print("=" * 60)
    print("GPT 训练示例")
    print("=" * 60)
    
    # 配置
    vocab_size = 1000  # 小型词汇表（示例用）
    max_seq_len = 128
    d_model = 256
    n_layers = 4
    n_heads = 8
    d_ff = 1024
    batch_size = 16
    learning_rate = 3e-4
    num_epochs = 10
    
    # 创建模型
    model = GPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.1
    )
    
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 模拟训练数据
    # 实际应用中应使用真实文本数据
    print("\n开始训练...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for step in range(10):  # 每 epoch 10 步
            # 生成随机训练数据
            input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_len))
            targets = input_ids.clone()  # 语言建模：预测下一个 token
            
            # 前向传播
            logits, loss = model(input_ids, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / 10
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("\n训练完成！")
    return model


# ============================================================
# 6. 文本生成示例
# ============================================================

def generate_example():
    """
    文本生成示例
    
    演示如何使用训练好的模型进行文本生成
    """
    print("\n" + "=" * 60)
    print("GPT 文本生成示例")
    print("=" * 60)
    
    # 配置（与训练示例一致）
    vocab_size = 1000
    max_seq_len = 128
    d_model = 256
    n_layers = 4
    n_heads = 8
    d_ff = 1024
    
    # 创建模型
    model = GPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.1
    )
    model.eval()
    
    # 简单 tokenizer 模拟（实际应使用 BPE 等）
    class SimpleTokenizer:
        def encode(self, text, return_tensors=None):
            # 简单字符级编码（示例用）
            ids = [ord(c) % vocab_size for c in text]
            ids = [1] + ids  # 添加 BOS token
            if return_tensors == 'pt':
                return torch.tensor([ids])
            return ids
        
        def decode(self, ids, skip_special_tokens=True):
            # 简单字符级解码
            ids = ids.tolist() if hasattr(ids, 'tolist') else ids
            chars = [chr(i % 128) for i in ids if i not in [0, 1]]
            return ''.join(chars)
        
        @property
        def eos_token_id(self):
            return 2
    
    tokenizer = SimpleTokenizer()
    
    # 测试不同采样策略
    prompts = [
        "Hello, ",
        "Once upon a time",
        "The answer is"
    ]
    
    strategies = [
        {"temperature": 0.5, "top_k": None, "top_p": None, "name": "Greedy (T=0.5)"},
        {"temperature": 1.0, "top_k": None, "top_p": None, "name": "Sampling (T=1.0)"},
        {"temperature": 1.0, "top_k": 50, "top_p": None, "name": "Top-k (k=50)"},
        {"temperature": 1.0, "top_k": None, "top_p": 0.9, "name": "Top-p (p=0.9)"},
        {"temperature": 0.8, "top_k": 40, "top_p": 0.9, "name": "Combined (T=0.8, k=40, p=0.9)"},
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        
        for strategy in strategies:
            # 编码输入
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            
            # 生成
            output_ids = model.generate(
                input_ids,
                max_new_tokens=30,
                temperature=strategy["temperature"],
                top_k=strategy["top_k"],
                top_p=strategy["top_p"],
                eos_token_id=tokenizer.eos_token_id
            )
            
            # 解码
            generated = tokenizer.decode(output_ids[0])
            print(f"{strategy['name']:30s}: {generated[:60]}...")
    
    print("\n生成示例完成！")


# ============================================================
# 7. 完整使用示例（主函数）
# ============================================================

def main():
    """
    主函数：演示完整流程
    """
    print("\n" + "🎯" * 30)
    print("GPT 与自回归生成 - 完整教程")
    print("🎯" * 30 + "\n")
    
    # 1. 训练示例
    model = train_example()
    
    # 2. 生成示例
    generate_example()
    
    # 3. 架构说明
    print("\n" + "=" * 60)
    print("GPT 架构总结")
    print("=" * 60)
    print("""
    GPT 模型核心组件：
    
    1. Token Embedding: 将离散 token 映射为连续向量
    2. Position Embedding: 可学习的位置编码
    3. Causal Self-Attention: 因果自注意力（掩码防止看到未来）
    4. Multi-Head Attention: 多头机制，捕捉不同子空间信息
    5. Feed-Forward Network: 位置无关的前馈网络（GELU 激活）
    6. Layer Normalization: 层归一化（预归一化架构）
    7. Residual Connection: 残差连接，缓解梯度消失
    
    文本生成策略：
    
    - Greedy Search: 每步选择概率最高的 token
    - Beam Search: 维护 k 个候选序列，全局更优
    - Sampling: 从概率分布中随机采样
    - Temperature: 调节分布尖锐度（T<1 更确定，T>1 更多样）
    - Top-k: 仅从 top-k 个 token 中采样
    - Top-p: 从累积概率达到 p 的最小集合中采样（Nucleus）
    
    In-context Learning:
    
    - Zero-shot: 仅提供任务描述
    - One-shot: 提供 1 个示例
    - Few-shot: 提供多个示例（通常 3-5 个）
    """)
    
    print("\n✅ 教程完成！")
    print("📚 更多信息请参考 theory.md")


if __name__ == "__main__":
    main()
