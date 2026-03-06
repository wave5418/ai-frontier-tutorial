#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT 模型完整实现
================

本文件包含 BERT 模型的完整 PyTorch 实现，包括：
- BERT Embedding 层
- BERT Encoder 层
- MLM（Masked Language Model）头
- NSP（Next Sentence Prediction）头
- 完整 BERT 模型
- 预训练示例代码
- 微调示例代码

适合学习和理解 BERT 内部结构。

作者：AI 前沿技术教程
日期：2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


# ============================================================================
# 1. BERT Embedding 层
# ============================================================================

class BERTEmbedding(nn.Module):
    """
    BERT 输入嵌入层
    
    将输入 token 转换为向量表示，包含三种嵌入的和：
    1. Token Embedding：词表嵌入
    2. Segment Embedding：句子 A/B 标识
    3. Position Embedding：位置编码
    
    输入形状：(batch_size, seq_length)
    输出形状：(batch_size, seq_length, hidden_size)
    """
    
    def __init__(self, 
                 vocab_size: int = 30522,      # 词表大小（BERT-Base）
                 hidden_size: int = 768,        # 隐藏层维度
                 max_position_embeddings: int = 512,  # 最大序列长度
                 type_vocab_size: int = 2,      # 句子类型数（A/B）
                 dropout: float = 0.1,
                 pad_token_id: int = 0):
        super().__init__()
        
        # Token 嵌入：将词 ID 映射到向量
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        
        # Segment 嵌入：区分句子 A 和句子 B
        self.segment_embedding = nn.Embedding(type_vocab_size, hidden_size)
        
        # Position 嵌入：位置编码（可学习）
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        
        # LayerNorm 和 Dropout
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
        # 注册位置 ID 缓冲区（避免每次创建）
        self.register_buffer(
            'position_ids', 
            torch.arange(max_position_embeddings).unsqueeze(0)
        )
    
    def forward(self, 
                input_ids: torch.Tensor,
                segment_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 词 ID 序列 (batch_size, seq_length)
            segment_ids: 句子类型 ID (batch_size, seq_length)，可选
            position_ids: 位置 ID (batch_size, seq_length)，可选
        
        Returns:
            embeddings: 嵌入向量 (batch_size, seq_length, hidden_size)
        """
        batch_size, seq_length = input_ids.shape
        
        # 获取 Token 嵌入
        token_embeds = self.token_embedding(input_ids)
        
        # 获取 Segment 嵌入
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        segment_embeds = self.segment_embedding(segment_ids)
        
        # 获取 Position 嵌入
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        position_embeds = self.position_embedding(position_ids)
        
        # 三种嵌入相加
        embeddings = token_embeds + segment_embeds + position_embeds
        
        # LayerNorm 和 Dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


# ============================================================================
# 2. BERT Encoder 层
# ============================================================================

class SelfAttention(nn.Module):
    """
    自注意力机制
    
    实现 Transformer 的核心注意力机制：
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads  # 每个头的维度
        
        assert hidden_size % num_attention_heads == 0, \
            "hidden_size 必须能被 num_attention_heads 整除"
        
        # Q, K, V 线性变换
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # 输出线性变换
        self.output = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = math.sqrt(self.head_size)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 输入 (batch_size, seq_length, hidden_size)
            attention_mask: 注意力掩码 (batch_size, 1, 1, seq_length)
        
        Returns:
            output: 注意力输出 (batch_size, seq_length, hidden_size)
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # 计算 Q, K, V
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # 多头拆分：(batch, seq, hidden) -> (batch, heads, seq, head_size)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        
        # 计算注意力分数：Q @ K^T
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / self.scale
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax 归一化
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 加权求和：Attention @ V
        context = torch.matmul(attention_probs, value)
        
        # 多头合并：(batch, heads, seq, head_size) -> (batch, seq, hidden)
        context = self._merge_heads(context)
        
        # 输出线性变换
        output = self.output(context)
        
        return output
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """拆分多头"""
        batch_size, seq_length, _ = x.shape
        x = x.view(batch_size, seq_length, self.num_heads, self.head_size)
        return x.permute(0, 2, 1, 3)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """合并多头"""
        batch_size, num_heads, seq_length, head_size = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_length, self.hidden_size)


class AttentionOutput(nn.Module):
    """
    注意力输出层
    
    包含残差连接和 LayerNorm
    """
    
    def __init__(self, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_output: torch.Tensor) -> torch.Tensor:
        """残差连接 + LayerNorm"""
        hidden_states = hidden_states + self.dropout(attention_output)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    """
    前馈神经网络
    
    两层 MLP，中间维度为 hidden_size * 4
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 dropout: float = 0.1):
        super().__init__()
        
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # GELU 激活函数
        self.activation = nn.GELU()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        intermediate = self.intermediate(hidden_states)
        intermediate = self.activation(intermediate)
        intermediate = self.dropout(intermediate)
        output = self.output(intermediate)
        output = self.dropout(output)
        return output


class BERTEncoderLayer(nn.Module):
    """
    BERT Encoder 单层
    
    结构：
    1. Multi-Head Self-Attention
    2. Add & LayerNorm
    3. Feed-Forward Network
    4. Add & LayerNorm
    """
    
    def __init__(self,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 dropout: float = 0.1):
        super().__init__()
        
        self.attention = SelfAttention(hidden_size, num_attention_heads, dropout)
        self.attention_output = AttentionOutput(hidden_size, dropout)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 输入 (batch_size, seq_length, hidden_size)
            attention_mask: 注意力掩码
        
        Returns:
            output: 输出 (batch_size, seq_length, hidden_size)
        """
        # Self-Attention + 残差
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.attention_output(hidden_states, attention_output)
        
        # Feed-Forward + 残差
        feed_forward_output = self.feed_forward(hidden_states)
        output = hidden_states + self.dropout(feed_forward_output)
        output = self.layer_norm(output)
        
        return output


class BERTEncoder(nn.Module):
    """
    BERT Encoder（多层）
    
    由 N 个 Encoder Layer 堆叠而成
    """
    
    def __init__(self,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            BERTEncoderLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_hidden_layers)
        ])
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                output_all_layers: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            hidden_states: 输入 (batch_size, seq_length, hidden_size)
            attention_mask: 注意力掩码
            output_all_layers: 是否返回所有层的输出
        
        Returns:
            output: 最后一层输出 (batch_size, seq_length, hidden_size)
            all_layer_outputs: 所有层输出（如果 output_all_layers=True）
        """
        all_layer_outputs = []
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            if output_all_layers:
                all_layer_outputs.append(hidden_states)
        
        if output_all_layers:
            return hidden_states, tuple(all_layer_outputs)
        return hidden_states


# ============================================================================
# 3. MLM（Masked Language Model）头
# ============================================================================

class MLMHead(nn.Module):
    """
    MLM 预测头
    
    用于预测被遮蔽的 Token
    
    结构：
    1. 线性变换（hidden_size -> hidden_size）
    2. GELU 激活
    3. LayerNorm
    4. 线性变换（hidden_size -> vocab_size）
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 vocab_size: int = 30522):
        super().__init__()
        
        # 预测层
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        
        # 激活函数
        self.activation = nn.GELU()
        
        # 偏置初始化为 0
        self.decoder.bias.data.zero_()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        预测被遮蔽的 Token
        
        Args:
            hidden_states: BERT 输出 (batch_size, seq_length, hidden_size)
        
        Returns:
            logits: 词表上的 logits (batch_size, seq_length, vocab_size)
        """
        # 线性变换 + 激活
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        
        # 映射到词表
        logits = self.decoder(hidden_states)
        
        return logits


# ============================================================================
# 4. NSP（Next Sentence Prediction）头
# ============================================================================

class NSPHead(nn.Module):
    """
    NSP 预测头
    
    用于预测句子对关系（IsNext / NotNext）
    
    使用 [CLS] token 的输出进行分类
    """
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        
        # 分类层
        self.classifier = nn.Linear(hidden_size, 2)  # 2 类：IsNext / NotNext
    
    def forward(self, cls_output: torch.Tensor) -> torch.Tensor:
        """
        预测句子关系
        
        Args:
            cls_output: [CLS] token 的输出 (batch_size, hidden_size)
        
        Returns:
            logits: 分类 logits (batch_size, 2)
        """
        return self.classifier(cls_output)


# ============================================================================
# 5. 完整 BERT 模型
# ============================================================================

class BERTModel(nn.Module):
    """
    完整的 BERT 模型
    
    包含：
    - Embedding 层
    - Encoder 层
    - MLM 头
    - NSP 头
    """
    
    def __init__(self,
                 vocab_size: int = 30522,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 max_position_embeddings: int = 512,
                 type_vocab_size: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        # 嵌入层
        self.embeddings = BERTEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            dropout=dropout
        )
        
        # Encoder
        self.encoder = BERTEncoder(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout=dropout
        )
        
        # MLM 头
        self.mlm_head = MLMHead(hidden_size, vocab_size)
        
        # NSP 头
        self.nsp_head = NSPHead(hidden_size)
        
        # 保存配置
        self.config = {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_hidden_layers,
            'num_attention_heads': num_attention_heads,
            'intermediate_size': intermediate_size,
            'max_position_embeddings': max_position_embeddings,
        }
    
    def forward(self,
                input_ids: torch.Tensor,
                segment_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                masked_positions: Optional[torch.Tensor] = None,
                output_all_layers: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 词 ID (batch_size, seq_length)
            segment_ids: 句子类型 ID (batch_size, seq_length)
            attention_mask: 注意力掩码 (batch_size, 1, 1, seq_length)
            masked_positions: 被遮蔽的位置 (batch_size, num_masked)
            output_all_layers: 是否返回所有层输出
        
        Returns:
            outputs: 包含以下键的字典
                - mlm_logits: MLM 预测 logits
                - nsp_logits: NSP 预测 logits
                - pooled_output: [CLS] 输出
                - all_layer_outputs: 所有层输出（可选）
        """
        # 1. 嵌入
        embedding_output = self.embeddings(input_ids, segment_ids)
        
        # 2. Encoder
        encoder_output = self.encoder(
            embedding_output, 
            attention_mask, 
            output_all_layers
        )
        
        if output_all_layers:
            sequence_output, all_layer_outputs = encoder_output
        else:
            sequence_output = encoder_output
            all_layer_outputs = None
        
        # 3. 获取 [CLS] 输出（用于 NSP 和分类任务）
        cls_output = sequence_output[:, 0, :]  # (batch_size, hidden_size)
        
        # 4. MLM 预测
        if masked_positions is not None:
            # 提取被遮蔽位置的输出
            batch_size, seq_length, hidden_size = sequence_output.shape
            masked_output = torch.gather(
                sequence_output, 
                1, 
                masked_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
            )
            mlm_logits = self.mlm_head(masked_output)
        else:
            mlm_logits = self.mlm_head(sequence_output)
        
        # 5. NSP 预测
        nsp_logits = self.nsp_head(cls_output)
        
        # 构建输出
        outputs = {
            'mlm_logits': mlm_logits,
            'nsp_logits': nsp_logits,
            'pooled_output': cls_output,
            'sequence_output': sequence_output,
        }
        
        if all_layer_outputs is not None:
            outputs['all_layer_outputs'] = all_layer_outputs
        
        return outputs


# ============================================================================
# 6. 预训练示例代码
# ============================================================================

def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    创建注意力掩码
    
    Args:
        input_ids: 输入词 ID
        pad_token_id: 填充 token ID
    
    Returns:
        attention_mask: (batch_size, 1, 1, seq_length)
    """
    # (batch_size, seq_length)
    mask = (input_ids != pad_token_id).float()
    
    # (batch_size, 1, 1, seq_length)
    mask = mask.unsqueeze(1).unsqueeze(2)
    
    # 转换为对数空间掩码（0 -> 0, 1 -> -inf）
    mask = (1.0 - mask) * -10000.0
    
    return mask


def create_mlm_labels(input_ids: torch.Tensor, 
                      mask_token_id: int,
                      mask_probability: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建 MLM 标签和掩码
    
    Args:
        input_ids: 输入词 ID
        mask_token_id: [MASK] token ID
        mask_probability: 掩码概率
    
    Returns:
        masked_input_ids: 被掩码的输入
        labels: 标签（被掩码位置为原词，其他为 -100）
    """
    batch_size, seq_length = input_ids.shape
    labels = input_ids.clone()
    masked_input_ids = input_ids.clone()
    
    # 创建随机掩码
    probability_matrix = torch.full_like(input_ids, mask_probability, dtype=torch.float)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # 标签中只保留被掩码位置
    labels[~masked_indices] = -100  # -100 表示忽略
    
    # 80% 替换为 [MASK]
    indices_replaced = torch.bernoulli(torch.full_like(input_ids, 0.8)).bool() & masked_indices
    masked_input_ids[indices_replaced] = mask_token_id
    
    # 10% 替换为随机词
    indices_random = torch.bernoulli(torch.full_like(input_ids, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint_like(input_ids, 1000)  # 假设词表大小 1000
    masked_input_ids[indices_random] = random_words[indices_random]
    
    # 10% 保持不变
    # （剩下的 10% 已经是原词，无需处理）
    
    return masked_input_ids, labels


def pretrain_example():
    """
    BERT 预训练示例
    
    演示如何构建预训练数据、计算损失、进行反向传播
    """
    print("=" * 60)
    print("BERT 预训练示例")
    print("=" * 60)
    
    # 模型配置
    vocab_size = 1000
    hidden_size = 256
    num_layers = 4
    num_heads = 4
    max_seq_length = 128
    batch_size = 8
    
    # 创建模型
    model = BERTModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=max_seq_length,
    )
    
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    
    # 模拟输入数据
    input_ids = torch.randint(1, vocab_size, (batch_size, max_seq_length))
    segment_ids = torch.zeros_like(input_ids)
    segment_ids[:, max_seq_length//2:] = 1  # 后半部分为句子 B
    
    # 创建注意力掩码
    attention_mask = create_attention_mask(input_ids)
    
    # 创建 MLM 掩码和标签
    mask_token_id = 100  # 假设 [MASK] 的 ID
    masked_input_ids, mlm_labels = create_mlm_labels(input_ids, mask_token_id)
    
    # 获取被掩码位置（用于 MLM 预测）
    masked_positions = (mlm_labels != -100).nonzero(as_tuple=False)
    if len(masked_positions) == 0:
        # 如果没有被掩码，随机选几个位置
        masked_positions = torch.randint(0, max_seq_length, (batch_size, 10))
    else:
        # 按 batch 分组
        masked_positions_list = []
        for i in range(batch_size):
            batch_masked = masked_positions[masked_positions[:, 0] == i, 1]
            if len(batch_masked) == 0:
                batch_masked = torch.tensor([0])
            masked_positions_list.append(batch_masked[:10])  # 最多 10 个
        # 补齐到相同长度
        max_masked = max(len(m) for m in masked_positions_list)
        masked_positions = torch.stack([
            F.pad(m, (0, max_masked - len(m)), value=0) 
            for m in masked_positions_list
        ])
    
    # 前向传播
    print("\n前向传播...")
    outputs = model(
        input_ids=masked_input_ids,
        segment_ids=segment_ids,
        attention_mask=attention_mask,
        masked_positions=masked_positions,
    )
    
    # 计算损失
    # MLM 损失
    mlm_logits = outputs['mlm_logits']  # (batch_size, num_masked, vocab_size)
    mlm_labels_flat = mlm_labels.view(-1)
    mlm_preds_flat = mlm_logits.view(-1, vocab_size)
    
    # 只计算被掩码位置的损失
    mlm_loss = F.cross_entropy(
        mlm_preds_flat, 
        mlm_labels_flat, 
        ignore_index=-100
    )
    
    # NSP 损失（模拟标签）
    nsp_logits = outputs['nsp_logits']  # (batch_size, 2)
    nsp_labels = torch.randint(0, 2, (batch_size,))  # 随机标签
    nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)
    
    # 总损失
    total_loss = mlm_loss + nsp_loss
    
    print(f"\nMLM 损失：{mlm_loss.item():.4f}")
    print(f"NSP 损失：{nsp_loss.item():.4f}")
    print(f"总损失：{total_loss.item():.4f}")
    
    # 反向传播
    print("\n反向传播...")
    total_loss.backward()
    
    # 打印梯度信息
    total_grad_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
    total_grad_norm = math.sqrt(total_grad_norm)
    print(f"梯度范数：{total_grad_norm:.4f}")
    
    print("\n预训练示例完成！")
    return model


# ============================================================================
# 7. 微调示例代码
# ============================================================================

class BERTForSequenceClassification(nn.Module):
    """
    BERT 用于句子分类任务
    
    在预训练 BERT 基础上添加分类层
    """
    
    def __init__(self, 
                 bert_model: BERTModel,
                 num_labels: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.bert = bert_model
        hidden_size = bert_model.config['hidden_size']
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self,
                input_ids: torch.Tensor,
                segment_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 词 ID
            segment_ids: 句子类型 ID
            attention_mask: 注意力掩码
            labels: 标签（可选）
        
        Returns:
            outputs: 包含 logits 和 loss（如果有 labels）
        """
        # BERT 前向传播
        outputs = self.bert(
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
        )
        
        # 获取 [CLS] 输出
        pooled_output = outputs['pooled_output']
        
        # 分类
        logits = self.classifier(pooled_output)
        
        result = {'logits': logits}
        
        # 计算损失
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result['loss'] = loss
        
        return result


class BERTForTokenClassification(nn.Module):
    """
    BERT 用于 Token 分类任务（如 NER）
    
    对每个 token 进行分类
    """
    
    def __init__(self, 
                 bert_model: BERTModel,
                 num_labels: int = 9,  # BIO 标注：B-PER, I-PER, B-ORG, I-ORG, ...
                 dropout: float = 0.1):
        super().__init__()
        
        self.bert = bert_model
        hidden_size = bert_model.config['hidden_size']
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self,
                input_ids: torch.Tensor,
                segment_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 词 ID (batch_size, seq_length)
            segment_ids: 句子类型 ID
            attention_mask: 注意力掩码
            labels: 标签 (batch_size, seq_length)
        
        Returns:
            outputs: 包含 logits 和 loss（如果有 labels）
        """
        # BERT 前向传播
        outputs = self.bert(
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
        )
        
        # 获取序列输出
        sequence_output = outputs['sequence_output']
        
        # 对每个 token 分类
        logits = self.classifier(sequence_output)
        
        result = {'logits': logits}
        
        # 计算损失
        if labels is not None:
            # 展平
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
            result['loss'] = loss
        
        return result


def finetune_classification_example():
    """
    BERT 微调示例（句子分类）
    
    演示如何加载预训练模型，添加分类层，进行微调
    """
    print("\n" + "=" * 60)
    print("BERT 微调示例（句子分类）")
    print("=" * 60)
    
    # 模型配置
    vocab_size = 1000
    hidden_size = 256
    num_layers = 4
    num_heads = 4
    max_seq_length = 64
    batch_size = 8
    num_labels = 2  # 二分类
    
    # 加载预训练 BERT
    print("加载预训练 BERT 模型...")
    bert_model = BERTModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=max_seq_length,
    )
    
    # 创建分类模型
    model = BERTForSequenceClassification(bert_model, num_labels=num_labels)
    
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    
    # 模拟训练数据
    input_ids = torch.randint(1, vocab_size, (batch_size, max_seq_length))
    segment_ids = torch.zeros_like(input_ids)
    attention_mask = create_attention_mask(input_ids)
    labels = torch.randint(0, num_labels, (batch_size,))
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # 模拟训练循环
    print("\n开始微调训练...")
    model.train()
    
    for step in range(5):
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs['loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # 计算准确率
        with torch.no_grad():
            predictions = outputs['logits'].argmax(dim=-1)
            accuracy = (predictions == labels).float().mean()
        
        print(f"Step {step+1}/5 - Loss: {loss.item():.4f} - Accuracy: {accuracy.item():.4f}")
    
    print("\n微调示例完成！")
    return model


def finetune_ner_example():
    """
    BERT 微调示例（NER 命名实体识别）
    
    演示 Token 级分类任务
    """
    print("\n" + "=" * 60)
    print("BERT 微调示例（NER 命名实体识别）")
    print("=" * 60)
    
    # 模型配置
    vocab_size = 1000
    hidden_size = 256
    num_layers = 4
    num_heads = 4
    max_seq_length = 64
    batch_size = 8
    num_labels = 9  # BIO 标注
    
    # 加载预训练 BERT
    print("加载预训练 BERT 模型...")
    bert_model = BERTModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=max_seq_length,
    )
    
    # 创建 NER 模型
    model = BERTForTokenClassification(bert_model, num_labels=num_labels)
    
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    
    # 模拟训练数据
    input_ids = torch.randint(1, vocab_size, (batch_size, max_seq_length))
    segment_ids = torch.zeros_like(input_ids)
    attention_mask = create_attention_mask(input_ids)
    labels = torch.randint(0, num_labels, (batch_size, max_seq_length))
    labels[:, -5:] = -100  # 填充部分忽略
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # 模拟训练循环
    print("\n开始 NER 微调训练...")
    model.train()
    
    for step in range(5):
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs['loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        print(f"Step {step+1}/5 - Loss: {loss.item():.4f}")
    
    print("\nNER 微调示例完成！")
    return model


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 设置随机种子（可复现）
    torch.manual_seed(42)
    
    # 运行预训练示例
    pretrain_model = pretrain_example()
    
    # 运行句子分类微调示例
    cls_model = finetune_classification_example()
    
    # 运行 NER 微调示例
    ner_model = finetune_ner_example()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
