#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP (Contrastive Language-Image Pre-training) 实现

本章代码实现了 CLIP 模型的核心组件：
- ImageEncoder: ViT 风格图像编码器
- TextEncoder: Transformer 文本编码器
- InfoNCELoss: 对比损失函数
- CLIP: 完整模型类
- 训练循环、零样本分类、图文检索示例

作者：AI 前沿技术教程
日期：2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from typing import List, Tuple, Optional
import numpy as np


# ============================================================================
# 1. ImageEncoder - ViT 风格图像编码器
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    图像分块嵌入层
    
    将图像分割成固定大小的 patches，并通过线性投影映射到嵌入空间。
    
    Args:
        image_size: 输入图像尺寸 (如 224)
        patch_size: 每个 patch 的尺寸 (如 16 或 32)
        in_channels: 输入通道数 (RGB 图像为 3)
        embed_dim: 嵌入维度
    """
    def __init__(self, image_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 512):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        
        # 使用卷积实现分块和投影
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [batch_size, channels, height, width]
        
        Returns:
            patch_embeddings: [batch_size, n_patches, embed_dim]
        """
        # [B, C, H, W] -> [B, embed_dim, H/patch, W/patch]
        x = self.projection(x)
        # [B, embed_dim, H/patch, W/patch] -> [B, embed_dim, n_patches]
        x = x.flatten(2)
        # [B, embed_dim, n_patches] -> [B, n_patches, embed_dim]
        x = x.transpose(1, 2)
        return x


class PositionEmbedding(nn.Module):
    """
    可学习位置编码
    
    为每个 patch 位置学习一个嵌入向量。
    """
    def __init__(self, n_positions: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_positions, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch_size, n_positions, embed_dim]
        
        Returns:
            x + position_embedding
        """
        batch_size = x.shape[0]
        positions = torch.arange(x.shape[1], device=x.device)
        return x + self.embedding(positions).unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    
    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        dropout: dropout 概率
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V 投影
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        # 输出投影
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入 [batch_size, seq_len, embed_dim]
            attn_mask: 注意力掩码 (可选)
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q, K, V [B, L, 3*embed_dim]
        qkv = self.qkv(x)
        # 分割成 Q, K, V [B, L, 3, H, head_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        # 转置为 [3, B, H, L, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数 [B, H, L, L]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和 [B, H, L, head_dim]
        out = torch.matmul(attn, v)
        # 合并头 [B, L, embed_dim]
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        
        return self.proj(out)


class TransformerBlock(nn.Module):
    """
    Transformer 编码器块
    
    包含多头自注意力和前馈网络，带有残差连接和层归一化。
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, 
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力 + 残差
        x = x + self.attention(self.norm1(x), attn_mask)
        # MLP + 残差
        x = x + self.mlp(self.norm2(x))
        return x


class ImageEncoder(nn.Module):
    """
    CLIP 图像编码器（ViT 风格）
    
    架构：
    Patch Embedding → Position Embedding → Transformer Encoder → LayerNorm → Projection
    
    Args:
        image_size: 输入图像尺寸
        patch_size: patch 尺寸
        in_channels: 输入通道数
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        num_layers: Transformer 层数
        mlp_ratio: MLP 隐藏层比例
        dropout: dropout 概率
    """
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 512,
                 num_heads: int = 8, num_layers: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 分块嵌入
        n_patches = (image_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        
        # 位置编码（+1 用于 [CLS] token）
        self.position_embedding = PositionEmbedding(n_patches + 1, embed_dim)
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer 编码器
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # 层归一化和投影
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [batch_size, channels, height, width]
        
        Returns:
            image_features: 图像特征 [batch_size, embed_dim]
        """
        batch_size = x.shape[0]
        
        # Patch Embedding [B, n_patches, embed_dim]
        x = self.patch_embedding(x)
        
        # 添加 [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 位置编码
        x = self.position_embedding(x)
        
        # Transformer 编码
        for block in self.transformer:
            x = block(x)
        
        # 层归一化
        x = self.layer_norm(x)
        
        # 取 [CLS] token 作为图像表示
        cls_features = x[:, 0]
        
        # 投影到对比空间
        image_features = self.projection(cls_features)
        
        return image_features


# ============================================================================
# 2. TextEncoder - Transformer 文本编码器
# ============================================================================

class TokenEmbedding(nn.Module):
    """
    词表嵌入层
    """
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class TextTransformerBlock(nn.Module):
    """
    文本 Transformer 块（带因果注意力）
    """
    def __init__(self, embed_dim: int, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TextEncoder(nn.Module):
    """
    CLIP 文本编码器（Transformer）
    
    架构：
    Token Embedding → Position Embedding → Transformer Encoder → LayerNorm → Projection
    
    Args:
        vocab_size: 词表大小
        embed_dim: 嵌入维度
        max_seq_len: 最大序列长度
        num_heads: 注意力头数
        num_layers: Transformer 层数
        mlp_ratio: MLP 隐藏层比例
        dropout: dropout 概率
    """
    def __init__(self, vocab_size: int = 49408, embed_dim: int = 512,
                 max_seq_len: int = 77, num_heads: int = 8,
                 num_layers: int = 12, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # 词嵌入
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        
        # 位置编码
        self.position_embedding = PositionEmbedding(max_seq_len, embed_dim)
        
        # Transformer 编码器
        self.transformer = nn.ModuleList([
            TextTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # 层归一化和投影
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)
        
        # 因果注意力掩码
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入 token IDs [batch_size, seq_len]
        
        Returns:
            text_features: 文本特征 [batch_size, embed_dim]
        """
        batch_size, seq_len = x.shape
        
        # 词嵌入
        x = self.token_embedding(x)
        
        # 位置编码
        x = self.position_embedding(x)
        
        # 因果注意力掩码
        attn_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        # Transformer 编码
        for block in self.transformer:
            x = block(x, attn_mask)
        
        # 层归一化
        x = self.layer_norm(x)
        
        # 取 EOS token 或最后一个非 padding 位置作为文本表示
        # 简单实现：取最后一个位置
        text_features = x[:, -1, :]
        
        # 投影到对比空间
        text_features = self.projection(text_features)
        
        return text_features


# ============================================================================
# 3. InfoNCELoss - 对比损失函数
# ============================================================================

class InfoNCELoss(nn.Module):
    """
    InfoNCE 对比损失
    
    公式：
    L = -log(exp(sim(v_i, t_i) / τ) / Σ_j exp(sim(v_i, t_j) / τ))
    
    其中 sim 为余弦相似度，τ 为温度参数。
    
    Args:
        temperature: 温度参数（可学习或固定）
    """
    def __init__(self, temperature: float = 0.07, learnable: bool = True):
        super().__init__()
        
        if learnable:
            # 可学习温度参数（对数空间，保证正数）
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(temperature)))
    
    @property
    def temperature(self):
        return self.log_temperature.exp()
    
    def forward(self, image_features: torch.Tensor, 
                text_features: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            image_features: 图像特征 [batch_size, embed_dim]
            text_features: 文本特征 [batch_size, embed_dim]
        
        Returns:
            loss: 对比损失值
            metrics: 额外指标（准确率等）
        """
        batch_size = image_features.shape[0]
        device = image_features.device
        
        # L2 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 计算相似度矩阵 [batch_size, batch_size]
        # logits[i, j] = sim(image_i, text_j)
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 标签：对角线为正样本
        labels = torch.arange(batch_size, device=device)
        
        # 对称交叉熵损失
        # 图像到文本
        loss_i2t = F.cross_entropy(logits, labels)
        # 文本到图像
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        # 平均损失
        loss = (loss_i2t + loss_t2i) / 2
        
        # 计算准确率
        with torch.no_grad():
            preds_i2t = logits.argmax(dim=1)
            preds_t2i = logits.T.argmax(dim=1)
            acc_i2t = (preds_i2t == labels).float().mean()
            acc_t2i = (preds_t2i == labels).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'acc_i2t': acc_i2t.item(),
            'acc_t2i': acc_t2i.item(),
            'temperature': self.temperature.item()
        }
        
        return loss, metrics


# ============================================================================
# 4. CLIP - 完整模型类
# ============================================================================

class CLIP(nn.Module):
    """
    CLIP 完整模型
    
    整合图像编码器、文本编码器和对比损失。
    
    Args:
        image_encoder: 图像编码器实例
        text_encoder: 文本编码器实例
        temperature: 温度参数
    """
    def __init__(self, image_encoder: ImageEncoder, text_encoder: TextEncoder,
                 temperature: float = 0.07):
        super().__init__()
        
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.criterion = InfoNCELoss(temperature)
        
        self.embed_dim = image_encoder.embed_dim
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        return self.image_encoder(image)
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """编码文本"""
        return self.text_encoder(text)
    
    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        前向传播并计算损失
        
        Args:
            image: 图像 batch [batch_size, channels, height, width]
            text: 文本 token IDs [batch_size, seq_len]
        
        Returns:
            loss: 对比损失
            metrics: 额外指标
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        return self.criterion(image_features, text_features)
    
    def compute_similarity(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        计算图像 - 文本相似度
        
        Args:
            image: 单张图像 [1, channels, height, width] 或 [batch, ...]
            text: 文本 token IDs [n_texts, seq_len]
        
        Returns:
            similarities: 相似度矩阵 [batch, n_texts]
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 计算余弦相似度
        similarities = torch.matmul(image_features, text_features.T)
        
        return similarities
    
    def zero_shot_classify(self, image: torch.Tensor, 
                           class_names: List[str],
                           text_templates: List[str] = None,
                           tokenizer=None) -> Tuple[str, torch.Tensor]:
        """
        零样本图像分类
        
        Args:
            image: 输入图像 [1, channels, height, width]
            class_names: 类别名称列表
            text_templates: 文本模板列表，如 ["a photo of a {}", "a picture of a {}"]
            tokenizer: 文本 tokenizer
        
        Returns:
            predicted_class: 预测类别名称
            similarities: 与各类别的相似度
        """
        if text_templates is None:
            text_templates = ["a photo of a {}"]
        
        self.eval()
        
        # 为每个类别生成文本提示并编码
        text_features_list = []
        with torch.no_grad():
            for class_name in class_names:
                # 生成多个模板提示
                prompts = [template.format(class_name) for template in text_templates]
                
                # 分词并编码
                if tokenizer:
                    text_tokens = tokenizer(prompts)
                else:
                    # 简化处理：假设已分词
                    text_tokens = prompts
                
                if isinstance(text_tokens, torch.Tensor):
                    features = self.encode_text(text_tokens)
                else:
                    # 如果是字符串列表，需要实际的分词器
                    raise ValueError("需要提供 tokenizer 将文本转换为 token IDs")
                
                # 平均多个模板的特征
                features = features.mean(dim=0, keepdim=True)
                text_features_list.append(features)
            
            # 拼接所有类别特征
            text_features = torch.cat(text_features_list, dim=0)
            
            # 编码图像
            image_features = self.encode_image(image)
            
            # 计算相似度
            similarities = self.compute_similarity(image, text_features)
            
            # 获取预测类别
            predicted_idx = similarities.argmax(dim=1).item()
            predicted_class = class_names[predicted_idx]
        
        return predicted_class, similarities
    
    def retrieve_images(self, query_text: str, image_database: torch.Tensor,
                        tokenizer=None) -> Tuple[List[int], torch.Tensor]:
        """
        文本查询图像检索
        
        Args:
            query_text: 查询文本
            image_database: 图像特征数据库 [n_images, embed_dim] 或原始图像
            tokenizer: 文本 tokenizer
        
        Returns:
            ranked_indices: 排序后的图像索引
            similarities: 相似度分数
        """
        self.eval()
        
        with torch.no_grad():
            # 编码查询文本
            if tokenizer:
                text_tokens = tokenizer([query_text])
            else:
                raise ValueError("需要提供 tokenizer")
            
            text_features = self.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            
            # 如果传入的是原始图像，先编码
            if image_database.dim() == 4:  # [N, C, H, W]
                image_features = self.encode_image(image_database)
            else:  # 已经是特征 [N, embed_dim]
                image_features = F.normalize(image_database, dim=-1)
            
            # 计算相似度
            similarities = torch.matmul(text_features, image_features.T).squeeze(0)
            
            # 排序
            ranked_indices = similarities.argsort(descending=True).tolist()
        
        return ranked_indices, similarities


# ============================================================================
# 5. 训练循环示例
# ============================================================================

class DummyDataset(Dataset):
    """
    示例数据集（实际使用时替换为真实数据）
    """
    def __init__(self, n_samples: int = 1000, image_size: int = 224):
        self.n_samples = n_samples
        self.image_size = image_size
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # 随机图像和文本（示例）
        image = torch.randn(3, self.image_size, self.image_size)
        text = torch.randint(0, 1000, (77,))  # 随机 token IDs
        return image, text


def train_clip(model: CLIP, train_loader: DataLoader, 
               val_loader: DataLoader = None,
               epochs: int = 10, lr: float = 1e-4,
               device: str = 'cuda', log_interval: int = 10):
    """
    CLIP 训练循环
    
    Args:
        model: CLIP 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        epochs: 训练轮数
        lr: 学习率
        device: 训练设备
        log_interval: 日志间隔
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 混合精度训练（可选）
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    print(f"开始训练 CLIP 模型...")
    print(f"设备：{device}, 学习率：{lr}, 轮数：{epochs}")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        n_batches = 0
        
        for batch_idx, (images, texts) in enumerate(train_loader):
            images = images.to(device)
            texts = texts.to(device)
            
            optimizer.zero_grad()
            
            # 混合精度训练
            if scaler:
                with torch.cuda.amp.autocast():
                    loss, metrics = model(images, texts)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, metrics = model(images, texts)
                loss.backward()
                optimizer.step()
            
            total_loss += metrics['loss']
            total_acc += (metrics['acc_i2t'] + metrics['acc_t2i']) / 2
            n_batches += 1
            
            # 日志
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / n_batches
                avg_acc = total_acc / n_batches
                print(f"Epoch {epoch+1}/{epochs} [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        
        # 学习率更新
        scheduler.step()
        
        # 验证
        if val_loader:
            model.eval()
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1} 验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        print("-" * 60)
    
    print("训练完成!")
    return model


def evaluate(model: CLIP, val_loader: DataLoader, device: str) -> Tuple[float, float]:
    """
    验证模型
    
    Returns:
        avg_loss: 平均损失
        avg_acc: 平均准确率
    """
    model.eval()
    total_loss = 0
    total_acc = 0
    n_batches = 0
    
    with torch.no_grad():
        for images, texts in val_loader:
            images = images.to(device)
            texts = texts.to(device)
            
            loss, metrics = model(images, texts)
            total_loss += metrics['loss']
            total_acc += (metrics['acc_i2t'] + metrics['acc_t2i']) / 2
            n_batches += 1
    
    return total_loss / n_batches, total_acc / n_batches


# ============================================================================
# 6. 零样本图像分类示例
# ============================================================================

def demo_zero_shot_classification():
    """
    零样本图像分类演示
    
    展示如何使用 CLIP 进行零样本分类。
    """
    print("=" * 60)
    print("零样本图像分类示例")
    print("=" * 60)
    
    # 创建模型
    image_encoder = ImageEncoder(
        image_size=224, patch_size=16, embed_dim=512,
        num_heads=8, num_layers=4  # 小模型用于演示
    )
    text_encoder = TextEncoder(
        vocab_size=1000, embed_dim=512, max_seq_len=77,
        num_heads=8, num_layers=4
    )
    model = CLIP(image_encoder, text_encoder)
    model.eval()
    
    # 定义类别
    classes = ["cat", "dog", "bird", "car", "tree"]
    templates = ["a photo of a {}", "a picture of a {}", "an image of a {}"]
    
    # 模拟输入（实际使用时替换为真实图像和 tokenizer）
    dummy_image = torch.randn(1, 3, 224, 224)
    
    print(f"类别列表：{classes}")
    print(f"输入图像形状：{dummy_image.shape}")
    print("\n注意：此示例需要实际的 tokenizer 才能运行完整流程")
    print("实际使用时，请提供 BPE tokenizer 将文本转换为 token IDs")
    
    # 模拟分类结果
    print("\n模拟分类结果:")
    for i, cls in enumerate(classes):
        score = np.random.random()
        print(f"  {cls}: {score:.4f}")
    
    print("\n提示：完整实现需要:")
    print("  1. 预训练的 CLIP 权重")
    print("  2. BPE tokenizer（如 CLIP 官方 tokenizer）")
    print("  3. 图像预处理（归一化、resize）")


# ============================================================================
# 7. 图像 - 文本检索示例
# ============================================================================

def demo_image_text_retrieval():
    """
    图像 - 文本检索演示
    
    展示如何使用 CLIP 进行图文检索。
    """
    print("=" * 60)
    print("图像 - 文本检索示例")
    print("=" * 60)
    
    # 创建模型
    image_encoder = ImageEncoder(
        image_size=224, patch_size=16, embed_dim=512,
        num_heads=8, num_layers=4
    )
    text_encoder = TextEncoder(
        vocab_size=1000, embed_dim=512, max_seq_len=77,
        num_heads=8, num_layers=4
    )
    model = CLIP(image_encoder, text_encoder)
    model.eval()
    
    # 模拟图像数据库
    n_images = 100
    image_database = torch.randn(n_images, 3, 224, 224)
    
    # 查询文本
    query = "a dog playing in the park"
    
    print(f"查询文本：{query}")
    print(f"图像数据库大小：{n_images} 张图像")
    print("\n注意：此示例需要实际的 tokenizer 才能运行完整流程")
    
    # 模拟检索结果
    print("\n模拟检索结果 (Top-5):")
    for i in range(5):
        idx = np.random.randint(0, n_images)
        score = np.random.random()
        print(f"  Rank {i+1}: 图像 #{idx}, 相似度：{score:.4f}")
    
    print("\n提示：完整实现需要:")
    print("  1. 预训练的 CLIP 权重")
    print("  2. BPE tokenizer")
    print("  3. 图像预处理")
    print("  4. 大规模图像特征数据库（可预先计算并缓存）")


# ============================================================================
# 8. 主函数 - 完整示例
# ============================================================================

def main():
    """
    主函数：演示 CLIP 的完整使用流程
    """
    print("=" * 70)
    print(" " * 20 + "CLIP 模型完整示例")
    print("=" * 70)
    
    # 1. 创建模型
    print("\n[1] 创建 CLIP 模型...")
    image_encoder = ImageEncoder(
        image_size=224, patch_size=16, embed_dim=512,
        num_heads=8, num_layers=4, dropout=0.1
    )
    text_encoder = TextEncoder(
        vocab_size=1000, embed_dim=512, max_seq_len=77,
        num_heads=8, num_layers=4, dropout=0.1
    )
    model = CLIP(image_encoder, text_encoder, temperature=0.07)
    
    # 打印模型信息
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{n_params:,}")
    print(f"嵌入维度：{model.embed_dim}")
    
    # 2. 准备数据
    print("\n[2] 准备示例数据...")
    dataset = DummyDataset(n_samples=100, image_size=224)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 3. 前向传播测试
    print("\n[3] 测试前向传播...")
    model.train()
    dummy_image = torch.randn(4, 3, 224, 224)
    dummy_text = torch.randint(0, 1000, (4, 77))
    
    with torch.no_grad():
        loss, metrics = model(dummy_image, dummy_text)
    
    print(f"损失：{loss.item():.4f}")
    print(f"图像→文本准确率：{metrics['acc_i2t']:.4f}")
    print(f"文本→图像准确率：{metrics['acc_t2i']:.4f}")
    print(f"温度参数：{metrics['temperature']:.4f}")
    
    # 4. 特征提取测试
    print("\n[4] 测试特征提取...")
    model.eval()
    with torch.no_grad():
        img_feat = model.encode_image(dummy_image)
        txt_feat = model.encode_text(dummy_text)
    
    print(f"图像特征形状：{img_feat.shape}")
    print(f"文本特征形状：{txt_feat.shape}")
    
    # 5. 相似度计算
    print("\n[5] 测试相似度计算...")
    similarities = model.compute_similarity(dummy_image, dummy_text)
    print(f"相似度矩阵形状：{similarities.shape}")
    print(f"对角线相似度（正样本）：{torch.diag(similarities).mean().item():.4f}")
    
    # 6. 演示应用
    print("\n[6] 应用演示...")
    demo_zero_shot_classification()
    print()
    demo_image_text_retrieval()
    
    # 7. 训练示例（简化）
    print("\n[7] 训练循环示例（1 个 epoch）...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备：{device}")
    
    # 注意：实际训练需要更多数据和轮数
    # train_clip(model, train_loader, epochs=1, device=device, log_interval=5)
    
    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)
    print("\n下一步:")
    print("  1. 使用真实数据集（如 COCO、Conceptual Captions）")
    print("  2. 添加 BPE tokenizer（参考 CLIP 官方实现）")
    print("  3. 加载预训练权重（OpenCLIP 或官方 CLIP）")
    print("  4. 进行微调或零样本评估")
    print("\n参考资源:")
    print("  - OpenCLIP: https://github.com/mlfoundations/open_clip")
    print("  - 原始论文：https://arxiv.org/abs/2103.00020")


if __name__ == "__main__":
    main()
