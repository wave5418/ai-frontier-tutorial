#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenVLA 开源视觉 - 语言 - 动作模型实现
=====================================

本模块提供 OpenVLA 模型的完整实现，包括：
- 模型架构（视觉编码器、语言模型、动作解码器）
- 数据集加载与处理
- 训练流程
- 机器人控制推理
- 仿真环境集成

作者：AI 前沿技术教程
版本：1.0.0
日期：2024 年

注意：本实现用于教学目的，简化了部分工程细节
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json
import time


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class OpenVLAConfig:
    """OpenVLA 模型配置"""
    
    # 视觉编码器配置
    vision_backbone: str = "vit_b_16"  # ViT 骨干网络类型
    image_size: int = 224  # 输入图像分辨率
    vision_hidden_dim: int = 768  # 视觉特征维度
    
    # 语言模型配置
    language_model: str = "llama-2-7b"  # 语言模型类型
    vocab_size: int = 32000  # 词表大小
    language_hidden_dim: int = 4096  # 语言特征维度
    
    # 动作解码器配置
    action_dim: int = 7  # 动作维度（机器人自由度）
    action_bins: int = 256  # 每个动作维度的离散化 bin 数
    max_action_len: int = 7  # 最大动作序列长度
    
    # 融合层配置
    hidden_dim: int = 768  # 融合层隐藏维度
    num_heads: int = 8  # 注意力头数
    num_layers: int = 4  # 融合层层数
    dropout: float = 0.1  # Dropout 比例
    
    # 训练配置
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 100000


# ============================================================================
# 视觉编码器
# ============================================================================

class VisionEncoder(nn.Module):
    """
    视觉编码器：使用 ViT 提取图像特征
    
    输入：RGB 图像 [B, 3, H, W]
    输出：视觉特征 [B, N, D]，N 为 patch 数量，D 为特征维度
    """
    
    def __init__(self, config: OpenVLAConfig):
        super().__init__()
        self.config = config
        
        # 简化的 ViT 实现（教学用）
        # 实际应用中可使用 pretrainedmodels 或 timm 库
        patch_size = 16
        num_patches = (config.image_size // patch_size) ** 2
        
        # Patch 嵌入层
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=config.vision_hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, config.vision_hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.vision_hidden_dim))
        
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.vision_hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.vision_hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # 层归一化
        self.norm = nn.LayerNorm(config.vision_hidden_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: 输入图像 [B, 3, H, W]
        
        Returns:
            vision_features: 视觉特征 [B, N, D]
        """
        B = images.shape[0]
        
        # Patch 嵌入
        patches = self.patch_embed(images)  # [B, D, H', W']
        patches = patches.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # 添加 cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        patches = torch.cat([cls_tokens, patches], dim=1)  # [B, N+1, D]
        
        # 添加位置编码
        patches = patches + self.pos_embed
        
        # Transformer 编码
        patches = patches.transpose(0, 1)  # [N+1, B, D]
        features = self.transformer(patches)
        features = features.transpose(0, 1)  # [B, N+1, D]
        
        # 层归一化
        features = self.norm(features)
        
        # 移除 cls token，返回 patch 特征
        vision_features = features[:, 1:, :]  # [B, N, D]
        
        return vision_features


# ============================================================================
# 语言编码器（简化的 LLM）
# ============================================================================

class LanguageEncoder(nn.Module):
    """
    语言编码器：简化的语言模型
    
    输入：文本 token IDs [B, L]
    输出：语言特征 [B, L, D]
    """
    
    def __init__(self, config: OpenVLAConfig):
        super().__init__()
        self.config = config
        
        # Token 嵌入
        self.token_embed = nn.Embedding(config.vocab_size, config.language_hidden_dim)
        
        # 位置编码
        self.max_seq_len = 512
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_len, config.language_hidden_dim))
        
        # Transformer 层（简化版，实际应使用完整 LLM）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.language_hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.language_hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 层归一化
        self.norm = nn.LayerNorm(config.language_hidden_dim)
    
    def forward(self, tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            tokens: 输入 token IDs [B, L]
            attention_mask: 注意力掩码 [B, L]
        
        Returns:
            language_features: 语言特征 [B, L, D]
        """
        B, L = tokens.shape
        
        # Token 嵌入
        features = self.token_embed(tokens)  # [B, L, D]
        
        # 位置编码
        features = features + self.pos_embed[:, :L, :]
        
        # Transformer 编码
        features = features.transpose(0, 1)  # [L, B, D]
        features = self.transformer(features, src_key_padding_mask=~attention_mask.bool())
        features = features.transpose(0, 1)  # [B, L, D]
        
        # 层归一化
        features = self.norm(features)
        
        return features


# ============================================================================
# 多模态融合层
# ============================================================================

class MultimodalFusion(nn.Module):
    """
    多模态融合层：融合视觉和语言特征
    
    使用交叉注意力机制将视觉和语言信息对齐
    """
    
    def __init__(self, config: OpenVLAConfig):
        super().__init__()
        self.config = config
        
        # 投影层：将视觉和语言特征映射到统一维度
        self.vision_proj = nn.Linear(config.vision_hidden_dim, config.hidden_dim)
        self.language_proj = nn.Linear(config.language_hidden_dim, config.hidden_dim)
        
        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            vision_features: 视觉特征 [B, N_v, D_v]
            language_features: 语言特征 [B, N_l, D_l]
            attention_mask: 语言注意力掩码 [B, N_l]
        
        Returns:
            fused_features: 融合特征 [B, N_l, D]
        """
        # 特征投影
        vision_proj = self.vision_proj(vision_features)  # [B, N_v, D]
        language_proj = self.language_proj(language_features)  # [B, N_l, D]
        
        # 交叉注意力：语言特征 query，视觉特征 key/value
        attn_output, _ = self.cross_attention(
            query=language_proj,
            key=vision_proj,
            value=vision_proj,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        # 残差连接 + 层归一化
        x = self.norm1(language_proj + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(x)
        fused_features = self.norm2(x + ffn_output)
        
        return fused_features


# ============================================================================
# 动作解码器
# ============================================================================

class ActionDecoder(nn.Module):
    """
    动作解码器：自回归生成离散化动作
    
    将融合特征解码为动作序列，每个动作维度离散化为 256 个 bin
    """
    
    def __init__(self, config: OpenVLAConfig):
        super().__init__()
        self.config = config
        
        # 动作嵌入
        self.action_embed = nn.Embedding(config.action_bins, config.hidden_dim)
        
        # 位置编码（用于动作序列）
        self.pos_embed = nn.Parameter(torch.randn(1, config.max_action_len, config.hidden_dim))
        
        # Transformer 解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        # 动作预测头（每个维度一个分类头）
        self.action_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.action_bins)
            for _ in range(config.action_dim)
        ])
        
        # 层归一化
        self.norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(
        self,
        fused_features: torch.Tensor,
        action_targets: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播
        
        Args:
            fused_features: 融合特征 [B, N, D]
            action_targets: 目标动作 [B, action_dim]（训练时使用）
            teacher_forcing: 是否使用教师强制（训练时为 True）
        
        Returns:
            action_logits: 动作 logits 列表 [action_dim x [B, action_bins]]
        """
        B = fused_features.shape[0]
        
        if action_targets is not None and teacher_forcing:
            # 训练模式：使用真实动作作为输入
            action_embeds = self.action_embed(action_targets)  # [B, action_dim, D]
            action_embeds = action_embeds + self.pos_embed[:, :self.config.action_dim, :]
        else:
            # 推理模式：自回归生成
            action_embeds = []
            for i in range(self.config.action_dim):
                if i == 0:
                    # 第一个动作：使用 SOS token
                    sos_token = torch.zeros(B, 1, dtype=torch.long, device=fused_features.device)
                    embed = self.action_embed(sos_token)
                else:
                    # 后续动作：使用之前预测的动作
                    embed = self.action_embed(prev_action)
                action_embeds.append(embed)
                prev_action = None  # 会在循环中更新
            
            action_embeds = torch.cat(action_embeds, dim=1)
            action_embeds = action_embeds + self.pos_embed[:, :self.config.action_dim, :]
        
        # Transformer 解码
        # 将融合特征作为 memory，动作嵌入作为 target
        output = self.transformer(
            tgt=action_embeds,
            memory=fused_features.unsqueeze(1)  # [B, 1, N, D] -> [B, 1, D]
        )
        output = self.norm(output)  # [B, action_dim, D]
        
        # 预测每个动作维度的 logits
        action_logits = []
        for i, head in enumerate(self.action_heads):
            logits = head(output[:, i, :])  # [B, action_bins]
            action_logits.append(logits)
        
        return action_logits
    
    def predict(self, fused_features: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        推理：自回归生成动作
        
        Args:
            fused_features: 融合特征 [B, N, D]
            temperature: 采样温度（越低越确定性）
        
        Returns:
            actions: 预测动作 [B, action_dim]
        """
        B = fused_features.shape[0]
        device = fused_features.device
        
        actions = []
        
        for i in range(self.config.action_dim):
            if i == 0:
                # 第一个动作：使用 SOS token
                sos_token = torch.zeros(B, 1, dtype=torch.long, device=device)
                action_embed = self.action_embed(sos_token)
            else:
                # 后续动作：使用之前预测的动作
                action_embed = self.action_embed(actions[-1].unsqueeze(1))
            
            # 添加位置编码
            action_embed = action_embed + self.pos_embed[:, i:i+1, :]
            
            # Transformer 解码
            output = self.transformer(
                tgt=action_embed,
                memory=fused_features.unsqueeze(1)
            )
            output = self.norm(output)
            
            # 预测当前动作维度
            logits = self.action_heads[i](output[:, 0, :])  # [B, action_bins]
            
            # 应用温度采样
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)  # [B]
            else:
                action = logits.argmax(dim=-1)  # [B]
            
            actions.append(action)
        
        # 堆叠所有动作维度
        actions = torch.stack(actions, dim=1)  # [B, action_dim]
        
        return actions


# ============================================================================
# OpenVLA 主模型
# ============================================================================

class OpenVLA(nn.Module):
    """
    OpenVLA 主模型：整合视觉、语言、动作模块
    
    完整的视觉 - 语言 - 动作模型，支持训练和推理
    """
    
    def __init__(self, config: OpenVLAConfig):
        super().__init__()
        self.config = config
        
        # 子模块
        self.vision_encoder = VisionEncoder(config)
        self.language_encoder = LanguageEncoder(config)
        self.fusion = MultimodalFusion(config)
        self.action_decoder = ActionDecoder(config)
        
        # 动作离散化辅助函数
        self.action_bins = config.action_bins
    
    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播（训练模式）
        
        Args:
            images: 输入图像 [B, 3, H, W]
            tokens: 语言 token IDs [B, L]
            attention_mask: 注意力掩码 [B, L]
            actions: 目标动作 [B, action_dim]（可选）
        
        Returns:
            action_logits: 动作 logits 列表
        """
        # 视觉编码
        vision_features = self.vision_encoder(images)
        
        # 语言编码
        language_features = self.language_encoder(tokens, attention_mask)
        
        # 多模态融合
        fused_features = self.fusion(vision_features, language_features, attention_mask)
        
        # 动作解码
        action_logits = self.action_decoder(fused_features, actions, teacher_forcing=True)
        
        return action_logits
    
    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = 0.5
    ) -> torch.Tensor:
        """
        推理：预测动作
        
        Args:
            images: 输入图像 [B, 3, H, W]
            tokens: 语言 token IDs [B, L]
            attention_mask: 注意力掩码 [B, L]
            temperature: 采样温度
        
        Returns:
            actions: 预测动作 [B, action_dim]
        """
        # 视觉编码
        vision_features = self.vision_encoder(images)
        
        # 语言编码
        language_features = self.language_encoder(tokens, attention_mask)
        
        # 多模态融合
        fused_features = self.fusion(vision_features, language_features, attention_mask)
        
        # 动作预测
        actions = self.action_decoder.predict(fused_features, temperature)
        
        return actions
    
    def discretize_actions(self, continuous_actions: np.ndarray) -> np.ndarray:
        """
        将连续动作离散化
        
        Args:
            continuous_actions: 连续动作 [B, action_dim]，范围 [-1, 1]
        
        Returns:
            discrete_actions: 离散动作 [B, action_dim]，范围 [0, action_bins-1]
        """
        # 从 [-1, 1] 映射到 [0, action_bins-1]
        discrete = (continuous_actions + 1) / 2 * (self.action_bins - 1)
        discrete = np.clip(discrete, 0, self.action_bins - 1).astype(np.int64)
        return discrete
    
    def continuousize_actions(self, discrete_actions: np.ndarray) -> np.ndarray:
        """
        将离散动作连续化
        
        Args:
            discrete_actions: 离散动作 [B, action_dim]，范围 [0, action_bins-1]
        
        Returns:
            continuous_actions: 连续动作 [B, action_dim]，范围 [-1, 1]
        """
        # 从 [0, action_bins-1] 映射到 [-1, 1]
        continuous = discrete / (self.action_bins - 1) * 2 - 1
        return continuous.astype(np.float32)


# ============================================================================
# 数据集类
# ============================================================================

class RobotTrajectoryDataset(Dataset):
    """
    机器人轨迹数据集
    
    加载和处理机器人演示数据，包括图像、语言指令和动作序列
    """
    
    def __init__(
        self,
        data_path: str,
        config: OpenVLAConfig,
        tokenizer=None,
        augment: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据路径（JSON 或 RLDS 格式）
            config: 模型配置
            tokenizer: 语言 tokenizer（可选）
            augment: 是否使用数据增强
        """
        self.data_path = Path(data_path)
        self.config = config
        self.tokenizer = tokenizer
        self.augment = augment
        
        # 加载数据索引
        self.samples = self._load_index()
        
        # 图像预处理
        self.image_transform = self._get_image_transform()
    
    def _load_index(self) -> List[Dict]:
        """加载数据索引"""
        index_path = self.data_path / "index.json"
        
        if index_path.exists():
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 模拟数据（用于演示）
            return [{"id": i, "path": str(self.data_path / f"traj_{i}.npz")} 
                    for i in range(100)]
    
    def _get_image_transform(self):
        """获取图像预处理变换"""
        # 简化的预处理（实际应使用 torchvision.transforms）
        return lambda x: x
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            sample: 包含图像、token、动作等的字典
        """
        sample_info = self.samples[idx]
        
        # 加载数据（模拟）
        # 实际应从文件中加载
        image = np.random.rand(3, self.config.image_size, self.config.image_size).astype(np.float32)
        instruction = "pick up the object"
        actions = np.random.rand(self.config.action_dim).astype(np.float32) * 2 - 1
        
        # 图像预处理
        image = torch.from_numpy(image)
        
        # 语言 tokenization（模拟）
        if self.tokenizer:
            tokens = self.tokenizer.encode(instruction)
        else:
            # 简化的 tokenization
            tokens = [ord(c) % self.config.vocab_size for c in instruction[:100]]
            tokens = tokens + [0] * (100 - len(tokens))
        tokens = torch.tensor(tokens[:100], dtype=torch.long)
        
        # 注意力掩码
        attention_mask = torch.ones_like(tokens)
        
        # 动作离散化
        discrete_actions = self._discretize_actions(actions)
        discrete_actions = torch.tensor(discrete_actions, dtype=torch.long)
        
        return {
            "image": image,
            "tokens": tokens,
            "attention_mask": attention_mask,
            "actions": discrete_actions,
            "instruction": instruction
        }
    
    def _discretize_actions(self, actions: np.ndarray) -> np.ndarray:
        """离散化动作"""
        discrete = (actions + 1) / 2 * (self.config.action_bins - 1)
        discrete = np.clip(discrete, 0, self.config.action_bins - 1).astype(np.int64)
        return discrete


# ============================================================================
# 训练器
# ============================================================================

class Trainer:
    """
    OpenVLA 训练器
    
    管理训练流程，包括优化、评估、保存等
    """
    
    def __init__(
        self,
        model: OpenVLA,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[OpenVLAConfig] = None
    ):
        """
        初始化训练器
        
        Args:
            model: OpenVLA 模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集（可选）
            config: 训练配置
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or OpenVLAConfig()
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练状态
        self.step = 0
        self.best_loss = float('inf')
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        训练一个 epoch
        
        Args:
            dataloader: 数据加载器
        
        Returns:
            avg_loss: 平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # 准备数据
            images = batch["image"].to(self.device)
            tokens = batch["tokens"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            actions = batch["actions"].to(self.device)
            
            # 前向传播
            action_logits = self.model(images, tokens, attention_mask, actions)
            
            # 计算损失
            loss = 0.0
            for i, logits in enumerate(action_logits):
                action_loss = self.criterion(logits, actions[:, i])
                loss += action_loss
            
            loss = loss / len(action_logits)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            
            # 更新状态
            total_loss += loss.item()
            num_batches += 1
            self.step += 1
            
            # 日志
            if self.step % 100 == 0:
                print(f"Step {self.step}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """
        验证
        
        Args:
            dataloader: 数据加载器
        
        Returns:
            avg_loss: 平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            images = batch["image"].to(self.device)
            tokens = batch["tokens"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            actions = batch["actions"].to(self.device)
            
            action_logits = self.model(images, tokens, attention_mask, actions)
            
            loss = 0.0
            for i, logits in enumerate(action_logits):
                action_loss = self.criterion(logits, actions[:, i])
                loss += action_loss
            
            loss = loss / len(action_logits)
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs: int = 100):
        """
        完整训练流程
        
        Args:
            num_epochs: 训练 epoch 数
        """
        print(f"开始训练，设备：{self.device}")
        print(f"训练步数：{self.config.max_steps}")
        
        # 创建数据加载器
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4
            )
        
        # 训练循环
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            print(f"训练损失：{train_loss:.4f}")
            
            # 验证
            if val_loader:
                val_loss = self.validate(val_loader)
                print(f"验证损失：{val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("best_model.pth")
                    print(f"保存最佳模型，验证损失：{val_loss:.4f}")
            
            # 保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
            
            # 早停判断
            if self.step >= self.config.max_steps:
                print("达到最大训练步数，停止训练")
                break
        
        print("训练完成！")
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step,
            "best_loss": self.best_loss,
            "config": self.config
        }
        torch.save(checkpoint, path)
        print(f"检查点已保存：{path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        self.best_loss = checkpoint["best_loss"]
        print(f"检查点已加载：{path}")


# ============================================================================
# 机器人控制推理
# ============================================================================

class RobotController:
    """
    机器人控制器
    
    将 OpenVLA 模型输出转换为机器人控制命令
    """
    
    def __init__(self, model: OpenVLA, robot_config: Dict):
        """
        初始化控制器
        
        Args:
            model: OpenVLA 模型
            robot_config: 机器人配置（关节限制、运动学等）
        """
        self.model = model
        self.robot_config = robot_config
        self.device = next(model.parameters()).device
    
    def predict_action(
        self,
        image: np.ndarray,
        instruction: str,
        tokenizer=None,
        temperature: float = 0.5
    ) -> np.ndarray:
        """
        预测单个动作
        
        Args:
            image: 输入图像 [H, W, 3]
            instruction: 语言指令
            tokenizer: 语言 tokenizer
            temperature: 采样温度
        
        Returns:
            action: 连续动作 [action_dim]
        """
        self.model.eval()
        
        # 预处理图像
        image_tensor = self._preprocess_image(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]
        
        # Tokenize 语言指令
        if tokenizer:
            tokens = tokenizer.encode(instruction)
        else:
            tokens = [ord(c) % self.model.config.vocab_size for c in instruction[:100]]
            tokens = tokens + [0] * (100 - len(tokens))
        tokens = torch.tensor([tokens[:100]], dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(tokens)
        
        # 模型推理
        with torch.no_grad():
            discrete_actions = self.model.predict(
                image_tensor,
                tokens,
                attention_mask,
                temperature
            )
        
        # 转换为连续动作
        discrete_actions = discrete_actions.cpu().numpy()[0]
        continuous_actions = self.model.continuousize_actions(discrete_actions)
        
        return continuous_actions
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 调整大小
        image = cv2.resize(image, (self.model.config.image_size, self.model.config.image_size))
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        
        return torch.from_numpy(image)
    
    def execute_action(self, action: np.ndarray):
        """
        执行动作（模拟）
        
        实际应用中需要连接到真实机器人 API
        
        Args:
            action: 动作向量 [action_dim]
        """
        print(f"执行动作：{action}")
        # TODO: 连接到真实机器人
        # robot_api.send_command(action)


# ============================================================================
# 仿真环境集成
# ============================================================================

class SimulationEnv:
    """
    仿真环境
    
    集成常见的机器人仿真环境（如 PyBullet、MuJoCo、Isaac Gym）
    """
    
    def __init__(self, env_name: str = "widowx"):
        """
        初始化仿真环境
        
        Args:
            env_name: 环境名称
        """
        self.env_name = env_name
        self.env = None
        self._init_env()
    
    def _init_env(self):
        """初始化仿真环境"""
        # 这里使用简化的模拟环境
        # 实际应集成 PyBullet、MuJoCo 等
        print(f"初始化仿真环境：{self.env_name}")
        self.step_count = 0
        self.max_steps = 100
    
    def reset(self) -> Dict:
        """
        重置环境
        
        Returns:
            obs: 初始观测
        """
        self.step_count = 0
        
        # 模拟观测
        obs = {
            "image": np.random.rand(224, 224, 3).astype(np.float32),
            "state": np.random.rand(7).astype(np.float32),
            "instruction": "pick up the object"
        }
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        环境步进
        
        Args:
            action: 动作向量
        
        Returns:
            obs: 新观测
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        self.step_count += 1
        
        # 模拟环境响应
        obs = {
            "image": np.random.rand(224, 224, 3).astype(np.float32),
            "state": np.random.rand(7).astype(np.float32),
            "instruction": "pick up the object"
        }
        
        # 模拟奖励（基于动作平滑度）
        reward = -np.sum(np.abs(action)) * 0.01
        
        # 检查是否结束
        done = self.step_count >= self.max_steps
        
        info = {"step": self.step_count}
        
        return obs, reward, done, info
    
    def render(self):
        """渲染环境"""
        # 实际应显示仿真画面
        pass
    
    def close(self):
        """关闭环境"""
        print("关闭仿真环境")


# ============================================================================
# 完整使用示例
# ============================================================================

def example_training():
    """训练示例"""
    print("=" * 60)
    print("OpenVLA 训练示例")
    print("=" * 60)
    
    # 创建配置
    config = OpenVLAConfig(
        action_dim=7,
        batch_size=16,
        learning_rate=1e-4,
        max_steps=1000
    )
    
    # 创建模型
    model = OpenVLA(config)
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    
    # 创建数据集（模拟）
    train_dataset = RobotTrajectoryDataset(
        data_path="./data/train",
        config=config
    )
    
    val_dataset = RobotTrajectoryDataset(
        data_path="./data/val",
        config=config
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # 开始训练（仅演示，实际训练需要更长时间）
    print("\n开始训练...")
    # trainer.train(num_epochs=5)  # 取消注释以实际训练
    
    print("训练示例完成！")


def example_inference():
    """推理示例"""
    print("\n" + "=" * 60)
    print("OpenVLA 推理示例")
    print("=" * 60)
    
    # 创建配置
    config = OpenVLAConfig(action_dim=7)
    
    # 创建模型
    model = OpenVLA(config)
    model.eval()
    
    # 创建控制器
    robot_config = {
        "joint_limits": [(-3.14, 3.14)] * 7,
        "max_velocity": 1.0
    }
    controller = RobotController(model, robot_config)
    
    # 准备输入
    image = np.random.rand(224, 224, 3).astype(np.float32) * 255
    instruction = "pick up the red block and place it in the box"
    
    # 推理
    print(f"\n语言指令：{instruction}")
    action = controller.predict_action(image, instruction, temperature=0.5)
    
    print(f"预测动作：{action}")
    print(f"动作范围：[{action.min():.3f}, {action.max():.3f}]")
    
    # 执行（模拟）
    controller.execute_action(action)
    
    print("\n推理示例完成！")


def example_simulation():
    """仿真环境示例"""
    print("\n" + "=" * 60)
    print("OpenVLA 仿真环境示例")
    print("=" * 60)
    
    # 创建环境
    env = SimulationEnv("widowx")
    
    # 创建模型
    config = OpenVLAConfig(action_dim=7)
    model = OpenVLA(config)
    model.eval()
    
    # 创建控制器
    controller = RobotController(model, {})
    
    # 运行仿真
    obs = env.reset()
    total_reward = 0.0
    
    print("\n开始仿真...")
    for step in range(10):
        # 模型推理
        action = controller.predict_action(
            obs["image"],
            obs["instruction"],
            temperature=0.5
        )
        
        # 环境步进
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}: Reward = {reward:.4f}, Total = {total_reward:.4f}")
        
        if done:
            break
    
    env.close()
    print(f"\n仿真完成！总奖励：{total_reward:.4f}")


def main():
    """主函数：运行所有示例"""
    print("\n" + "=" * 60)
    print("OpenVLA 开源视觉 - 语言 - 动作模型")
    print("AI 前沿技术教程 - 第 18 章")
    print("=" * 60)
    
    # 运行示例
    example_training()
    example_inference()
    example_simulation()
    
    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 注意：需要安装以下依赖
    # pip install torch numpy opencv-python
    
    try:
        import cv2
    except ImportError:
        print("警告：未安装 opencv-python，部分功能可能不可用")
        print("请运行：pip install opencv-python")
    
    main()
