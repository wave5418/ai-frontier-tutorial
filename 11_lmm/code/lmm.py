# -*- coding: utf-8 -*-
"""
第 11 章 多模态大模型 (LMM) 代码实现
=====================================
包含：图像编码器、投影层、多模态 LLM、训练流程

作者：AI 前沿技术教程
日期：2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


# ============================================================
# 1. 图像编码器 (基于 ViT/CLIP)
# ============================================================

class VisionTransformer(nn.Module):
    """
    简化的 Vision Transformer 实现
    用作多模态模型的图像编码器
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        
        # Patch Embedding
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            num_channels, hidden_size, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Class Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Position Embedding
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, hidden_size)
        )
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            ViTLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_hidden_layers)
        ])
        
        self.layernorm = nn.LayerNorm(hidden_size)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch_size, channels, height, width]
        Returns:
            hidden_states: [batch_size, num_patches+1, hidden_size]
        """
        batch_size = pixel_values.shape[0]
        
        # Patch Embedding
        patch_embeds = self.patch_embedding(pixel_values)  # [B, D, H', W']
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # Add Class Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patch_embeds = torch.cat([cls_tokens, patch_embeds], dim=1)  # [B, N+1, D]
        
        # Add Position Embedding
        patch_embeds = patch_embeds + self.position_embedding
        
        # Transformer Layers
        hidden_states = patch_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # LayerNorm
        hidden_states = self.layernorm(hidden_states)
        
        return hidden_states


class ViTLayer(nn.Module):
    """ViT 单层 Transformer"""
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.layernorm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.layernorm1(hidden_states)
        hidden_states, _ = self.attention(hidden_states, hidden_states, hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.layernorm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


# ============================================================
# 2. 投影层实现
# ============================================================

class SimpleProjector(nn.Module):
    """
    简单 MLP 投影层 (LLaVA 风格)
    将视觉特征投影到语言模型空间
    """
    def __init__(self, vision_dim: int = 1024, llm_dim: int = 4096, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or llm_dim
        
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim)
        )
    
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: [batch_size, num_patches, vision_dim]
        Returns:
            projected_features: [batch_size, num_patches, llm_dim]
        """
        return self.projector(image_features)


class QFormerProjector(nn.Module):
    """
    Q-Former 风格的投影层 (BLIP-2 风格)
    使用可学习的查询向量从视觉特征中提取信息
    """
    def __init__(
        self,
        vision_dim: int = 768,
        llm_dim: int = 4096,
        num_query: int = 32,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
    ):
        super().__init__()
        self.num_query = num_query
        self.llm_dim = llm_dim
        
        # 可学习的查询向量
        self.query_tokens = nn.Parameter(torch.randn(1, num_query, vision_dim))
        
        # Transformer 层 (处理查询)
        self.layers = nn.ModuleList([
            QFormerLayer(vision_dim, num_attention_heads)
            for _ in range(num_hidden_layers)
        ])
        
        # 投影到 LLM 维度
        self.projection = nn.Linear(vision_dim, llm_dim)
    
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: [batch_size, num_patches, vision_dim]
        Returns:
            query_outputs: [batch_size, num_query, llm_dim]
        """
        batch_size = image_features.shape[0]
        
        # 扩展查询向量
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # 通过 Transformer 层
        hidden_states = query_tokens
        for layer in self.layers:
            hidden_states = layer(hidden_states, image_features)
        
        # 投影到 LLM 维度
        query_outputs = self.projection(hidden_states)
        
        return query_outputs


class QFormerLayer(nn.Module):
    """Q-Former 层 (带交叉注意力)"""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        # 自注意力 (查询之间)
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        
        # 交叉注意力 (查询 - 图像特征)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.layernorm3 = nn.LayerNorm(hidden_size)
    
    def forward(self, query: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        # 自注意力
        residual = query
        query = self.layernorm1(query)
        query, _ = self.self_attention(query, query, query)
        query = residual + query
        
        # 交叉注意力
        residual = query
        query = self.layernorm2(query)
        query, _ = self.cross_attention(query, image_features, image_features)
        query = residual + query
        
        # MLP
        residual = query
        query = self.layernorm3(query)
        query = self.mlp(query)
        query = residual + query
        
        return query


# ============================================================
# 3. 多模态 LLM 实现
# ============================================================

@dataclass
class MultimodalConfig:
    """多模态模型配置"""
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    max_position_embeddings: int = 2048
    vision_dim: int = 1024
    num_visual_tokens: int = 576  # CLIP ViT-L/14: (224/14)^2 = 256, 但通常保留更多


class LlamaDecoderLayer(nn.Module):
    """简化的 LLaMA Decoder 层"""
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Self Attention
        self.self_attn = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, 
            batch_first=True
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size)
        
        # MLP (SwiGLU)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.swiglu(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def swiglu(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return F.silu(gate) * up


class RMSNorm(nn.Module):
    """RMS Normalization (LLaMA 使用)"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MultimodalLLM(nn.Module):
    """
    多模态大语言模型
    整合视觉编码器和语言模型
    """
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # LLM 层
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_size)
        
        # 语言模型头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 视觉投影层
        self.vision_projector = SimpleProjector(
            vision_dim=config.vision_dim,
            llm_dim=config.hidden_size
        )
        
        # 特殊 token 嵌入 (<image>)
        self.image_token_id = config.vocab_size  # 假设新增的特殊 token
        self.image_token_embed = nn.Parameter(torch.randn(config.hidden_size))
    
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len] - 输入 token IDs
            pixel_values: [batch_size, channels, height, width] - 图像
            attention_mask: [batch_size, seq_len] - 注意力掩码
            labels: [batch_size, seq_len] - 训练标签
        Returns:
            dict with logits, loss (if labels provided)
        """
        # 获取词嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 如果有图像，处理视觉特征
        if pixel_values is not None:
            # 编码图像
            # 这里假设已经通过外部 ViT 获取了图像特征
            # 实际使用时需要传入 image_features 而不是 pixel_values
            batch_size = pixel_values.shape[0]
            
            # 简化处理：假设 pixel_values 已经是 ViT 输出的特征
            if len(pixel_values.shape) == 3:
                # [B, N, D]
                image_features = pixel_values
            else:
                # 需要外部 ViT 处理
                raise ValueError("请传入已编码的图像特征 [B, N, D]")
            
            # 投影到 LLM 空间
            visual_tokens = self.vision_projector(image_features)  # [B, N, D]
            
            # 将视觉 token 插入到隐藏状态中
            # 简化实现：假设<image>token 在位置 0
            hidden_states = self._merge_visual_tokens(hidden_states, visual_tokens)
        
        # 通过 LLM 层
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # 归一化
        hidden_states = self.norm(hidden_states)
        
        # 计算 logits
        logits = self.lm_head(hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {"logits": logits, "loss": loss}
    
    def _merge_visual_tokens(self, text_hidden: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        将视觉 token 合并到文本隐藏状态中
        简化实现：在开头插入视觉 token
        """
        # 实际实现需要找到<image>token 的位置并替换
        # 这里简化为在开头插入
        batch_size = text_hidden.shape[0]
        num_visual = visual_tokens.shape[1]
        
        # 取第一个视觉 token 作为代表 (简化)
        visual_rep = visual_tokens[:, :1, :].expand(-1, 1, -1)
        
        # 在开头插入视觉表示
        merged = torch.cat([visual_rep, text_hidden[:, 1:, :]], dim=1)
        
        return merged
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """自回归生成"""
        self.eval()
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # 前向传播
            outputs = self.forward(generated, pixel_values)
            logits = outputs["logits"][:, -1, :]
            
            # 采样
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                
                # Top-p 采样
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > top_p
                sorted_indices[mask] = -1
                probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
                probs = probs / probs.sum(-1, keepdim=True)
                
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # 检查 EOS
            if next_token.item() == 2:  # 假设 2 是 EOS
                break
            
            generated = torch.cat([generated, next_token], dim=-1)
        
        return generated


# ============================================================
# 4. 多模态对话数据集
# ============================================================

class MultimodalChatDataset(Dataset):
    """
    多模态对话数据集
    支持图像 - 文本对和对话历史
    """
    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_processor=None,
        max_length: int = 512,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        
        # 加载数据 (简化示例)
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """加载数据 (示例)"""
        # 实际实现需要从文件加载
        return [
            {
                "image": "path/to/image.jpg",
                "conversations": [
                    {"from": "human", "value": "这张图片里有什么？"},
                    {"from": "gpt", "value": "图片中显示了一只可爱的狗。"}
                ]
            }
        ]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # 处理图像
        image = self._load_image(item["image"])
        if self.image_processor:
            pixel_values = self.image_processor(image)
        else:
            pixel_values = image
        
        # 处理对话
        conversations = item["conversations"]
        text = self._format_conversations(conversations)
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 创建 labels (用于训练)
        labels = encoded["input_ids"].clone()
        # 将 padding 和 human 部分设为 -100 (不计算损失)
        # 实际实现需要更精细的处理
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "labels": labels.squeeze(0)
        }
    
    def _load_image(self, image_path: str):
        """加载图像 (简化)"""
        # 实际实现使用 PIL 或 torchvision
        return torch.randn(3, 224, 224)  # 占位符
    
    def _format_conversations(self, conversations: List[Dict]) -> str:
        """格式化对话"""
        formatted = ""
        for conv in conversations:
            if conv["from"] == "human":
                formatted += f"USER: {conv['value']}\n"
            else:
                formatted += f"ASSISTANT: {conv['value']}\n"
        return formatted


# ============================================================
# 5. 完整训练流程
# ============================================================

class MultimodalTrainer:
    """
    多模态模型训练器
    """
    def __init__(
        self,
        model: MultimodalLLM,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        num_epochs: int = 3,
        device: str = "cuda",
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs * len(train_dataset) // batch_size
        )
    
    def train(self) -> None:
        """执行训练"""
        self.model.to(self.device)
        self.model.train()
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                # 移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                loss = outputs["loss"]
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # 日志
                if num_batches % 10 == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {avg_loss:.4f}, LR: {lr:.6f}")
            
            # 评估
            if self.eval_dataset:
                eval_loss = self.evaluate()
                print(f"Epoch {epoch+1}, Eval Loss: {eval_loss:.4f}")
            
            # 保存检查点
            self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """评估"""
        self.model.eval()
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size
        )
        
        total_loss = 0
        num_batches = 0
        
        for batch in eval_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = self.model(
                input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            total_loss += outputs["loss"].item()
            num_batches += 1
        
        self.model.train()
        return total_loss / num_batches
    
    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        print(f"检查点已保存到 {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"已从 {path} 加载检查点")


# ============================================================
# 6. 使用示例
# ============================================================

def example_usage():
    """
    多模态模型使用示例
    """
    print("=" * 60)
    print("多模态大模型 (LMM) 使用示例")
    print("=" * 60)
    
    # 1. 创建配置
    config = MultimodalConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=2,  # 示例用小型模型
        num_attention_heads=8,
        vision_dim=1024,
    )
    
    # 2. 创建模型
    model = MultimodalLLM(config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 创建投影层
    projector = SimpleProjector(vision_dim=1024, llm_dim=4096)
    
    # 4. 示例输入
    batch_size = 2
    seq_len = 128
    num_patches = 256
    
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    image_features = torch.randn(batch_size, num_patches, 1024)
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 32000, (batch_size, seq_len))
    
    # 5. 前向传播
    outputs = model(
        input_ids=input_ids,
        pixel_values=image_features,
        attention_mask=attention_mask,
        labels=labels
    )
    
    print(f"Logits 形状：{outputs['logits'].shape}")
    print(f"Loss: {outputs['loss']:.4f}")
    
    # 6. 生成示例
    generated = model.generate(
        input_ids=input_ids[:, :10],
        pixel_values=image_features,
        max_new_tokens=20
    )
    print(f"生成序列长度：{generated.shape[1]}")
    
    print("\n示例完成!")


if __name__ == "__main__":
    example_usage()
