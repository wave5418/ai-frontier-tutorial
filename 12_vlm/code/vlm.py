# -*- coding: utf-8 -*-
"""
第 12 章 视觉语言模型 (VLM) 代码实现
=====================================
包含：完整 VLM 实现、VQA 示例、图像描述、视觉对话

作者：AI 前沿技术教程
日期：2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import math


# ============================================================
# 1. 完整 VLM 实现
# ============================================================

@dataclass
class VLMConfig:
    """VLM 配置类"""
    # 语言模型配置
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    max_position_embeddings: int = 2048
    
    # 视觉模型配置
    vision_hidden_size: int = 1024  # CLIP ViT-L 输出维度
    vision_num_patches: int = 256   # (224/14)^2
    image_size: int = 224
    
    # 投影配置
    projector_type: str = "mlp"  # mlp, qformer, linear
    num_query_tokens: int = 32
    
    # 特殊 token
    image_token_id: int = 32000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


class VisionEncoder(nn.Module):
    """
    视觉编码器
    可以使用预训练的 CLIP ViT 或自定义 ViT
    """
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config
        
        # 简化 ViT 实现 (实际使用预训练 CLIP)
        self.patch_embed = nn.Conv2d(
            3, config.vision_hidden_size,
            kernel_size=14, stride=14  # CLIP ViT-L/14
        )
        
        num_patches = (config.image_size // 14) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.vision_hidden_size))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, config.vision_hidden_size))
        
        # Transformer 层
        self.layers = nn.ModuleList([
            ViTBlock(config.vision_hidden_size, config.vision_hidden_size // 64)
            for _ in range(24)  # ViT-L 有 24 层
        ])
        
        self.norm = nn.LayerNorm(config.vision_hidden_size)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W]
        Returns:
            features: [B, N, D] (不包括 CLS token)
        """
        B = pixel_values.shape[0]
        
        # Patch Embedding
        x = self.patch_embed(pixel_values)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)    # [B, N, D]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add Position Embedding
        x = x + self.pos_embed
        
        # Transformer
        for layer in self.layers:
            x = layer(x)
        
        # Norm
        x = self.norm(x)
        
        # 返回除 CLS 外的所有 patch 特征
        return x[:, 1:, :]


class ViTBlock(nn.Module):
    """ViT Transformer Block"""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self Attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class Projector(nn.Module):
    """
    视觉 - 语言投影层
    支持多种投影策略
    """
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config
        self.projector_type = config.projector_type
        
        if config.projector_type == "linear":
            # 简单线性投影
            self.project = nn.Linear(config.vision_hidden_size, config.hidden_size)
        
        elif config.projector_type == "mlp":
            # 2 层 MLP (LLaVA 风格)
            self.project = nn.Sequential(
                nn.Linear(config.vision_hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        
        elif config.projector_type == "qformer":
            # Q-Former 风格
            self.project = QFormerModule(
                vision_dim=config.vision_hidden_size,
                llm_dim=config.hidden_size,
                num_query=config.num_query_tokens
            )
        
        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [B, N, vision_dim]
        Returns:
            projected: [B, N', hidden_dim]
        """
        return self.project(vision_features)


class QFormerModule(nn.Module):
    """Q-Former 投影模块"""
    def __init__(self, vision_dim: int, llm_dim: int, num_query: int = 32):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_query, vision_dim))
        
        self.layers = nn.ModuleList([
            QFormerBlock(vision_dim, vision_dim // 64)
            for _ in range(6)
        ])
        
        self.projection = nn.Linear(vision_dim, llm_dim)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        B = vision_features.shape[0]
        
        # Expand query tokens
        query = self.query_tokens.expand(B, -1, -1)
        
        # Cross-attention with vision features
        for layer in self.layers:
            query = layer(query, vision_features)
        
        # Project to LLM dimension
        return self.projection(query)


class QFormerBlock(nn.Module):
    """Q-Former Block (带交叉注意力)"""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        # Self attention (among queries)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        
        # Cross attention (query to vision)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm3 = nn.LayerNorm(hidden_size)
    
    def forward(self, query: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        # Self Attention
        residual = query
        query = self.norm1(query)
        query, _ = self.self_attn(query, query, query)
        query = residual + query
        
        # Cross Attention
        residual = query
        query = self.norm2(query)
        query, _ = self.cross_attn(query, vision_features, vision_features)
        query = residual + query
        
        # MLP
        residual = query
        query = self.norm3(query)
        query = self.mlp(query)
        query = residual + query
        
        return query


class LLMDecoder(nn.Module):
    """
    简化的 LLM 解码器
    基于 LLaMA 架构
    """
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config
        
        # Token Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Decoder Layers
        self.layers = nn.ModuleList([
            LlamaDecoderBlock(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Final Norm
        self.norm = RMSNorm(config.hidden_size)
        
        # Output Head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, seq_len]
            inputs_embeds: 可选，已嵌入的输入 (用于插入视觉 token)
            attention_mask: [B, seq_len]
        Returns:
            logits: [B, seq_len, vocab_size]
        """
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        
        # Create causal mask
        seq_len = hidden_states.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device),
            diagonal=1
        ).bool()
        causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))
        
        # Apply decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Output logits
        logits = self.lm_head(hidden_states)
        
        return logits


class LlamaDecoderBlock(nn.Module):
    """LLaMA Decoder Block"""
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Self Attention
        self.self_attn = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads,
            batch_first=True
        )
        self.input_layernorm = RMSNorm(config.hidden_size)
        
        # MLP (SwiGLU)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Self Attention
        residual = x
        x = self.input_layernorm(x)
        x, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = residual + x
        
        # MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.swiglu(x)
        x = self.down_proj(x)
        x = residual + x
        
        return x
    
    def swiglu(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return F.silu(gate) * up


class RMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class VisionLanguageModel(nn.Module):
    """
    完整的视觉语言模型
    整合视觉编码器、投影层和语言解码器
    """
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config
        
        # Components
        self.vision_encoder = VisionEncoder(config)
        self.projector = Projector(config)
        self.llm = LLMDecoder(config)
        
        # Image token embedding
        self.image_token_embed = nn.Parameter(
            torch.randn(config.hidden_size)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [B, seq_len] - 包含<image>标记的输入
            pixel_values: [B, 3, H, W] - 图像
            attention_mask: [B, seq_len] - 注意力掩码
            labels: [B, seq_len] - 训练标签
        Returns:
            dict with logits, loss
        """
        # Get text embeddings
        inputs_embeds = self.llm.embed_tokens(input_ids)
        
        # Process visual features if images provided
        if pixel_values is not None:
            # Encode images
            vision_features = self.vision_encoder(pixel_values)  # [B, N, vision_dim]
            
            # Project to LLM space
            visual_tokens = self.projector(vision_features)  # [B, N', hidden_dim]
            
            # Merge visual tokens with text embeddings
            inputs_embeds = self._merge_visual_features(
                inputs_embeds, visual_tokens, input_ids
            )
        
        # Run LLM
        logits = self.llm(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        
        # Compute loss if labels provided
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
    
    def _merge_visual_features(
        self,
        inputs_embeds: torch.Tensor,
        visual_tokens: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        将视觉特征合并到文本嵌入中
        找到<image>token 位置并替换为视觉 token
        """
        B, seq_len, hidden_dim = inputs_embeds.shape
        num_visual = visual_tokens.shape[1]
        
        # Find image token positions
        image_mask = (input_ids == self.config.image_token_id)  # [B, seq_len]
        
        # Create new embeddings
        new_embeds = inputs_embeds.clone()
        
        # For each sample, replace image token with visual tokens
        for b in range(B):
            image_positions = torch.where(image_mask[b])[0]
            if len(image_positions) > 0:
                pos = image_positions[0]
                # 简化：在 image token 位置插入视觉 token
                # 实际实现需要调整序列长度
                if pos + num_visual <= seq_len:
                    new_embeds[b, pos:pos+num_visual] = visual_tokens[b, :num_visual]
        
        return new_embeds
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        自回归生成
        """
        self.eval()
        
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                input_ids=generated,
                pixel_values=pixel_values if generated.shape[1] == input_ids.shape[1] else None
            )
            logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Top-p sampling
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > top_p
                sorted_indices[mask] = -1
                
                filtered_probs = torch.zeros_like(probs).scatter(
                    -1, sorted_indices, sorted_probs
                )
                filtered_probs = filtered_probs / filtered_probs.sum(-1, keepdim=True)
                
                next_token = torch.multinomial(filtered_probs, 1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Check EOS
            if next_token.item() == self.config.eos_token_id:
                break
            
            generated = torch.cat([generated, next_token], dim=-1)
        
        return generated


# ============================================================
# 2. VQA 示例
# ============================================================

class VQADataset(Dataset):
    """
    视觉问答数据集
    """
    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_processor=None,
        max_length: int = 128,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        
        # 加载数据
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """加载 VQA 数据 (示例)"""
        return [
            {
                "image": "image_001.jpg",
                "question": "图片里有什么动物？",
                "answer": "猫"
            },
            {
                "image": "image_002.jpg",
                "question": "这是什么颜色？",
                "answer": "红色"
            }
        ]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Load image
        pixel_values = self._load_image(item["image"])
        
        # Format input
        text = f"USER: <image>\n{item['question']}\nASSISTANT:"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels
        labels = encoded["input_ids"].clone()
        # Mask the question part (only compute loss on answer)
        # 简化实现
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "labels": labels.squeeze(0),
            "question": item["question"],
            "answer": item["answer"]
        }
    
    def _load_image(self, path: str) -> torch.Tensor:
        """加载图像 (占位符)"""
        return torch.randn(3, 224, 224)


class VQAEvaluator:
    """
    VQA 评估器
    """
    def __init__(self, model: VisionLanguageModel, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估 VQA 性能"""
        self.model.eval()
        
        correct = 0
        total = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            pixel_values = batch["pixel_values"].to(self.device)
            answers = batch["answer"]
            
            # Generate answer
            generated = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=20
            )
            
            # Decode and compare
            for gen, ans in zip(generated, answers):
                gen_text = self.tokenizer.decode(gen, skip_special_tokens=True)
                # Simple exact match (实际应使用 VQA 评估脚本)
                if ans.strip().lower() in gen_text.lower():
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {"accuracy": accuracy, "correct": correct, "total": total}


# ============================================================
# 3. 图像描述生成
# ============================================================

class ImageCaptioningDataset(Dataset):
    """
    图像描述数据集
    """
    def __init__(self, data_path: str, tokenizer, image_processor=None, max_length: int = 64):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """加载图像描述数据"""
        return [
            {
                "image": "image_001.jpg",
                "caption": "一只黑色的猫坐在窗台上晒太阳"
            },
            {
                "image": "image_002.jpg",
                "caption": "一群人在公园里野餐"
            }
        ]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        pixel_values = self._load_image(item["image"])
        
        # Format for captioning
        text = f"USER: <image>\n请描述这张图片。\nASSISTANT: {item['caption']}"
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "caption": item["caption"]
        }
    
    def _load_image(self, path: str) -> torch.Tensor:
        return torch.randn(3, 224, 224)


class CaptionGenerator:
    """
    图像描述生成器
    """
    def __init__(self, model: VisionLanguageModel, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
    
    @torch.no_grad()
    def generate_caption(
        self,
        pixel_values: torch.Tensor,
        prompt: str = "请描述这张图片。",
        max_length: int = 100,
    ) -> str:
        """生成图像描述"""
        self.model.eval()
        
        # Prepare input
        text = f"USER: <image>\n{prompt}\nASSISTANT:"
        encoded = self.tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        
        pixel_values = pixel_values.to(self.device)
        
        # Generate
        generated = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9
        )
        
        # Decode
        output = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "ASSISTANT:" in output:
            output = output.split("ASSISTANT:")[-1].strip()
        
        return output


# ============================================================
# 4. 视觉对话示例
# ============================================================

class VisualDialogDataset(Dataset):
    """
    视觉对话数据集
    支持多轮对话
    """
    def __init__(self, data_path: str, tokenizer, image_processor=None, max_length: int = 256):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """加载视觉对话数据"""
        return [
            {
                "image": "image_001.jpg",
                "dialog": [
                    {"from": "human", "value": "图片里有什么？"},
                    {"from": "gpt", "value": "图片里有一只猫。"},
                    {"from": "human", "value": "它是什么颜色的？"},
                    {"from": "gpt", "value": "它是黑色的。"},
                    {"from": "human", "value": "它在做什么？"},
                    {"from": "gpt", "value": "它正坐在窗台上。"}
                ]
            }
        ]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        pixel_values = self._load_image(item["image"])
        
        # Format dialog
        text = "USER: <image>\n"
        for turn in item["dialog"]:
            if turn["from"] == "human":
                text += f"USER: {turn['value']}\n"
            else:
                text += f"ASSISTANT: {turn['value']}\n"
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "dialog": item["dialog"]
        }
    
    def _load_image(self, path: str) -> torch.Tensor:
        return torch.randn(3, 224, 224)


class VisualDialogSystem:
    """
    视觉对话系统
    支持多轮对话
    """
    def __init__(self, model: VisionLanguageModel, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
        self.dialog_history = []
        self.current_image = None
    
    def reset(self):
        """重置对话"""
        self.dialog_history = []
        self.current_image = None
    
    def set_image(self, pixel_values: torch.Tensor):
        """设置当前图像"""
        self.current_image = pixel_values.to(self.device)
        self.reset()
    
    @torch.no_grad()
    def respond(self, user_input: str, max_length: int = 100) -> str:
        """
        对应用户输入
        """
        self.model.eval()
        
        # Build context
        context = "USER: <image>\n" if self.current_image is not None else ""
        for turn in self.dialog_history:
            context += f"USER: {turn['user']}\n"
            context += f"ASSISTANT: {turn['assistant']}\n"
        
        context += f"USER: {user_input}\nASSISTANT:"
        
        # Tokenize
        encoded = self.tokenizer(context, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        
        # Generate
        generated = self.model.generate(
            input_ids=input_ids,
            pixel_values=self.current_image,
            max_new_tokens=max_length,
            temperature=0.7
        )
        
        # Decode
        output = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Extract response
        if "ASSISTANT:" in output:
            response = output.split("ASSISTANT:")[-1].strip()
        else:
            response = output
        
        # Update history
        self.dialog_history.append({
            "user": user_input,
            "assistant": response
        })
        
        return response


# ============================================================
# 5. 使用示例
# ============================================================

def example_vlm_usage():
    """
    VLM 使用示例
    """
    print("=" * 60)
    print("视觉语言模型 (VLM) 使用示例")
    print("=" * 60)
    
    # 1. 创建配置
    config = VLMConfig(
        vocab_size=32000,
        hidden_size=1024,  # 小型配置用于示例
        num_hidden_layers=4,
        num_attention_heads=8,
        vision_hidden_size=512,
        projector_type="mlp"
    )
    
    # 2. 创建模型
    model = VisionLanguageModel(config)
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 示例输入
    batch_size = 2
    seq_len = 64
    
    input_ids = torch.randint(100, 31000, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(100, 31000, (batch_size, seq_len))
    
    # 4. 前向传播
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        labels=labels
    )
    
    print(f"Logits 形状：{outputs['logits'].shape}")
    print(f"Loss: {outputs['loss']:.4f}")
    
    # 5. 生成示例
    generated = model.generate(
        input_ids=input_ids[:, :10],
        pixel_values=pixel_values,
        max_new_tokens=20
    )
    print(f"生成序列长度：{generated.shape[1]}")
    
    print("\nVLM 示例完成!")


def example_vqa_pipeline():
    """
    VQA 完整流程示例
    """
    print("\n" + "=" * 60)
    print("VQA 流程示例")
    print("=" * 60)
    
    # 创建小型模型
    config = VLMConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        vision_hidden_size=128
    )
    
    model = VisionLanguageModel(config)
    
    # 模拟 VQA 输入
    input_ids = torch.randint(10, 990, (1, 32))
    pixel_values = torch.randn(1, 3, 224, 224)
    
    # 生成答案
    answer_ids = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=10
    )
    
    print(f"问题输入长度：{input_ids.shape[1]}")
    print(f"答案输出长度：{answer_ids.shape[1]}")
    print("VQA 流程完成!")


if __name__ == "__main__":
    example_vlm_usage()
    example_vqa_pipeline()
