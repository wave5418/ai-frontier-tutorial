"""
第 16 章 VLA 基础 - 代码实现
Vision-Language-Action 模型示例

本代码演示 VLA 模型的核心组件和训练流程
适合学习和理解 VLA 架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np


# ============================================================
# 1. 视觉编码器 (Vision Encoder)
# ============================================================

class VisionEncoder(nn.Module):
    """
    简化的视觉编码器
    使用 CNN + Transformer 提取图像特征
    
    输入：RGB 图像 (B, 3, H, W)
    输出：视觉特征序列 (B, N, D)
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 计算 patch 数量
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch 嵌入层：将图像分割为 patch 并投影到嵌入空间
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, 3, H, W)
        
        Returns:
            视觉特征 (B, N+1, D)
        """
        B = x.shape[0]
        
        # Patch 嵌入
        x = self.patch_embed(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # 添加 cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer 编码
        x = self.transformer(x)
        x = self.norm(x)
        
        return x


# ============================================================
# 2. 语言编码器 (Language Encoder)
# ============================================================

class LanguageEncoder(nn.Module):
    """
    简化的语言编码器
    使用嵌入层 + Transformer 处理文本
    
    输入：文本 token IDs (B, L)
    输出：语言特征 (B, L, D)
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 768,
        max_seq_len: int = 64,
        num_heads: int = 12,
        num_layers: int = 6
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token 嵌入
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 文本 token IDs (B, L)
            attention_mask: 注意力掩码 (B, L)
        
        Returns:
            语言特征 (B, L, D)
        """
        # Token 嵌入
        x = self.token_embed(x)  # (B, L, D)
        
        # 添加位置编码
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Transformer 编码
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        x = self.norm(x)
        
        return x


# ============================================================
# 3. 多模态融合模块 (Multimodal Fusion)
# ============================================================

class MultimodalFusion(nn.Module):
    """
    多模态融合模块
    使用交叉注意力融合视觉和语言特征
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 交叉注意力：语言查询视觉
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        vis_features: torch.Tensor,
        lang_features: torch.Tensor,
        lang_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        融合视觉和语言特征
        
        Args:
            vis_features: 视觉特征 (B, N, D)
            lang_features: 语言特征 (B, L, D)
            lang_mask: 语言掩码 (B, L)
        
        Returns:
            融合特征 (B, L, D)
        """
        # 交叉注意力：语言作为 query，视觉作为 key/value
        attn_output, _ = self.cross_attn(
            query=lang_features,
            key=vis_features,
            value=vis_features,
            key_padding_mask=lang_mask
        )
        
        # 残差连接 + 层归一化
        x = self.norm1(lang_features + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


# ============================================================
# 4. 动作预测头 (Action Head)
# ============================================================

class ActionHead(nn.Module):
    """
    动作预测头
    将融合特征映射到动作空间
    
    支持两种模式：
    1. 离散动作：分类
    2. 连续动作：回归
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        action_dim: int = 7,  # 动作维度（如 7 自由度机械臂）
        action_type: str = "continuous"  # "discrete" 或 "continuous"
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_type = action_type
        
        if action_type == "discrete":
            # 离散动作：每个动作维度是一个分类问题
            self.action_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, 256),
                    nn.GELU(),
                    nn.Linear(256, action_dim)  # 假设每个维度有 action_dim 个离散值
                ) for _ in range(7)  # 7 个动作维度
            ])
        else:
            # 连续动作：回归
            self.action_mlp = nn.Sequential(
                nn.Linear(embed_dim, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, action_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测动作
        
        Args:
            x: 融合特征 (B, L, D)
        
        Returns:
            动作预测
        """
        # 使用序列的最后一个 token（或 pooled 表示）
        x = x[:, -1, :]  # (B, D)
        
        if self.action_type == "discrete":
            # 离散动作：每个维度单独预测
            actions = [head(x) for head in self.action_heads]
            return torch.stack(actions, dim=1)  # (B, 7, action_dim)
        else:
            # 连续动作
            return self.action_mlp(x)  # (B, action_dim)


# ============================================================
# 5. VLA 模型 (完整架构)
# ============================================================

class VLAModel(nn.Module):
    """
    完整的 VLA 模型
    
    架构：
    视觉输入 → 视觉编码器 → \
                              → 多模态融合 → 动作预测头 → 动作输出
    语言输入 → 语言编码器 → /
    """
    
    def __init__(
        self,
        image_size: int = 224,
        vocab_size: int = 50257,
        embed_dim: int = 768,
        action_dim: int = 7,
        action_type: str = "continuous"
    ):
        super().__init__()
        
        # 视觉编码器
        self.vision_encoder = VisionEncoder(
            image_size=image_size,
            embed_dim=embed_dim
        )
        
        # 语言编码器
        self.language_encoder = LanguageEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim
        )
        
        # 多模态融合
        self.fusion = MultimodalFusion(embed_dim=embed_dim)
        
        # 动作预测头
        self.action_head = ActionHead(
            embed_dim=embed_dim,
            action_dim=action_dim,
            action_type=action_type
        )
    
    def forward(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: 输入图像 (B, 3, H, W)
            text_ids: 文本 token IDs (B, L)
            text_mask: 文本掩码 (B, L)
        
        Returns:
            动作预测
        """
        # 编码视觉
        vis_features = self.vision_encoder(images)  # (B, N+1, D)
        
        # 编码语言
        lang_features = self.language_encoder(text_ids, text_mask)  # (B, L, D)
        
        # 多模态融合
        fused_features = self.fusion(vis_features, lang_features, text_mask)  # (B, L, D)
        
        # 预测动作
        actions = self.action_head(fused_features)
        
        return actions


# ============================================================
# 6. 数据集和训练
# ============================================================

class VLADataset(Dataset):
    """
    VLA 训练数据集
    
    每条数据包含：
    - 图像
    - 文本指令
    - 专家动作
    """
    
    def __init__(
        self,
        data_path: str,
        image_size: int = 224,
        max_text_len: int = 64
    ):
        super().__init__()
        
        self.image_size = image_size
        self.max_text_len = max_text_len
        
        # 实际使用时从文件加载数据
        # 这里使用模拟数据
        self.size = 1000
        self.vocab_size = 50257
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 模拟数据（实际使用时替换为真实数据加载）
        
        # 随机图像
        image = torch.randn(3, self.image_size, self.image_size)
        
        # 随机文本 token IDs
        text_len = np.random.randint(10, self.max_text_len)
        text_ids = torch.randint(0, self.vocab_size, (text_len,))
        text_mask = torch.zeros(self.max_text_len, dtype=torch.bool)
        text_mask[:text_len] = True
        
        # 随机动作（7 自由度机械臂）
        action = torch.randn(7)
        
        return {
            "image": image,
            "text_ids": text_ids,
            "text_mask": text_mask,
            "action": action
        }


class VLATrainer:
    """
    VLA 模型训练器
    """
    
    def __init__(
        self,
        model: VLAModel,
        learning_rate: float = 1e-4,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100
        )
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> float:
        """
        训练一个 epoch
        
        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # 准备数据
            images = batch["image"].to(self.device)
            text_ids = batch["text_ids"].to(self.device)
            text_mask = batch["text_mask"].to(self.device)
            actions = batch["action"].to(self.device)
            
            # 前向传播
            pred_actions = self.model(images, text_ids, text_mask)
            
            # 计算损失（MSE）
            loss = F.mse_loss(pred_actions, actions)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # 更新学习率
        self.scheduler.step()
        
        return total_loss / num_batches
    
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> float:
        """
        评估模型
        
        Returns:
            平均损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                text_ids = batch["text_ids"].to(self.device)
                text_mask = batch["text_mask"].to(self.device)
                actions = batch["action"].to(self.device)
                
                pred_actions = self.model(images, text_ids, text_mask)
                loss = F.mse_loss(pred_actions, actions)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches


# ============================================================
# 7. 仿真环境示例
# ============================================================

class SimpleRobotEnv:
    """
    简单的机器人仿真环境
    用于测试 VLA 模型
    
    环境状态：
    - 相机图像
    - 任务指令（文本）
    
    动作：
    - 7 自由度机械臂关节角度
    """
    
    def __init__(
        self,
        image_size: int = 224,
        max_steps: int = 100
    ):
        self.image_size = image_size
        self.max_steps = max_steps
        self.step_count = 0
        
        # 机器人状态
        self.robot_joint_angles = np.zeros(7)
        self.target_position = np.random.randn(3)
    
    def reset(self) -> Dict[str, any]:
        """
        重置环境
        
        Returns:
            初始观测
        """
        self.step_count = 0
        self.robot_joint_angles = np.zeros(7)
        self.target_position = np.random.randn(3)
        
        # 生成初始图像（模拟）
        image = np.random.randn(3, self.image_size, self.image_size)
        
        # 任务指令
        instruction = "将机械臂移动到目标位置"
        
        return {
            "image": image,
            "instruction": instruction,
            "robot_state": self.robot_joint_angles.copy()
        }
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        执行动作
        
        Args:
            action: 动作 (7,)
        
        Returns:
            observation, reward, done, info
        """
        self.step_count += 1
        
        # 更新机器人状态
        self.robot_joint_angles = action
        
        # 计算奖励（基于与目标的距离）
        # 简化：假设关节角度直接映射到末端位置
        current_position = np.tanh(self.robot_joint_angles[:3])  # 简化映射
        distance = np.linalg.norm(current_position - self.target_position)
        reward = -distance  # 距离越小奖励越高
        
        # 检查是否完成
        done = (distance < 0.1) or (self.step_count >= self.max_steps)
        
        # 生成新图像（模拟）
        image = np.random.randn(3, self.image_size, self.image_size)
        instruction = "将机械臂移动到目标位置"
        
        obs = {
            "image": image,
            "instruction": instruction,
            "robot_state": self.robot_joint_angles.copy()
        }
        
        info = {"distance": distance, "steps": self.step_count}
        
        return obs, reward, done, info


# ============================================================
# 8. 使用示例
# ============================================================

def main():
    """
    主函数：演示 VLA 模型的创建和训练
    """
    print("=" * 60)
    print("VLA 模型示例 - 第 16 章")
    print("=" * 60)
    
    # 创建模型
    print("\n1. 创建 VLA 模型...")
    model = VLAModel(
        image_size=224,
        vocab_size=50257,
        embed_dim=768,
        action_dim=7,
        action_type="continuous"
    )
    
    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型总参数量：{total_params:,}")
    
    # 创建训练器
    print("\n2. 创建训练器...")
    trainer = VLATrainer(model, learning_rate=1e-4)
    
    # 创建数据集
    print("\n3. 创建数据集...")
    train_dataset = VLADataset("data/train", image_size=224)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 训练（演示用，只训练 1 个 epoch）
    print("\n4. 开始训练（演示：1 个 epoch）...")
    train_loss = trainer.train_epoch(train_loader, epoch=1)
    print(f"   训练损失：{train_loss:.4f}")
    
    # 创建仿真环境
    print("\n5. 创建仿真环境...")
    env = SimpleRobotEnv(image_size=224)
    
    # 环境交互示例
    print("\n6. 环境交互示例...")
    obs = env.reset()
    print(f"   初始图像形状：{obs['image'].shape}")
    print(f"   任务指令：{obs['instruction']}")
    
    # 模拟几步交互
    for step in range(3):
        # 随机动作
        action = np.random.randn(7)
        obs, reward, done, info = env.step(action)
        print(f"   步骤 {step+1}: 奖励={reward:.3f}, 距离={info['distance']:.3f}")
        
        if done:
            break
    
    print("\n" + "=" * 60)
    print("VLA 模型示例完成！")
    print("=" * 60)
    
    print("\n关键要点：")
    print("1. VLA 模型融合视觉、语言和动作")
    print("2. 使用 Transformer 架构处理多模态输入")
    print("3. 通过行为克隆从专家数据中学习")
    print("4. 可以端到端训练从感知到动作的映射")
    
    return model


if __name__ == "__main__":
    model = main()
