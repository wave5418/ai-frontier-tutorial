"""
第 17 章 RT-2 与机器人学习 - 代码实现
Robotic Transformer 2 风格模型示例

本代码演示 RT-2 模型的核心思想和实现
包括动作 token 化、VLA 架构、训练和推理

参考：https://arxiv.org/abs/2307.15818
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


# ============================================================
# 1. 动作 Token 化 (Action Tokenization)
# ============================================================

class ActionTokenizer:
    """
    动作 Token 化器
    
    将连续动作离散化为 token，便于用 Transformer 处理
    
    RT-2 使用 256 个 bin 对每个动作维度进行离散化
    """
    
    def __init__(
        self,
        action_dim: int = 7,
        num_bins: int = 256,
        action_ranges: Optional[List[Tuple[float, float]]] = None
    ):
        """
        初始化动作 Token 化器
        
        Args:
            action_dim: 动作维度（如 7 自由度机械臂）
            num_bins: 离散化 bin 数量
            action_ranges: 每个维度的范围 [(min, max), ...]
                          如果为 None，使用默认范围 [-1, 1]
        """
        self.action_dim = action_dim
        self.num_bins = num_bins
        
        if action_ranges is None:
            # 默认范围：[-1, 1]
            self.action_ranges = [(-1.0, 1.0)] * action_dim
        else:
            self.action_ranges = action_ranges
    
    def discretize(self, actions: np.ndarray) -> np.ndarray:
        """
        将连续动作离散化为 token IDs
        
        Args:
            actions: 连续动作 (..., action_dim)
        
        Returns:
            离散化的 token IDs (..., action_dim)
        """
        actions = np.asarray(actions)
        original_shape = actions.shape
        
        # 确保最后一维是 action_dim
        actions = actions.reshape(-1, self.action_dim)
        
        tokens = np.zeros_like(actions, dtype=np.int32)
        
        for i in range(self.action_dim):
            min_val, max_val = self.action_ranges[i]
            
            # 截断到有效范围
            actions_i = np.clip(actions[:, i], min_val, max_val)
            
            # 线性映射到 [0, num_bins-1]
            tokens[:, i] = (
                (actions_i - min_val) / (max_val - min_val) * (self.num_bins - 1)
            ).round().astype(np.int32)
        
        return tokens.reshape(*original_shape[:-1], self.action_dim)
    
    def continuous(self, tokens: np.ndarray) -> np.ndarray:
        """
        将 token IDs 还原为连续动作
        
        Args:
            tokens: 离散化的 token IDs (..., action_dim)
        
        Returns:
            连续动作 (..., action_dim)
        """
        tokens = np.asarray(tokens)
        original_shape = tokens.shape
        
        tokens = tokens.reshape(-1, self.action_dim)
        
        actions = np.zeros_like(tokens, dtype=np.float32)
        
        for i in range(self.action_dim):
            min_val, max_val = self.action_ranges[i]
            
            # 映射回连续范围
            actions[:, i] = (
                tokens[:, i].astype(np.float32) / (self.num_bins - 1) * (max_val - min_val)
                + min_val
            )
        
        return actions.reshape(*original_shape[:-1], self.action_dim)
    
    def get_vocab_offset(self) -> int:
        """
        获取动作 token 在词表中的起始偏移
        
        RT-2 将动作 token 附加到文本词表之后
        """
        return 0  # 简化：假设动作 token 从 0 开始
    
    def get_vocab_size(self) -> int:
        """
        获取动作词表大小
        """
        return self.num_bins * self.action_dim


# ============================================================
# 2. 视觉编码器 (Vision Encoder)
# ============================================================

class RT2VisionEncoder(nn.Module):
    """
    RT-2 风格的视觉编码器
    
    基于 ViT（Vision Transformer）
    输出视觉 token 序列
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        pretrained: bool = False
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 计算 patch 数量
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch 嵌入
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 位置编码（可学习）
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True  # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, 3, H, W)
        
        Returns:
            视觉 token 序列 (B, N, D)
        """
        B = x.shape[0]
        
        # Patch 嵌入
        x = self.patch_embed(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer 编码
        x = self.transformer(x)
        x = self.norm(x)
        
        return x


# ============================================================
# 3. 语言编码器 (Language Encoder)
# ============================================================

class RT2LanguageEncoder(nn.Module):
    """
    RT-2 风格的语言编码器
    
    简化的语言模型，实际使用时可替换为 LLaMA、PaLM 等
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 768,
        max_seq_len: int = 128,
        num_heads: int = 12,
        num_layers: int = 12
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
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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
# 4. RT-2 模型架构 (完整 VLA)
# ============================================================

class RT2Model(nn.Module):
    """
    RT-2 风格的 VLA 模型
    
    架构：
    1. 视觉编码器处理图像
    2. 语言编码器处理文本
    3. 融合视觉和语言特征
    4. 预测动作 tokens
    
    关键设计：
    - 动作被离散化为 tokens
    - 使用自回归方式预测动作序列
    """
    
    def __init__(
        self,
        image_size: int = 224,
        vocab_size: int = 32000,
        embed_dim: int = 768,
        action_dim: int = 7,
        num_action_bins: int = 256,
        freeze_vision: bool = False,
        freeze_language: bool = False
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.num_action_bins = num_action_bins
        
        # 动作词表大小（每个维度 num_bins 个 token）
        self.action_vocab_size = num_action_bins
        
        # 视觉编码器
        self.vision_encoder = RT2VisionEncoder(
            image_size=image_size,
            embed_dim=embed_dim
        )
        
        # 语言编码器
        self.language_encoder = RT2LanguageEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim
        )
        
        # 多模态融合（使用交叉注意力）
        self.fusion = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=12,
            batch_first=True
        )
        
        self.fusion_norm = nn.LayerNorm(embed_dim)
        
        # 动作预测头（为每个动作维度预测一个 token）
        self.action_heads = nn.ModuleList([
            nn.Linear(embed_dim, self.action_vocab_size)
            for _ in range(action_dim)
        ])
        
        # 可选：冻结编码器
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        if freeze_language:
            for param in self.language_encoder.parameters():
                param.requires_grad = False
    
    def encode(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码视觉和语言输入
        
        Args:
            images: 输入图像 (B, 3, H, W)
            text_ids: 文本 token IDs (B, L)
            text_mask: 文本掩码 (B, L)
        
        Returns:
            融合特征 (B, L, D)
        """
        # 编码视觉
        vis_features = self.vision_encoder(images)  # (B, N, D)
        
        # 编码语言
        lang_features = self.language_encoder(text_ids, text_mask)  # (B, L, D)
        
        # 融合：语言 query 视觉 key/value
        fused, _ = self.fusion(
            query=lang_features,
            key=vis_features,
            value=vis_features
        )
        
        fused = self.fusion_norm(fused)
        
        return fused
    
    def predict_actions(
        self,
        fused_features: torch.Tensor
    ) -> torch.Tensor:
        """
        预测动作 tokens
        
        Args:
            fused_features: 融合特征 (B, L, D)
        
        Returns:
            动作 token logits (B, action_dim, num_bins)
        """
        # 使用最后一个 token 的特征
        x = fused_features[:, -1, :]  # (B, D)
        
        # 为每个动作维度预测
        action_logits = [
            head(x) for head in self.action_heads
        ]  # List of (B, num_bins)
        
        action_logits = torch.stack(action_logits, dim=1)  # (B, action_dim, num_bins)
        
        return action_logits
    
    def forward(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        完整前向传播
        
        Args:
            images: 输入图像 (B, 3, H, W)
            text_ids: 文本 token IDs (B, L)
            text_mask: 文本掩码 (B, L)
        
        Returns:
            动作 token logits (B, action_dim, num_bins)
        """
        fused = self.encode(images, text_ids, text_mask)
        action_logits = self.predict_actions(fused)
        return action_logits
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        自回归生成动作
        
        Args:
            images: 输入图像
            text_ids: 文本 token IDs
            text_mask: 文本掩码
            temperature: 采样温度
        
        Returns:
            生成的动作 tokens (action_dim,)
        """
        self.eval()
        
        # 编码
        fused = self.encode(images, text_ids, text_mask)
        
        # 预测动作 logits
        action_logits = self.predict_actions(fused)  # (B, action_dim, num_bins)
        
        # 采样
        if temperature > 0:
            probs = F.softmax(action_logits / temperature, dim=-1)
            # 从分布中采样
            action_tokens = torch.multinomial(
                probs.view(-1, self.num_action_bins),
                num_samples=1
            ).view(action_logits.shape[:2])
        else:
            # 贪婪解码
            action_tokens = action_logits.argmax(dim=-1)
        
        return action_tokens[0].cpu().numpy()


# ============================================================
# 5. 数据集
# ============================================================

class RT2Dataset(Dataset):
    """
    RT-2 训练数据集
    
    每条数据：
    - 图像
    - 文本指令
    - 动作序列（已 token 化）
    """
    
    def __init__(
        self,
        data_path: str,
        image_size: int = 224,
        max_text_len: int = 128,
        action_dim: int = 7,
        num_action_bins: int = 256
    ):
        super().__init__()
        
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.action_dim = action_dim
        self.num_action_bins = num_action_bins
        self.vocab_size = 32000
        
        # 模拟数据大小
        self.size = 1000
        
        # 动作 Token 化器
        self.action_tokenizer = ActionTokenizer(
            action_dim=action_dim,
            num_bins=num_action_bins
        )
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 模拟数据（实际使用时替换为真实数据）
        
        # 随机图像
        image = torch.randn(3, self.image_size, self.image_size)
        
        # 随机文本
        text_len = np.random.randint(10, self.max_text_len)
        text_ids = torch.randint(0, self.vocab_size, (text_len,))
        text_mask = torch.zeros(self.max_text_len, dtype=torch.bool)
        text_mask[:text_len] = True
        
        # 随机连续动作
        continuous_actions = np.random.uniform(-1, 1, self.action_dim)
        
        # 离散化为 tokens
        action_tokens = self.action_tokenizer.discretize(continuous_actions)
        action_tokens = torch.from_numpy(action_tokens).long()
        
        return {
            "image": image,
            "text_ids": text_ids,
            "text_mask": text_mask,
            "action_tokens": action_tokens,
            "continuous_actions": torch.from_numpy(continuous_actions).float()
        }


# ============================================================
# 6. 训练器
# ============================================================

class RT2Trainer:
    """
    RT-2 模型训练器
    """
    
    def __init__(
        self,
        model: RT2Model,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
    
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
            action_tokens = batch["action_tokens"].to(self.device)  # (B, action_dim)
            
            # 前向传播
            action_logits = self.model(images, text_ids, text_mask)  # (B, action_dim, num_bins)
            
            # 计算损失：每个动作维度的交叉熵
            loss = 0.0
            for i in range(self.model.action_dim):
                loss += self.criterion(
                    action_logits[:, i, :],  # (B, num_bins)
                    action_tokens[:, i]      # (B,)
                )
            loss = loss / self.model.action_dim
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        self.scheduler.step()
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Tuple[float, float]:
        """
        评估模型
        
        Returns:
            (平均损失，动作准确率)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        for batch in dataloader:
            images = batch["image"].to(self.device)
            text_ids = batch["text_ids"].to(self.device)
            text_mask = batch["text_mask"].to(self.device)
            action_tokens = batch["action_tokens"].to(self.device)
            
            action_logits = self.model(images, text_ids, text_mask)
            
            # 损失
            loss = 0.0
            for i in range(self.model.action_dim):
                loss += self.criterion(
                    action_logits[:, i, :],
                    action_tokens[:, i]
                )
            loss = loss / self.model.action_dim
            
            # 准确率
            pred_tokens = action_logits.argmax(dim=-1)
            correct = (pred_tokens == action_tokens).sum().item()
            total_correct += correct
            total_samples += action_tokens.numel()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy


# ============================================================
# 7. 仿真环境
# ============================================================

class RT2RobotEnv:
    """
    RT-2 机器人仿真环境
    
    模拟 7 自由度机械臂控制任务
    """
    
    def __init__(
        self,
        image_size: int = 224,
        max_steps: int = 50,
        action_dim: int = 7
    ):
        self.image_size = image_size
        self.max_steps = max_steps
        self.action_dim = action_dim
        self.step_count = 0
        
        # 机器人状态
        self.joint_angles = np.zeros(action_dim)
        self.target_angles = np.random.uniform(-1, 1, action_dim)
        
        # 动作 Token 化器
        self.action_tokenizer = ActionTokenizer(
            action_dim=action_dim,
            num_bins=256
        )
    
    def reset(self) -> Dict[str, any]:
        """重置环境"""
        self.step_count = 0
        self.joint_angles = np.zeros(self.action_dim)
        self.target_angles = np.random.uniform(-1, 1, self.action_dim)
        
        # 生成观测
        image = np.random.randn(3, self.image_size, self.image_size)
        instruction = "将机械臂移动到目标位置"
        
        return {
            "image": image,
            "instruction": instruction,
            "robot_state": self.joint_angles.copy()
        }
    
    def step(self, action_tokens: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        执行动作
        
        Args:
            action_tokens: 动作 tokens (action_dim,)
        
        Returns:
            obs, reward, done, info
        """
        self.step_count += 1
        
        # 将 tokens 转换为连续动作
        continuous_action = self.action_tokenizer.continuous(action_tokens)
        
        # 更新机器人状态
        self.joint_angles = continuous_action
        
        # 计算奖励（基于与目标的距离）
        distance = np.linalg.norm(self.joint_angles - self.target_angles)
        reward = np.exp(-distance * 5)  # 指数奖励
        
        # 检查是否完成
        done = (distance < 0.1) or (self.step_count >= self.max_steps)
        
        # 生成新观测
        image = np.random.randn(3, self.image_size, self.image_size)
        instruction = "将机械臂移动到目标位置"
        
        obs = {
            "image": image,
            "instruction": instruction,
            "robot_state": self.joint_angles.copy()
        }
        
        info = {
            "distance": distance,
            "steps": self.step_count,
            "target": self.target_angles.copy()
        }
        
        return obs, reward, done, info


# ============================================================
# 8. 使用示例
# ============================================================

def main():
    """
    主函数：演示 RT-2 模型的创建、训练和推理
    """
    print("=" * 60)
    print("RT-2 模型示例 - 第 17 章")
    print("=" * 60)
    
    # 1. 创建动作 Token 化器
    print("\n1. 创建动作 Token 化器...")
    tokenizer = ActionTokenizer(action_dim=7, num_bins=256)
    
    # 测试 Token 化
    test_action = np.array([0.5, -0.3, 0.8, -0.5, 0.2, -0.7, 0.4])
    tokens = tokenizer.discretize(test_action)
    recovered = tokenizer.continuous(tokens)
    
    print(f"   原始动作：{test_action}")
    print(f"   Token IDs: {tokens}")
    print(f"   恢复动作：{recovered}")
    print(f"   最大误差：{np.abs(test_action - recovered).max():.4f}")
    
    # 2. 创建 RT-2 模型
    print("\n2. 创建 RT-2 模型...")
    model = RT2Model(
        image_size=224,
        vocab_size=32000,
        embed_dim=768,
        action_dim=7,
        num_action_bins=256
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型总参数量：{total_params:,}")
    
    # 3. 创建训练器
    print("\n3. 创建训练器...")
    trainer = RT2Trainer(model, learning_rate=1e-4)
    
    # 4. 创建数据集
    print("\n4. 创建数据集...")
    dataset = RT2Dataset("data/train", image_size=224)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 5. 训练（演示用）
    print("\n5. 开始训练（演示：1 个 epoch）...")
    train_loss = trainer.train_epoch(dataloader, epoch=1)
    print(f"   训练损失：{train_loss:.4f}")
    
    # 评估
    print("\n6. 评估模型...")
    eval_loss, accuracy = trainer.evaluate(dataloader)
    print(f"   评估损失：{eval_loss:.4f}")
    print(f"   动作准确率：{accuracy:.2%}")
    
    # 7. 推理示例
    print("\n7. 推理示例...")
    model.eval()
    
    # 模拟输入
    images = torch.randn(1, 3, 224, 224)
    text_ids = torch.randint(0, 32000, (1, 20))
    text_mask = torch.ones(128, dtype=torch.bool)
    text_mask[20:] = False
    
    # 生成动作
    action_tokens = model.generate(images, text_ids, text_mask)
    print(f"   生成的动作 tokens: {action_tokens}")
    
    # 转换为连续动作
    continuous_action = tokenizer.continuous(action_tokens)
    print(f"   连续动作：{continuous_action}")
    
    # 8. 环境交互
    print("\n8. 仿真环境交互...")
    env = RT2RobotEnv(image_size=224)
    
    obs = env.reset()
    print(f"   初始距离：{np.linalg.norm(obs['robot_state'] - env.target_angles):.3f}")
    
    # 模拟几步
    total_reward = 0.0
    for step in range(5):
        # 使用模型预测（这里用随机 token 模拟）
        action_tokens = np.random.randint(0, 256, 7)
        
        obs, reward, done, info = env.step(action_tokens)
        total_reward += reward
        print(f"   步骤 {step+1}: 奖励={reward:.3f}, 距离={info['distance']:.3f}")
        
        if done:
            print("   任务完成！")
            break
    
    print(f"   总奖励：{total_reward:.3f}")
    
    # 总结
    print("\n" + "=" * 60)
    print("RT-2 模型示例完成！")
    print("=" * 60)
    
    print("\n关键要点：")
    print("1. RT-2 将机器人控制表述为序列预测问题")
    print("2. 动作被离散化为 tokens，与文本 token 统一处理")
    print("3. 利用预训练 VLM 的知识实现零样本泛化")
    print("4. 端到端训练从视觉 - 语言到动作的映射")
    
    print("\n与 VLA 的关系：")
    print("- RT-2 是 VLA 模型的具体实现")
    print("- 核心创新：动作 token 化 + 大规模 VLM 预训练")
    print("- 展现了强大的零样本推理和泛化能力")
    
    return model


if __name__ == "__main__":
    model = main()
