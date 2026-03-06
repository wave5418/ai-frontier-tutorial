# 第 11 章 多模态大模型 (Large Multimodal Models)

## 11.1 多模态大模型定义与发展

### 11.1.1 什么是多模态大模型

多模态大模型（Large Multimodal Models, LMM）是能够同时理解和生成多种模态数据（如文本、图像、音频、视频等）的人工智能模型。与传统的单模态模型相比，LMM 具有以下特点：

- **跨模态理解**：能够理解不同模态之间的语义关联
- **联合推理**：在多种模态信息基础上进行推理
- **多模态生成**：可以生成文本、图像或跨模态内容

### 11.1.2 发展历程

```
2021: CLIP (对比学习) → 2022: Flamingo (视觉语言) → 2023: LLaVA (指令微调)
         ↓                      ↓                        ↓
    图像 - 文本对齐          少样本学习              对话式多模态
```

关键里程碑：
1. **CLIP (2021)**：通过对比学习实现图像 - 文本对齐
2. **Flamingo (2022)**：引入 Perceiver Resampler，实现少样本多模态学习
3. **BLIP-2 (2023)**：Q-Former 架构，高效连接视觉和语言
4. **LLaVA (2023)**：端到端多模态对话模型

## 11.2 多模态数据预处理

### 11.2.1 图像预处理

```python
# 图像标准化流程
1. 读取图像 (RGB 格式)
2. 调整大小 (通常 224×224 或 336×336)
3. 归一化 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
4. 转换为 Tensor
```

### 11.2.2 文本预处理

```python
# 文本 Tokenization
1. 使用 LLM 的 tokenizer
2. 添加特殊标记 (<image>, </image>)
3. 填充和截断到固定长度
```

### 11.2.3 数据对齐

图像 - 文本对需要确保语义一致性：
- 图像描述要准确反映图像内容
- 避免噪声标签
- 考虑使用数据清洗工具（如 CLIP 过滤）

## 11.3 视觉 - 语言对齐方法

### 11.3.1 对比学习 (Contrastive Learning)

CLIP 采用的方法：
- 图像编码器和文本编码器独立训练
- 通过对比损失拉近匹配对的嵌入，推远不匹配对
- 损失函数：InfoNCE Loss

```
L = -log(exp(sim(I, T) / τ) / Σ exp(sim(I, T_j) / τ))
```

### 11.3.2 生成式对齐

- 使用图像特征作为条件生成文本
- 通常采用交叉注意力机制
- 更适合需要详细理解的任务

### 11.3.3 对齐质量评估

1. **零样本分类**：在 ImageNet 等数据集上测试
2. **检索任务**：图像→文本和文本→图像检索
3. **下游任务**：VQA、图像描述等

## 11.4 Flamingo 架构详解

### 11.4.1 核心创新

Flamingo (DeepMind, 2022) 的关键设计：

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│  图像编码器  │ →  │ Perceiver        │ →  │  语言模型   │
│  (ViT/CLIP) │    │ Resampler        │    │  (LLaMA)    │
└─────────────┘    └──────────────────┘    └─────────────┘
     冻结                可训练              部分可训练
```

### 11.4.2 Perceiver Resampler

作用：将可变长度的视觉特征压缩为固定数量的视觉 token

```python
# Perceiver Resampler 核心逻辑
class PerceiverResampler(nn.Module):
    def __init__(self, num_latents=64, dim=4096):
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.cross_attn = CrossAttention(dim)
    
    def forward(self, image_features):
        # image_features: [B, N, D] (N 是可变长度)
        # 输出：[B, num_latents, D] (固定长度)
        return self.cross_attn(self.latents, image_features)
```

### 11.4.3 交叉注意力块 (Cross-Attention Block)

```
每层 LLM 后插入 X 个交叉注意力块
- Key/Value: 视觉特征
- Query: 语言模型隐藏状态
- 使语言模型能够"看到"图像
```

### 11.4.4 训练策略

1. **预训练**：在大规模图像 - 文本对上训练
2. **少样本学习**：冻结大部分参数，仅微调少量层
3. **交错数据**：图像和文本交错输入

## 11.5 BLIP-2 与 Q-Former

### 11.5.1 BLIP-2 架构

BLIP-2 (Bootstrapping Language-Image Pre-training) 的核心思想：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  图像编码器  │ →  │   Q-Former   │ →  │  冻结的     │
│  (ViT)      │    │  (可训练)     │    │  LLM        │
└─────────────┘    └─────────────┘    └─────────────┘
```

### 11.5.2 Q-Former 设计

Q-Former 是一个轻量级的 Transformer 模块：

```python
class QFormer(nn.Module):
    def __init__(self, num_query=32, hidden_size=768):
        # 可学习的查询向量
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query, hidden_size))
        
        # 自注意力层 (处理查询)
        self.self_attention = nn.ModuleList([TransformerBlock()])
        
        # 交叉注意力层 (连接图像特征)
        self.cross_attention = nn.ModuleList([CrossAttentionBlock()])
```

### 11.5.3 两阶段训练

**阶段 1：图像 - 文本对比学习**
- Q-Former 学习提取与文本相关的视觉特征
- 使用对比损失对齐图像和文本

**阶段 2：生成式训练**
- Q-Former 输出作为 LLM 的软提示
- 训练 Q-Former 生成适合 LLM 的特征

### 11.5.4 优势

- **参数高效**：仅训练 Q-Former，冻结 LLM
- **灵活**：可以连接不同的图像编码器和 LLM
- **性能好**：在多个基准上达到 SOTA

## 11.6 LLaVA 架构详解

### 11.6.1 LLaVA 概述

LLaVA (Large Language and Vision Assistant) 是端到端的多模态对话模型：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  CLIP ViT   │ →  │  投影层      │ →  │  Vicuna     │
│  (L/14)     │    │  (MLP)      │    │  (LLaMA)    │
└─────────────┘    └─────────────┘    └─────────────┘
```

### 11.6.2 简单投影层

LLaVA 使用简单的 2 层 MLP 作为投影：

```python
class VisionProjector(nn.Module):
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.linear_1 = nn.Linear(vision_dim, llm_dim)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(llm_dim, llm_dim)
    
    def forward(self, image_features):
        # image_features: [B, N, vision_dim]
        x = self.linear_1(image_features)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x  # [B, N, llm_dim]
```

### 11.6.3 指令微调数据

LLaVA 创建了大规模多模态指令数据：
- **LLaVA-Instruct-150K**：15 万条指令数据
- 包含对话、详细描述、推理等任务
- 使用 GPT-4 辅助生成

### 11.6.4 两阶段训练

**阶段 1：特征对齐**
- 冻结 LLM 和 ViT
- 仅训练投影层
- 使用图像 - 文本对

**阶段 2：指令微调**
- 解冻 LLM（部分或全部）
- 使用指令数据微调
- 学习遵循指令和对话

## 11.7 多模态指令微调

### 11.7.1 指令格式

```
USER: <image>
这张图片里有什么？

ASSISTANT:
图片中显示了一只可爱的金毛犬...
```

### 11.7.2 数据格式

```json
{
  "id": "example_001",
  "image": "image_001.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\n问题内容"},
    {"from": "gpt", "value": "回答内容"}
  ]
}
```

### 11.7.3 训练技巧

1. **学习率调度**：使用 cosine decay
2. **梯度检查点**：节省显存
3. **混合精度**：FP16/BF16 训练
4. **LoRA**：参数高效微调

### 11.7.4 评估指标

- **VQA 准确率**：视觉问答准确度
- **CIDEr**：图像描述质量
- **人工评估**：对话质量和有用性

## 11.8 代码实现示例

详见 `code/lmm.py`，包含：
- 图像编码器实现
- 投影层设计
- 多模态 LLM 整合
- 训练流程

## 11.9 参考文献

1. **CLIP**: Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (ICML 2021)

2. **Flamingo**: Alayrac et al. "Flamingo: a Visual Language Model for Few-Shot Learning" (NeurIPS 2022)

3. **BLIP-2**: Li et al. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" (ICML 2023)

4. **LLaVA**: Liu et al. "Visual Instruction Tuning" (NeurIPS 2023)

5. **LLaVA-1.5**: Liu et al. "Improved Baselines with Visual Instruction Tuning" (arXiv 2023)

6. **Qwen-VL**: Bai et al. "Qwen-VL: A Versatile Vision-Language Model" (arXiv 2023)

7. **CogVLM**: Wang et al. "CogVLM: Visual Expert for Pretrained Language Models" (arXiv 2023)

---

*下一章：第 12 章 视觉语言模型 (VLM)*
