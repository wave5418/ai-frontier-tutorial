# 第 12 章 视觉语言模型 (Vision-Language Models, VLM)

## 12.1 VLM 定义与任务

### 12.1.1 什么是视觉语言模型

视觉语言模型（Vision-Language Models, VLM）是专门设计用于理解和生成视觉 - 语言内容的 AI 模型。与多模态大模型（LMM）相比，VLM 更专注于视觉和语言的深度整合。

**核心能力：**
- 视觉理解：识别、检测、分割图像内容
- 语言理解：解析问题、指令、上下文
- 跨模态推理：结合视觉和语言信息进行推理
- 内容生成：生成描述、答案、对话

### 12.1.2 主要任务

| 任务 | 输入 | 输出 | 示例 |
|------|------|------|------|
| **VQA** (视觉问答) | 图像 + 问题 | 答案 | "这是什么动物？" → "猫" |
| **图像描述** (Image Captioning) | 图像 | 描述文本 | "一只黑猫坐在窗台上" |
| **视觉对话** (Visual Dialog) | 图像 + 对话历史 | 回复 | 多轮图像相关对话 |
| **视觉定位** (Visual Grounding) | 图像 + 文本 | 边界框 | "找到红色的车" → [x,y,w,h] |
| **OCR-VQA** | 图像 (含文字) | 答案 | "招牌上写的什么？" |
| **文档理解** | 文档图像 | 结构化信息 | 表格、表单解析 |

### 12.1.3 VLM 与 LMM 的区别

```
LMM (多模态大模型):
- 更通用，支持多种模态
- 强调对话和指令遵循
- 通常基于大型语言模型扩展

VLM (视觉语言模型):
- 专注视觉 - 语言任务
- 强调视觉理解深度
- 可能包含专用视觉模块
```

## 12.2 ViT + LLM 架构

### 12.2.1 基本架构

```
┌─────────────────────────────────────────────────────────┐
│                      VLM 架构                            │
├─────────────────────────────────────────────────────────┤
│  ┌───────────┐    ┌───────────┐    ┌───────────────┐   │
│  │   ViT     │ →  │  投影层   │ →  │     LLM       │   │
│  │ (编码器)  │    │ (Adapter) │    │ (解码器/生成) │   │
│  └───────────┘    └───────────┘    └───────────────┘   │
│       ↓                  ↓                  ↓           │
│   视觉特征          对齐特征          语言生成          │
└─────────────────────────────────────────────────────────┘
```

### 12.2.2 Vision Transformer (ViT)

ViT 将图像视为序列的 patch：

```python
# ViT 处理流程
1. 图像分块：H×W → (H/P)×(W/P) 个 patch
2. 线性投影：每个 patch → embedding
3. 位置编码：添加位置信息
4. Transformer 编码：自注意力处理
5. 输出：[CLS] token 或所有 patch 特征
```

### 12.2.3 连接策略

**策略 1：早期融合 (Early Fusion)**
```
图像特征 + 文本嵌入 → 拼接 → LLM
优点：充分交互
缺点：计算量大
```

**策略 2：晚期融合 (Late Fusion)**
```
图像 → ViT → 特征
文本 → LLM → 表示
特征 + 表示 → 融合 → 输出
优点：模块化
缺点：交互有限
```

**策略 3：交叉注意力 (Cross-Attention)**
```
文本 Query ← 交叉注意力 → 图像 Key/Value
优点：动态关注相关视觉区域
缺点：需要额外训练
```

## 12.3 视觉指令微调

### 12.3.1 指令数据格式

```json
{
  "id": "vqa_001",
  "image": "image_001.jpg",
  "instruction": "这张图片里有多少个人？",
  "output": "图片中有 3 个人。"
}
```

### 12.3.2 指令类型

1. **描述类**：描述图像内容
2. **推理类**：需要逻辑推理
3. **知识类**：需要外部知识
4. **OCR 类**：识别图像中的文字
5. **计数类**：数物体数量
6. **定位类**：指出物体位置

### 12.3.3 微调策略

**全量微调：**
- 解冻所有参数
- 需要大量数据和计算资源
- 效果最好但成本高

**参数高效微调：**
- **LoRA**: 低秩适配器
- **Adapter**: 插入小型模块
- **Prefix Tuning**: 可学习前缀

```python
# LoRA 示例
class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8):
        self.linear = linear
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
    
    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x))
```

### 12.3.4 训练技巧

1. **多任务学习**：同时训练多种 VLM 任务
2. **课程学习**：从简单到复杂
3. **数据增强**：图像和文本增强
4. **负样本挖掘**：困难样本优先

## 12.4 高分辨率图像处理

### 12.4.1 挑战

标准 ViT 通常处理 224×224 或 336×336 的图像，但：
- 高分辨率图像包含更多细节
- 小物体可能丢失
- 文本识别需要高分辨率

### 12.4.2 解决方案

**方案 1：分块处理 (Tiling)**
```
将大图像分成多个重叠的 patch
分别处理每个 patch
聚合特征
```

**方案 2：多尺度处理**
```
同时处理多个分辨率
融合多尺度特征
```

**方案 3：动态分辨率**
```
根据图像内容动态选择分辨率
平衡计算和精度
```

### 12.4.3 LLaVA-1.5 高分辨率方案

```python
# AnyRes 策略
1. 将图像分割成多个 crop
2. 每个 crop 通过 ViT 编码
3. 聚合所有 crop 的特征
4. 保持空间关系
```

## 12.5 视频理解扩展

### 12.5.1 视频 VLM 架构

```
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│  帧采样   │ →  │  图像编码  │ →  │  时间建模  │ →  │  语言生成  │
└───────────┘    └───────────┘    └───────────┘    └───────────┘
```

### 12.5.2 时间建模方法

**方法 1：帧级特征 + 时间注意力**
```python
# 对帧序列应用时间注意力
frame_features = [ViT(frame) for frame in frames]
temporal_features = TemporalAttention(frame_features)
```

**方法 2：3D ViT**
```
直接处理时空立方体
计算量大但捕捉时间动态
```

**方法 3：帧间差分**
```
计算相邻帧的差异
捕捉运动信息
```

### 12.5.3 视频任务

- **视频问答** (Video QA)
- **视频描述** (Video Captioning)
- **动作识别** (Action Recognition)
- **事件定位** (Event Localization)

## 12.6 主流 VLM 对比

### 12.6.1 模型对比表

| 模型 | 视觉编码器 | 语言模型 | 连接方式 | 特点 |
|------|-----------|---------|---------|------|
| **LLaVA** | CLIP ViT-L | Vicuna (LLaMA) | MLP 投影 | 简单高效，开源 |
| **LLaVA-1.5** | CLIP ViT-L | Vicuna | MLP + 高分辨率 | 改进的推理能力 |
| **Qwen-VL** | ViT | Qwen | 交叉注意力 | 多语言支持 |
| **CogVLM** | ViT | LLM + 视觉专家 | 并行架构 | 视觉专家模块 |
| **BLIP-2** | ViT | OPT/LLaMA | Q-Former | 参数高效 |
| **InstructBLIP** | ViT | Vicuna | Q-Former | 指令微调 |
| **MiniGPT-4** | ViT | Vicuna | 线性投影 | 早期工作 |

### 12.6.2 LLaVA 系列

```
LLaVA (2023.04):
- 首个开源端到端 VLM
- CLIP ViT-L/14 + Vicuna
- 简单 MLP 投影

LLaVA-1.5 (2023.10):
- 改进训练策略
- 更高分辨率支持
- 更好的推理能力

LLaVA-NeXT (2024):
- 支持更多模态
- 更强的推理
- 更大的上下文
```

### 12.6.3 Qwen-VL

阿里巴巴通义千问视觉版本：

```
特点:
- 支持中英文
- 4 阶段训练
- 细粒度理解
- OCR 能力

架构:
- ViT (视觉)
- Qwen (语言)
- 交叉注意力连接
```

### 12.6.4 CogVLM

```
创新点:
- 视觉专家模块 (Visual Expert)
- 并行处理视觉和语言
- 无需额外视觉 token

优势:
- 更好的视觉理解
- 保持语言能力
- 高效推理
```

## 12.7 代码实现示例

详见 `code/vlm.py`，包含：
- 完整 VLM 实现
- VQA 示例
- 图像描述生成
- 视觉对话示例

## 12.8 实践建议

### 12.8.1 模型选择

**资源有限：**
- MiniGPT-4 或早期 LLaVA
- 使用量化版本

**追求性能：**
- LLaVA-1.5 或 LLaVA-NeXT
- Qwen-VL-Chat

**特定任务：**
- OCR 任务：Qwen-VL
- 视频理解：Video-LLaVA
- 多语言：Qwen-VL

### 12.8.2 部署考虑

1. **量化**：INT8/INT4 量化减少显存
2. **蒸馏**：大模型→小模型
3. **缓存**：KV Cache 加速推理
4. **批处理**：提高吞吐量

### 12.8.3 评估指标

- **VQA Accuracy**：问答准确率
- **CIDEr**：描述质量
- **BLEU/ROUGE**：文本生成质量
- **人工评估**：主观质量

## 12.9 参考文献

1. **LLaVA**: Liu et al. "Visual Instruction Tuning" (NeurIPS 2023)

2. **LLaVA-1.5**: Liu et al. "Improved Baselines with Visual Instruction Tuning" (arXiv 2023)

3. **Qwen-VL**: Bai et al. "Qwen-VL: A Versatile Vision-Language Model" (arXiv 2023)

4. **CogVLM**: Wang et al. "CogVLM: Visual Expert for Pretrained Language Models" (arXiv 2023)

5. **BLIP-2**: Li et al. "BLIP-2: Bootstrapping Language-Image Pre-training" (ICML 2023)

6. **InstructBLIP**: Dai et al. "InstructBLIP: Towards General-purpose Vision-Language Models" (NeurIPS 2023)

7. **Flamingo**: Alayrac et al. "Flamingo: a Visual Language Model for Few-Shot Learning" (NeurIPS 2022)

8. **ViT**: Dosovitskiy et al. "An Image is Worth 16x16 Words" (ICLR 2021)

9. **CLIP**: Radford et al. "Learning Transferable Visual Models" (ICML 2021)

10. **Video-LLaVA**: Lin et al. "Video-LLaVA: Learning United Visual Representation" (arXiv 2023)

---

*教程完 - 祝学习愉快！*
