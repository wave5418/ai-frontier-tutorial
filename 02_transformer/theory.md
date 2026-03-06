# 第 02 章 Transformer 架构

## 1. Transformer 背景

### 1.1 为什么需要 Transformer？

在 Transformer 出现之前，序列建模任务（如机器翻译、文本生成）主要依赖于**循环神经网络（RNN）**及其变体（LSTM、GRU）。然而，RNN 存在以下核心问题：

| 问题 | 描述 |
|------|------|
| **序列依赖** | RNN 必须按顺序处理序列，无法并行化，训练速度慢 |
| **长距离依赖** | 即使使用 LSTM，长序列中的远距离信息仍会衰减 |
| **计算效率** | 时间复杂度 O(n)，无法充分利用 GPU 并行计算 |

### 1.2 Transformer 的突破

2017 年，Google 团队在论文《**Attention Is All You Need**》中提出了 Transformer 架构，其核心思想是：

> **完全抛弃循环和卷积，仅使用注意力机制**

**Transformer 的优势：**
- ✅ **并行化**：所有位置可同时计算，训练速度提升数个数量级
- ✅ **长距离依赖**：任意两个位置间的距离为 O(1)
- ✅ **可扩展性**：更容易扩展到更长序列和更大数据集

---

## 2. Self-Attention 机制详解

### 2.1 核心思想

Self-Attention（自注意力）允许序列中的每个位置关注序列中的所有其他位置，从而捕获全局依赖关系。

### 2.2 数学公式

对于输入序列 $X = [x_1, x_2, ..., x_n]$，每个 $x_i \in \mathbb{R}^{d_{model}}$：

**步骤 1：线性变换**
$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

其中：
- $Q$ (Query)：查询矩阵，表示"我想找什么"
- $K$ (Key)：键矩阵，表示"我有什么"
- $V$ (Value)：值矩阵，表示"实际内容"
- $W^Q, W^K, W^V \in \mathbb{R}^{d_{model} \times d_k}$ 为可学习参数

**步骤 2：注意力分数计算**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**缩放因子 $\sqrt{d_k}$ 的作用：**
- 防止点积结果过大导致 softmax 梯度消失
- 保持数值稳定性

### 2.3 图示说明

```
输入序列: [我, 爱, 机, 器, 学, 习]
            ↓
        嵌入层 (Embedding)
            ↓
    ┌───────┴───────┐
    Q       K       V   (线性变换)
    ↓       ↓       ↓
    └───┬───┴───┬───┘
        ↓   ↓   ↓
    Q×K^T (注意力分数矩阵)
        ↓
    softmax (归一化)
        ↓
    × V (加权求和)
        ↓
输出: 每个词的新表示（融合了全局信息）
```

**注意力矩阵示例（6×6）：**

|     | 我   | 爱   | 机   | 器   | 学   | 习   |
|-----|------|------|------|------|------|------|
| **我** | 0.4  | 0.3  | 0.1  | 0.1  | 0.05 | 0.05 |
| **爱** | 0.3  | 0.4  | 0.1  | 0.1  | 0.05 | 0.05 |
| **机** | 0.1  | 0.1  | 0.3  | 0.3  | 0.1  | 0.1  |
| **器** | 0.1  | 0.1  | 0.3  | 0.3  | 0.1  | 0.1  |
| **学** | 0.05 | 0.05 | 0.1  | 0.1  | 0.35 | 0.35 |
| **习** | 0.05 | 0.05 | 0.1  | 0.1  | 0.35 | 0.35 |

> 每行和为 1，表示当前词对其他词的关注程度

---

## 3. Multi-Head Attention

### 3.1 动机

单个注意力头只能捕获一种类型的依赖关系。**Multi-Head Attention** 通过并行使用多个注意力头，让模型能够同时关注不同子空间的信息。

### 3.2 结构

```
         输入 X
           │
    ┌──────┴──────┬──────────┐
    ↓             ↓          ↓
  Head₁         Head₂      Headₙ
 (Q₁,K₁,V₁)   (Q₂,K₂,V₂)  (Qₙ,Kₙ,Vₙ)
    ↓             ↓          ↓
  Attention₁   Attention₂  Attentionₙ
    ↓             ↓          ↓
    └──────┬──────┴──────────┘
           │
        Concat (拼接)
           │
        Linear (线性变换)
           │
         输出
```

### 3.3 数学表达

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**参数维度：**
- $d_{model} = 512$（模型维度）
- $h = 8$（注意力头数）
- $d_k = d_v = d_{model}/h = 64$（每头维度）

---

## 4. Position Encoding（位置编码）

### 4.1 为什么需要位置编码？

Transformer 没有循环结构，**无法天然感知序列顺序**。位置编码为每个位置注入顺序信息。

### 4.2 正弦位置编码

原始论文使用正弦和余弦函数：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**优点：**
- 可以外推到比训练时更长的序列
- $PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数，便于学习相对位置

### 4.3 图示

```
位置 0: [sin(θ₀), cos(θ₀), sin(θ₁), cos(θ₁), ...]
位置 1: [sin(θ₀+Δ), cos(θ₀+Δ), sin(θ₁+Δ), cos(θ₁+Δ), ...]
位置 2: [sin(θ₀+2Δ), cos(θ₀+2Δ), sin(θ₁+2Δ), cos(θ₁+2Δ), ...]
```

### 4.4 可学习位置编码

现代实现（如 BERT）常使用**可学习的位置嵌入**，效果相当且更简单：

```python
self.pos_embedding = nn.Embedding(max_len, d_model)
```

---

## 5. Encoder-Decoder 架构

### 5.1 整体结构

```
┌─────────────────────────────────────────────────────────┐
│                      ENCODER                            │
│  ┌─────────────┐    ┌─────────────┐    (重复 N=6 层)    │
│  │ Multi-Head  │ →  │   Feed      │ →  Add & Norm      │
│  │  Attention  │    │   Forward   │                    │
│  └─────────────┘    └─────────────┘                    │
└─────────────────────────────────────────────────────────┘
                          ↓
                    Encoder 输出
                          ↓
┌─────────────────────────────────────────────────────────┐
│                      DECODER                            │
│  ┌─────────────┐    ┌─────────────┐    (重复 N=6 层)    │
│  │ Masked      │ →  │ Multi-Head  │ →  Feed Forward   │
│  │  Attention  │    │  Attention  │    Add & Norm      │
│  │ (自注意力)   │    │ (交叉注意力)│                    │
│  └─────────────┘    └─────────────┘                    │
└─────────────────────────────────────────────────────────┘
                          ↓
                    Linear + Softmax → 输出概率
```

### 5.2 Encoder

- **输入**：源序列嵌入 + 位置编码
- **输出**：上下文表示（记忆）
- **层数**：通常 6 层

### 5.3 Decoder

- **输入**：目标序列（右移一位）+ 位置编码
- **Masked Attention**：防止看到未来位置（因果掩码）
- **Cross Attention**：关注 Encoder 输出
- **输出**：下一个词的概率分布

### 5.4 训练 vs 推理

| 阶段 | Encoder 输入 | Decoder 输入 |
|------|-------------|-------------|
| **训练** | 完整源序列 | 完整目标序列（右移） |
| **推理** | 完整源序列 | 已生成的部分序列（自回归） |

---

## 6. Layer Normalization

### 6.1 作用

Layer Normalization（层归一化）稳定训练过程，加速收敛。

### 6.2 公式

对于输入 $x = [x_1, ..., x_d]$：

$$\mu = \frac{1}{d}\sum_{i=1}^{d}x_i$$
$$\sigma = \sqrt{\frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2 + \epsilon}$$
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$

其中 $\gamma, \beta$ 为可学习参数。

### 6.3 Pre-LN vs Post-LN

**原始 Transformer（Post-LN）：**
```
x → Attention → Add → LayerNorm → FFN → Add → LayerNorm → 输出
```

**现代改进（Pre-LN）：**
```
x → LayerNorm → Attention → Add → LayerNorm → FFN → Add → 输出
```

> Pre-LN 更稳定，更适合深层网络

---

## 7. 代码实现示例（PyTorch）

完整代码见：`code/transformer.py`

### 7.1 核心组件

| 组件 | 文件位置 | 说明 |
|------|---------|------|
| SelfAttention | `SelfAttention` 类 | 缩放点积注意力 |
| MultiHeadAttention | `MultiHeadAttention` 类 | 多头注意力 |
| PositionalEncoding | `PositionalEncoding` 类 | 正弦位置编码 |
| TransformerEncoder | `TransformerEncoder` 类 | 编码器层 |
| Transformer | `Transformer` 类 | 完整模型 |

### 7.2 使用示例

```python
# 创建模型
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_len=512
)

# 前向传播
src = torch.randint(0, 10000, (32, 100))  # batch=32, seq_len=100
tgt = torch.randint(0, 10000, (32, 80))
output = model(src, tgt)  # output: (32, 80, 10000)
```

---

## 8. 参考文献

1. **Vaswani, A., et al. (2017).** ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
   - Transformer 原始论文，必读

2. **Devlin, J., et al. (2019).** ["BERT: Pre-training of Deep Bidirectional Transformers"](https://arxiv.org/abs/1810.04805). NAACL 2019.
   - Transformer 在 NLP 的里程碑应用

3. **Dosovitskiy, A., et al. (2021).** ["An Image is Worth 16x16 Words: Transformers for Image Recognition"](https://arxiv.org/abs/2010.11929). ICLR 2021.
   - Vision Transformer (ViT)

4. **Radford, A., et al. (2019).** ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf). GPT-2.
   - Decoder-only Transformer

5. **Brown, T., et al. (2020).** ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165). NeurIPS 2020.
   - GPT-3，展示 Transformer 的规模效应

6. **The Annotated Transformer.** [Harvard NLP](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
   - 论文代码逐行注释，极佳学习资源

---

## 附录：关键超参数（原始论文）

| 参数 | Base 模型 | Big 模型 |
|------|----------|---------|
| $d_{model}$ | 512 | 1024 |
| $d_{ff}$ | 2048 | 4096 |
| $n_{heads}$ | 8 | 16 |
| $n_{layers}$ | 6 | 6 |
| Dropout | 0.1 | 0.3 |
| Label Smoothing | 0.1 | 0.1 |
