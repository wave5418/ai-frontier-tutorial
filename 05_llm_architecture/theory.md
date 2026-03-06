# 第 05 章 LLM 架构演进

## 1. LLM 定义与发展历程

### 1.1 什么是 LLM

**大语言模型（Large Language Model, LLM）** 是基于深度学习技术，在海量文本数据上训练得到的大规模神经网络模型。它们能够理解、生成和处理自然语言，展现出惊人的语言理解和推理能力。

**核心特征：**
- **规模巨大**：参数量从数十亿到数千亿不等
- **自监督学习**：通过预测下一个 token 的方式学习语言规律
- **涌现能力**：规模达到一定程度后，展现出训练时未明确教授的能力
- **通用性**：可应用于多种 NLP 任务，无需针对每个任务重新训练

### 1.2 发展历程

| 时期 | 代表模型 | 参数量 | 关键突破 |
|------|----------|--------|----------|
| 2017 | Transformer | 65M | 自注意力机制，并行训练 |
| 2018 | BERT | 340M | 双向编码，预训练 + 微调范式 |
| 2019 | GPT-2 | 1.5B | 大规模生成式预训练 |
| 2020 | GPT-3 | 175B | 少样本学习，涌现能力 |
| 2022 | Chinchilla | 70B | Scaling Law 指导最优训练 |
| 2023 | LLaMA | 7B-65B | 开源模型，高效架构设计 |
| 2024 | LLaMA 3 | 8B-70B | 更高质量数据，更强性能 |

---

## 2. Scaling Law 详解

### 2.1 什么是 Scaling Law

**Scaling Law（缩放定律）** 描述了模型性能与模型规模、数据量、计算量之间的数学关系。它指导我们如何最优地分配计算资源。

### 2.2 Chinchilla Scaling Law

DeepMind 在 2022 年提出的 Chinchilla 论文发现：

```
L(N, D) = E + A/N^α + B/D^β
```

其中：
- `L`：训练损失
- `N`：模型参数量
- `D`：训练 token 数量
- `E, A, B, α, β`：拟合常数

**关键发现：**
- 最优训练时，模型大小和训练数据应按比例增长
- 推荐比例：**1 token ≈ 20 参数**（例如 70B 模型应训练约 3.5T tokens）
- 之前的模型（如 GPT-3）普遍训练不足

### 2.3 实践指导

```
最优参数量 N_opt ∝ C^0.5
最优数据量 D_opt ∝ C^0.5
```

其中 C 是总计算预算。这意味着应该同时扩大模型和数据，而不是只扩大其中一个。

---

## 3. LLM 架构演进（从 Transformer 到 LLaMA）

### 3.1 原始 Transformer (2017)

**架构特点：**
- Encoder-Decoder 结构
- 多头自注意力（Multi-Head Attention）
- 位置编码（正弦/余弦）
- LayerNorm + 残差连接
- FFN：ReLU 激活

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### 3.2 GPT 系列演进

**GPT (2018)：**
- 仅 Decoder 结构
- 因果掩码（Causal Mask）
- 单向语言建模

**GPT-2 (2019)：**
- 预 LayerNorm（Pre-Norm）
- 更大的上下文窗口

**GPT-3 (2020)：**
- 稀疏注意力模式
- 更大的规模

### 3.3 LLaMA 架构 (2023)

LLaMA 在 GPT 基础上进行了多项改进：

| 组件 | GPT-3 | LLaMA | 改进原因 |
|------|-------|-------|----------|
| 位置编码 | 绝对位置编码 | RoPE | 更好的外推能力 |
| 激活函数 | GeLU | SwiGLU | 更好的表达力 |
| 归一化 | LayerNorm | RMSNorm | 更快，效果相当 |
| 注意力 | 标准 | 分组查询（GQA） | 减少 KV Cache 内存 |

---

## 4. 关键技术创新

### 4.1 RoPE（Rotary Position Embedding）

**旋转位置编码** 通过旋转矩阵将位置信息编码到查询和键向量中。

**核心思想：**
- 位置信息通过相对位置影响注意力
- 具有良好的外推性（能处理比训练时更长的序列）

**数学形式：**
```
q_m = RoPE(q, m)  # 位置 m 的查询
k_n = RoPE(k, n)  # 位置 n 的键

q_m^T k_n = f(q, k, m-n)  # 只依赖于相对位置
```

**优势：**
- 无需学习位置参数
- 支持长度外推
- 实现简单高效

### 4.2 SwiGLU（Swish Gated Linear Unit）

**SwiGLU** 是 GLU 变体，使用 Swish 激活函数：

```
SwiGLU(x) = Swish(xW) ⊗ (xV)
          = (xW · σ(xW)) ⊗ (xV)
```

其中：
- `W, V`：可学习权重矩阵
- `σ`：Sigmoid 函数
- `⊗`：逐元素乘法

**优势：**
- 比 ReLU/GeLU 表达力更强
- 门控机制允许信息流动控制
- 在 LLM 中表现优异

### 4.3 RMSNorm（Root Mean Square Layer Normalization）

**RMSNorm** 简化了 LayerNorm，仅使用均方根进行归一化：

```
RMSNorm(x) = (x / RMS(x)) · γ

RMS(x) = √(Σx_i² / n + ε)
```

**对比 LayerNorm：**
```
LayerNorm(x) = ((x - μ) / σ) · γ + β
```

**优势：**
- 省去均值计算，更快
- 省去偏置 β，参数更少
- 效果相当甚至更好

### 4.4 分组查询注意力（GQA）

**GQA** 在 MQA（多查询注意力）和 MHA（多头注意力）之间取得平衡：

- **MHA**：每个头有独立的 K、V
- **MQA**：所有头共享一组 K、V
- **GQA**：将头分组，每组共享 K、V

**优势：**
- 显著减少 KV Cache 内存
- 推理速度更快
- 性能损失很小

---

## 5. 训练技巧

### 5.1 混合精度训练（Mixed Precision）

**核心思想：** 同时使用 FP16 和 FP32 进行训练。

```
前向传播：FP16（快速，省内存）
梯度计算：FP32（避免精度损失）
权重更新：FP32 → FP16
```

**优势：**
- 减少内存占用（约 50%）
- 加速计算（Tensor Core 优化）
- 保持数值稳定性

### 5.2 梯度检查点（Gradient Checkpointing）

**问题：** 训练深层网络时，激活值占用大量内存。

**解决：** 只保存部分激活值，需要时重新计算。

```
# 标准训练：保存所有激活值
内存：O(N)  # N 为层数

# 梯度检查点：只保存检查点
内存：O(√N)  # 时间换空间
```

**代价：** 计算时间增加约 20-30%，但可训练更大模型。

### 5.3 其他训练技巧

| 技巧 | 作用 | 效果 |
|------|------|------|
| 学习率预热 | 逐步增加学习率 | 稳定训练初期 |
| 余弦退火 | 平滑降低学习率 | 更好收敛 |
| 梯度裁剪 | 限制梯度范数 | 防止爆炸 |
| 权重衰减 | L2 正则化 | 防止过拟合 |
| Flash Attention | 优化注意力计算 | 2-3x 加速 |

---

## 6. 推理优化

### 6.1 KV Cache

**问题：** 自回归生成时，重复计算已生成 token 的 K、V 值。

**解决：** 缓存历史 K、V 值，仅计算新 token。

```python
# 无 Cache：每步 O(n²)
for i in range(seq_len):
    K, V = compute_all(seq[:i+1])  # 重复计算
    
# 有 Cache：每步 O(n)
cache = {}
for i in range(seq_len):
    k_new, v_new = compute_new(token[i])
    cache[i] = (k_new, v_new)
    K, V = concatenate(cache)
```

**优势：**
- 推理速度提升 10-100x
- 内存占用随序列线性增长

### 6.2 量化（Quantization）

**核心思想：** 降低权重和激活的精度。

| 量化类型 | 精度 | 压缩比 | 质量损失 |
|----------|------|--------|----------|
| FP16 | 16-bit | 2x | 无 |
| INT8 | 8-bit | 4x | 微小 |
| INT4 | 4-bit | 8x | 可接受 |
| NF4 | 4-bit | 8x | 最小（LLM 优化） |

**常见量化方案：**
- **LLM.int8()**：仅量化权重，激活保持 FP16
- **QLoRA**：INT4 量化 + LoRA 微调
- **AWQ**：感知权重量化，保护重要权重

### 6.3 其他优化技术

- **投机采样（Speculative Sampling）**：小模型草稿，大模型验证
- **PagedAttention**：分页管理 KV Cache，减少碎片
- **连续批处理（Continuous Batching）**：动态调整批大小

---

## 7. 主流 LLM 对比

### 7.1 架构对比

| 模型 | 参数量 | 上下文 | 注意力 | 激活 | 位置编码 |
|------|--------|--------|--------|------|----------|
| LLaMA 2 | 7B-70B | 4K | MHA | SwiGLU | RoPE |
| LLaMA 3 | 8B-70B | 8K | GQA | SwiGLU | RoPE |
| ChatGLM3 | 6B | 32K | MQA | GeLU | RoPE |
| Qwen1.5 | 0.5B-72B | 32K | GQA | SwiGLU | RoPE |
| Mistral | 7B | 8K | GQA | SwiGLU | RoPE |
| Yi | 6B-34B | 32K-200K | MHA | SwiGLU | RoPE |

### 7.2 性能对比（MMLU 基准）

| 模型 | MMLU | 训练数据 | 开源 |
|------|------|----------|------|
| GPT-4 | 86.4 | 私有 | ❌ |
| Claude 3 Opus | 86.8 | 私有 | ❌ |
| LLaMA 3 70B | 79.5 | 15T tokens | ✅ |
| Qwen1.5 72B | 77.2 | 3T tokens | ✅ |
| Mistral 7B | 60.1 | 未知 | ✅ |
| ChatGLM3 6B | 55.3 | 中文优化 | ✅ |

### 7.3 选择建议

- **资源有限**：LLaMA 3 8B、Mistral 7B、Qwen1.5 7B
- **中文场景**：ChatGLM3、Qwen1.5、Yi
- **长上下文**：Yi-200K、Qwen1.5-32K、GLM-32K
- **最强开源**：LLaMA 3 70B、Qwen1.5 72B

---

## 8. 代码实现示例

详见 `code/llm_arch.py`，包含：
- RMSNorm 实现
- RoPE 旋转位置编码
- SwiGLU 激活函数
- LLaMA 风格 Decoder 层
- 完整 LLM 模型
- KV Cache 推理示例

---

## 9. 参考文献

### 核心论文

1. **Transformer**
   - Vaswani et al. "Attention Is All You Need" (NeurIPS 2017)
   - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

2. **Scaling Law**
   - Hoffmann et al. "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)
   - [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

3. **RoPE**
   - Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
   - [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

4. **SwiGLU**
   - Shazeer et al. "GLU Variants Improve Transformer" (2020)
   - [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)

5. **RMSNorm**
   - Zhang & Sennrich. "Root Mean Square Layer Normalization" (2019)
   - [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)

6. **LLaMA**
   - Touvron et al. "LLaMA: Open and Efficient Foundation Language Models" (2023)
   - [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)

7. **Flash Attention**
   - Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
   - [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

8. **QLoRA**
   - Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
   - [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

### 优秀实现

- [LLaMA 官方 GitHub](https://github.com/meta-llama/llama)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - C++ 高效推理

---

*本章完*
