# 第 03 章 BERT 与双向编码

## 1. BERT 背景与贡献

### 1.1 背景

BERT（Bidirectional Encoder Representations from Transformers）是由 Google AI 语言团队在 2018 年提出的预训练语言模型。在 BERT 之前，自然语言处理领域的预训练模型主要采用单向语言模型：

- **ELMo**（2018）：使用双向 LSTM，但是是浅层双向，两个方向独立
- **GPT**（2018）：使用 Transformer Decoder，仅从左到右单向建模
- **OpenAI Transformer**：同样采用单向语言模型

这些模型的局限性在于无法同时利用上下文信息，限制了语言理解能力。

### 1.2 主要贡献

BERT 的核心贡献包括：

1. **深度双向表示**：首次成功训练深度双向 Transformer Encoder
2. **预训练 + 微调范式**：提出通用的预训练任务，可在多种下游任务上微调
3. **SOTA 性能**：在 11 个 NLP 任务上刷新了最佳成绩
4. **开源模型**：发布预训练模型，推动 NLP 社区发展

### 1.3 模型版本

| 模型 | 层数 | 隐藏层维度 | 注意力头数 | 参数量 |
|------|------|-----------|-----------|--------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

---

## 2. 双向 Encoder 架构

### 2.1 Transformer Encoder 结构

BERT 完全基于 Transformer 的 Encoder 部分，包含以下组件：

```
输入嵌入 → [Encoder Layer] × N → 输出
              ↓
        Multi-Head Self-Attention
              ↓
        Add & LayerNorm
              ↓
        Feed-Forward Network
              ↓
        Add & LayerNorm
```

### 2.2 输入表示

BERT 的输入是三种嵌入的和：

```
Input = Token Embedding + Segment Embedding + Position Embedding
```

- **Token Embedding**：词表嵌入（WordPiece 分词）
- **Segment Embedding**：句子 A/B 标识（用于 NSP 任务）
- **Position Embedding**：位置编码（可学习，最大长度 512）

### 2.3 特殊 Token

- `[CLS]`：分类标记，放在句首，用于分类任务
- `[SEP]`：分隔标记，分隔句子对
- `[MASK]`：掩码标记，用于 MLM 任务

### 2.4 注意力机制

Self-Attention 公式：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Multi-Head Attention 并行多个注意力头，捕捉不同子空间的信息。

---

## 3. MLM（Masked Language Model）任务详解

### 3.1 任务设计

MLM 是 BERT 的核心预训练任务，灵感来自完形填空：

1. 随机遮蔽输入中 15% 的 Token
2. 模型预测被遮蔽的 Token
3. 使用交叉熵损失训练

### 3.2 遮蔽策略

对于选中的 15% Token，采用以下策略：

- **80%** 替换为 `[MASK]`
- **10%** 替换为随机词
- **10%** 保持不变

这种策略的好处：
- 防止模型只依赖 `[MASK]` 标记
- 强制模型学习真正的语言表示
- 微调时没有 `[MASK]`，保持一致性

### 3.3 损失函数

```python
L_MLM = -Σ log P(w_i | context)
```

其中 w_i 是被遮蔽的词，context 是双向上下文。

### 3.4 示例

```
输入：我 爱 学 习 机 器 学 习
掩码：我 [MASK] 学 习 [MASK] 器 学 习
预测：我 [爱] 学 习 [机] 器 学 习
```

---

## 4. NSP（Next Sentence Prediction）任务

### 4.1 任务设计

NSP 用于学习句子间关系，对问答、推理任务重要：

1. 输入句子对 (A, B)
2. 50% 情况下 B 是 A 的下一句（标签：IsNext）
3. 50% 情况下 B 是随机句子（标签：NotNext）
4. 模型预测句子关系

### 4.2 输入格式

```
[CLS] 句子 A [SEP] 句子 B [SEP]
```

- 句子 A 和 B 使用不同的 Segment Embedding
- `[CLS]` 的输出用于分类

### 4.3 损失函数

```python
L_NSP = -log P(label | [CLS]_output)
```

### 4.4 总损失

```python
L_total = L_MLM + L_NSP
```

---

## 5. BERT 预训练流程

### 5.1 数据准备

- **语料**：BookCorpus（800M 词）+ Wikipedia（2500M 词）
- **分词**：WordPiece 词表（30000 词）
- **序列长度**：128（90%），512（10%）

### 5.2 训练配置

| 超参数 | BERT-Base | BERT-Large |
|--------|-----------|------------|
| Batch Size | 256 | 256 |
| 学习率 | 1e-4 | 1e-4 |
| 训练步数 | 100K | 100K |
| 优化器 | Adam | Adam |
| Dropout | 0.1 | 0.1 |

### 5.3 训练步骤

```
1. 加载文本语料
2. 分词并添加特殊 Token
3. 随机遮蔽 Token（MLM）
4. 构建句子对（NSP）
5. 前向传播计算损失
6. 反向传播更新参数
7. 重复直到收敛
```

### 5.4 训练时间

- BERT-Base：4 天（4 个 TPU v3）
- BERT-Large：1 周（16 个 TPU v3）

---

## 6. Fine-tuning 微调方法

### 6.1 微调范式

```
预训练 BERT → 添加任务层 → 端到端微调
```

特点：
- 无需修改模型架构
- 所有参数联合训练
- 少量数据即可取得好效果

### 6.2 常见任务适配

#### 句子分类（如情感分析）

```
[CLS] 输入句子 [SEP] → BERT → [CLS]_output → 分类层 → 标签
```

#### 句子对分类（如问答）

```
[CLS] 问题 [SEP] 答案 [SEP] → BERT → [CLS]_output → 分类层 → 标签
```

#### Token 分类（如 NER）

```
[CLS] 词 1 词 2 ... [SEP] → BERT → 每 token 输出 → 分类层 → 标签序列
```

#### 问答（SQuAD）

```
[CLS] 问题 [SEP] 段落 [SEP] → BERT → 起始/结束位置预测
```

### 6.3 微调超参数

| 任务 | Batch Size | 学习率 | Epochs |
|------|-----------|--------|--------|
| GLUE | 16-32 | 2e-5 - 5e-5 | 3-4 |
| SQuAD | 32 | 3e-5 - 5e-5 | 3-4 |
| NER | 16-32 | 2e-5 - 5e-5 | 3-4 |

### 6.4 技巧

- 使用线性学习率衰减
- 梯度裁剪（max_grad_norm=1.0）
-  warmup 步数（10% 总步数）

---

## 7. BERT 变体

### 7.1 RoBERTa（2019）

Facebook 提出，优化 BERT 训练策略：

- **移除 NSP**：发现 NSP 无益甚至有害
- **动态掩码**：每次 epoch 重新生成掩码
- **更大 batch**：8K 样本
- **更多数据**：160GB 文本
- **更大词表**：50K BPE

性能提升：GLUE 从 80.5 → 88.5

### 7.2 ALBERT（2019）

Google 提出，参数高效的 BERT：

- **词表分解**：嵌入矩阵分解（30K×128 + 128×768）
- **参数共享**：所有层共享参数
- **SOP 任务**：替代 NSP，预测句子顺序

效果：参数量减少 18 倍，性能相当或更好

### 7.3 DistilBERT（2019）

Hugging Face 提出，知识蒸馏：

- **学生模型**：6 层（教师 12 层）
- **蒸馏损失**：隐藏层 + 注意力 + 预测
- **速度提升**：60% 更快
- **性能保持**：97% BERT 性能

### 7.4 其他变体

| 模型 | 特点 |
|------|------|
| **XLNet** | 自回归 + 自编码，排列语言模型 |
| **ELECTRA** | 替换 Token 检测，更高效 |
| **DeBERTa** | 解耦注意力，增强表示 |
| **Chinese-BERT-wwm** | 全词掩码，适合中文 |

---

## 8. 代码实现示例

详细代码实现请参考 `code/bert.py`，包含：

1. BERT Embedding 层
2. BERT Encoder 层
3. MLM 预测头
4. NSP 预测头
5. 完整 BERT 模型
6. 预训练示例
7. 微调示例

---

## 9. 参考文献

1. **BERT 原论文**
   - Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.
   - https://arxiv.org/abs/1810.04805

2. **RoBERTa**
   - Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." arXiv:1907.11692.

3. **ALBERT**
   - Lan, Z., et al. (2019). "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations." ICLR 2020.

4. **DistilBERT**
   - Sanh, V., et al. (2019). "DistilBERT, a distilled version of BERT." NeurIPS 2019.

5. **Transformer**
   - Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS 2017.

6. **Hugging Face Transformers**
   - https://huggingface.co/transformers/

7. **BERT 官方代码**
   - https://github.com/google-research/bert

---

**本章小结**：

BERT 通过双向编码和预训练 - 微调范式，彻底改变了 NLP 领域。理解 BERT 的核心思想（双向注意力、MLM、NSP）对于掌握现代语言模型至关重要。后续章节将在此基础上探索更先进的模型架构。
