# 第 04 章 GPT 与自回归生成

## 1. GPT 系列发展

### GPT-1 (2018)
- **参数量**: 1.17 亿
- **训练数据**: BookCorpus (约 7000 本未出版书籍)
- **核心贡献**: 首次将 Transformer 解码器结构用于语言模型预训练
- **架构**: 12 层 Transformer 解码器，12 个注意力头，768 维隐藏层
- **训练目标**: 标准语言建模（预测下一个词）

### GPT-2 (2019)
- **参数量**: 15 亿（最大版本）
- **训练数据**: WebText (40GB 互联网文本)
- **核心贡献**: 证明大规模预训练可实现零样本迁移
- **关键改进**: 
  - 更大的模型和数据集
  - 修改的初始化方式
  - 词汇表扩大到 50257 个 BPE token

### GPT-3 (2020)
- **参数量**: 1750 亿
- **训练数据**: Common Crawl + WebText + Books + Wikipedia
- **核心贡献**: In-context Learning（上下文学习）
- **突破性能力**:
  - 少样本学习（Few-shot Learning）
  - 零样本推理（Zero-shot Reasoning）
  - 代码生成

### GPT-4 (2023)
- **参数量**: 未公开（估计万亿级）
- **核心突破**:
  - 多模态理解（图像 + 文本）
  - 更强的推理能力
  - 更长的上下文窗口（128K tokens）
  - 更高的对齐度和安全性

---

## 2. 自回归语言模型原理

### 基本概念
自回归（Autoregressive）模型通过**序列分解**建模联合概率：

$$P(x_1, x_2, ..., x_n) = \prod_{i=1}^{n} P(x_i | x_1, x_2, ..., x_{i-1})$$

### 核心思想
1. **从左到右生成**: 每次只预测下一个 token
2. **条件概率**: 基于已生成的所有历史 token
3. **链式法则**: 将复杂联合分布分解为简单条件概率的乘积

### 数学表达
对于输入序列 $X = (x_1, x_2, ..., x_n)$，语言模型的目标是最大化：

$$\mathcal{L} = \sum_{i=1}^{n} \log P(x_i | x_{<i})$$

其中 $x_{<i}$ 表示位置 $i$ 之前的所有 token。

---

## 3. Causal Attention（因果注意力）

### 定义
因果注意力（也称 Masked Self-Attention）确保每个位置只能关注**它自己及之前的位置**，不能看到未来信息。

### 注意力掩码
```
位置:    0  1  2  3  4
        ┌─────────────
     0  │ 0 -∞ -∞ -∞ -∞
     1  │ 0  0 -∞ -∞ -∞
     2  │ 0  0  0 -∞ -∞
     3  │ 0  0  0  0 -∞
     4  │ 0  0  0  0  0
```
- `0`: 允许关注（权重正常计算）
- `-∞`: 禁止关注（softmax 后权重为 0）

### 实现方式
```python
# 创建因果掩码
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * -1e9
# attention_scores = Q @ K^T / sqrt(d_k) + causal_mask
# attention_weights = softmax(attention_scores, dim=-1)
```

### 为什么需要因果注意力？
1. **训练时**: 防止模型"偷看"未来，确保预测基于真实历史
2. **推理时**: 保持自回归性质，逐个生成 token
3. **与 BERT 对比**: BERT 使用双向注意力，适合编码任务；GPT 使用单向注意力，适合生成任务

---

## 4. GPT 架构详解

### 整体架构
```
输入 Token → Embedding → [GPT Block × N] → LayerNorm → 输出投影 → 概率分布
                                    ↑
                              位置编码（可学习）
```

### 核心组件

#### 4.1 Token Embedding
- 将离散 token 映射为连续向量
- 维度：`vocab_size × d_model`（如 50257 × 768）

#### 4.2 位置编码（Position Embedding）
- GPT 使用**可学习的位置嵌入**（非 Transformer 原版的正弦编码）
- 维度：`max_seq_len × d_model`
- 与 token embedding 相加：`input = token_emb + pos_emb`

#### 4.3 GPT Block（ decoder 层）
每个 Block 包含两个子层：
1. **Masked Multi-Head Self-Attention**
2. **Position-wise Feed-Forward Network (FFN)**

每个子层后接：
- **残差连接**（Residual Connection）
- **层归一化**（Layer Normalization）

#### 4.4 多头注意力（Multi-Head Attention）
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W^O
其中 head_i = Attention(Q × W_i^Q, K × W_i^K, V × W_i^V)
```

- **并行计算**: 多个注意力头同时计算
- **不同子空间**: 每个头学习不同的表示子空间
- **典型配置**: 12 头（GPT-1/2-small），96 头（GPT-3）

#### 4.5 前馈网络（FFN）
```python
FFN(x) = GELU(x × W_1 + b_1) × W_2 + b_2
```
- 第一层：扩展维度（d_model → 4×d_model）
- 激活函数：GELU（高斯误差线性单元）
- 第二层：投影回原维度（4×d_model → d_model）

#### 4.6 输出层
- **LayerNorm**: 最终归一化
- **线性投影**: 将隐藏状态映射到词汇表大小
- **Softmax**: 转换为概率分布

---

## 5. 文本生成策略

### 5.1 Greedy Search（贪婪搜索）
**策略**: 每一步选择概率最高的 token

```python
next_token = argmax(P(token | context))
```

**优点**:
- 简单高效
- 确定性输出

**缺点**:
- 容易陷入重复循环
- 缺乏多样性
- 可能错过全局最优序列

### 5.2 Beam Search（束搜索）
**策略**: 维护 k 个候选序列（束宽），每步扩展并保留 top-k

**算法流程**:
1. 初始化：从 `<start>` 开始，束宽为 k
2. 扩展：对每个候选序列，计算所有可能的下一个 token
3. 评分：计算序列累积概率（或平均对数概率）
4. 剪枝：保留 top-k 序列
5. 终止：当 k 个序列都生成 `<end>` 或达到最大长度

**优点**:
- 比贪婪搜索更全局
- 适合翻译、摘要等任务

**缺点**:
- 计算开销大（k 倍）
- 仍可能缺乏多样性
- 不适合开放域对话

### 5.3 Sampling（随机采样）
**策略**: 从概率分布中随机采样

```python
next_token = sample(P(token | context))
```

**优点**:
- 生成多样化
- 避免重复
- 更自然的人类风格

**缺点**:
- 可能生成低质量内容
- 不可控
- 需要配合其他策略（temperature, top-k, top-p）

---

## 6. Temperature 调节

### 原理
Temperature（温度）参数控制概率分布的"尖锐度"：

$$P_{new}(x) = \frac{\exp(\log P(x) / T)}{\sum_{x'} \exp(\log P(x') / T)}$$

### 效果
| Temperature | 效果 | 适用场景 |
|-------------|------|----------|
| T < 1 (如 0.5) | 分布更尖锐，高概率 token 更突出 | 事实性问答、代码生成 |
| T = 1 | 原始分布 | 默认设置 |
| T > 1 (如 1.5) | 分布更平滑，低概率 token 有机会 | 创意写作、头脑风暴 |
| T → 0 | 趋近贪婪搜索 | 确定性任务 |
| T → ∞ | 趋近均匀分布 | 几乎不用 |

### 实现
```python
def apply_temperature(logits, temperature):
    if temperature == 0:
        return greedy_search(logits)
    logits = logits / temperature
    probs = softmax(logits, dim=-1)
    return sample(probs)
```

---

## 7. Top-k / Top-p Sampling

### 7.1 Top-k Sampling
**策略**: 只从概率最高的 k 个 token 中采样

**步骤**:
1. 取 logits top-k 个 token
2. 将其他 token 概率设为 0
3. 重新归一化
4. 从剩余 token 中采样

```python
def top_k_sampling(logits, k, temperature=1.0):
    top_k_values, top_k_indices = torch.topk(logits, k)
    mask = torch.ones_like(logits) * -float('inf')
    mask.scatter_(1, top_k_indices, 0)
    logits = logits + mask
    probs = softmax(logits / temperature, dim=-1)
    return sample(probs)
```

**优点**: 简单，控制候选集大小
**缺点**: k 固定，不考虑分布形状

### 7.2 Top-p Sampling (Nucleus Sampling)
**策略**: 从累积概率达到 p 的最小 token 集合中采样

**步骤**:
1. 对 logits 降序排序
2. 计算累积概率
3. 找到累积概率 ≥ p 的最小集合
4. 重新归一化后采样

```python
def top_p_sampling(logits, p, temperature=1.0):
    probs = softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到截断点
    nucleus_mask = cumsum_probs <= p
    nucleus_mask[:, 1:] = nucleus_mask[:, :-1].clone()
    nucleus_mask[:, 0] = True
    
    # 还原原始顺序
    mask = torch.zeros_like(logits).scatter_(1, sorted_indices, nucleus_mask)
    probs = probs * mask
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return sample(probs)
```

**优点**: 自适应候选集大小，更灵活
**缺点**: 计算稍复杂

### 7.3 组合使用
实际应用中常组合使用：
```python
# 先 top-k，再 top-p
logits = apply_top_k(logits, k=50)
logits = apply_top_p(logits, p=0.9)
probs = apply_temperature(logits, temperature=0.8)
next_token = sample(probs)
```

---

## 8. In-context Learning（上下文学习）

### 定义
In-context Learning 指模型通过**输入示例**（而非参数更新）学习新任务的能力。

### 三种模式

#### 8.1 Zero-shot（零样本）
```
输入：翻译以下句子到法语："Hello, world!"
输出：Bonjour le monde!
```
- 仅提供任务描述
- 无示例

#### 8.2 One-shot（单样本）
```
输入：
英语：Good morning → 法语：Bonjour
英语：How are you? → 法语：

输出：Comment allez-vous?
```
- 提供 1 个示例
- 模型模仿示例格式

#### 8.3 Few-shot（少样本）
```
输入：
情感分析：
"这部电影太棒了！" → 正面
"剧情很无聊。" → 负面
"演员表现出色。" → 正面
"浪费时间。" → 

输出：负面
```
- 提供多个示例（通常 2-10 个）
- 模型从示例中归纳规律

### 原理分析
1. **模式匹配**: 模型识别输入中的任务模式
2. **隐式学习**: 通过注意力机制关注示例中的关键信息
3. **元学习**: 预训练过程中已学习"如何学习"的能力

### 关键因素
- **示例质量**: 清晰、一致的示例效果更好
- **示例顺序**: 有时会影响输出
- **示例数量**: 通常 3-5 个效果最佳
- **格式一致性**: 输入输出格式需统一

---

## 9. 代码实现示例

完整的 GPT 实现请参考 `code/gpt.py`，核心要点：

### 关键类
- `CausalSelfAttention`: 因果自注意力
- `GPTBlock`: GPT 解码器块
- `GPT`: 完整 GPT 模型
- `generate()`: 文本生成函数

### 训练流程
1. 准备文本数据，tokenize
2. 构建训练批次（输入 + 目标）
3. 前向传播，计算损失
4. 反向传播，更新参数
5. 重复直到收敛

### 生成流程
1. 编码输入 prompt
2. 迭代生成 next token
3. 应用采样策略（temperature, top-k, top-p）
4. 解码 token 为文本
5. 直到生成 `<eos>` 或达到最大长度

---

## 10. 参考文献

### 核心论文
1. **GPT-1**: Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training". OpenAI.

2. **GPT-2**: Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners". OpenAI.

3. **GPT-3**: Brown, T., et al. (2020). "Language Models are Few-Shot Learners". NeurIPS 2020.

4. **GPT-4**: OpenAI (2023). "GPT-4 Technical Report".

5. **Transformer**: Vaswani, A., et al. (2017). "Attention Is All You Need". NeurIPS 2017.

### 相关技术
6. **BPE Tokenization**: Sennrich, R., et al. (2016). "Neural Machine Translation of Rare Words with Subword Units".

7. **LayerNorm**: Ba, J. L., et al. (2016). "Layer Normalization".

8. **GELU Activation**: Hendrycks, D., & Gimpel, K. (2016). "Gaussian Error Linear Units (GELUs)".

9. **Nucleus Sampling**: Holtzman, A., et al. (2020). "The Curious Case of Neural Text Degeneration". ICLR 2020.

### 学习资源
10. **The Annotated Transformer**: http://nlp.seas.harvard.edu/2018/04/03/attention.html

11. **nanoGPT**: https://github.com/karpathy/nanoGPT (Andrej Karpathy 的极简 GPT 实现)

12. **Hugging Face Transformers**: https://huggingface.co/docs/transformers

---

*本章完*
