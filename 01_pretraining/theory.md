# 第一章：预训练语言模型

**从 Word2Vec 到 BERT 的演进**

---

## 📚 目录

1. [语言模型基础](#语言模型基础)
2. [Word2Vec 与词嵌入](#word2vec-与词嵌入)
3. [RNN 与 LSTM](#rnn-与-lstm)
4. [预训练范式](#预训练范式)
5. [代码实战](#代码实战)

---

## 语言模型基础

### 什么是语言模型

语言模型 (Language Model, LM) 的目标是计算一个句子出现的概率：

$$P(w_1, w_2, ..., w_n) = \prod_{i=1}^{n} P(w_i | w_1, w_2, ..., w_{i-1})$$

### N-gram 模型

最简单的语言模型，只考虑前 N-1 个词：

$$P(w_i | w_1, ..., w_{i-1}) \approx P(w_i | w_{i-N+1}, ..., w_{i-1})$$

**问题**:
- 数据稀疏
- 无法捕捉长距离依赖
- 维度灾难

---

## Word2Vec 与词嵌入

### 核心思想

将词映射到低维连续向量空间，语义相似的词向量距离近。

### 两种架构

#### 1. CBOW (Continuous Bag of Words)

根据上下文预测中心词：

```
上下文：[今天，天气，很，好] → 预测：真
```

#### 2. Skip-gram

根据中心词预测上下文：

```
中心词：真 → 预测：[今天，天气，很，好]
```

### 代码实现

```python
import torch
import torch.nn as nn

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.W1 = nn.Embedding(vocab_size, embed_dim)  # 输入权重
        self.W2 = nn.Linear(embed_dim, vocab_size)     # 输出权重
    
    def forward(self, x):
        # x: [batch_size, context_size]
        embed = self.W1(x)  # [batch_size, context_size, embed_dim]
        embed = embed.mean(dim=1)  # [batch_size, embed_dim]
        out = self.W2(embed)  # [batch_size, vocab_size]
        return out
```

### 训练

```python
model = Word2Vec(vocab_size=10000, embed_dim=300)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练循环
for context, target in dataloader:
    optimizer.zero_grad()
    output = model(context)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

---

## RNN 与 LSTM

### RNN (循环神经网络)

处理序列数据，隐藏状态传递信息：

$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$$

**问题**: 梯度消失，无法捕捉长距离依赖

### LSTM (长短期记忆网络)

引入门控机制：

```python
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.i2f = nn.Linear(input_size, hidden_size)
        self.h2f = nn.Linear(hidden_size, hidden_size)
        self.i2i = nn.Linear(input_size, hidden_size)
        self.h2i = nn.Linear(hidden_size, hidden_size)
        self.i2g = nn.Linear(input_size, hidden_size)
        self.h2g = nn.Linear(hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, h_prev, c_prev):
        f_t = torch.sigmoid(self.i2f(x) + self.h2f(h_prev))  # 遗忘门
        i_t = torch.sigmoid(self.i2i(x) + self.h2i(h_prev))  # 输入门
        g_t = torch.tanh(self.i2g(x) + self.h2g(h_prev))     # 候选记忆
        o_t = torch.sigmoid(self.i2o(x) + self.h2o(h_prev))  # 输出门
        
        c_t = f_t * c_prev + i_t * g_t  # 更新细胞状态
        h_t = o_t * torch.tanh(c_t)     # 更新隐藏状态
        
        return h_t, c_t
```

---

## 预训练范式

### 为什么需要预训练

1. **数据利用**: 利用海量无标注文本
2. **迁移学习**: 预训练 + 微调模式
3. **泛化能力**: 学习通用语言表示

### 预训练任务

#### 1. 语言建模 (LM)

$$\mathcal{L} = -\sum_{i=1}^{n} \log P(w_i | w_{<i})$$

#### 2. 掩码语言建模 (MLM)

随机 mask 部分词，让模型预测：

```
输入：今天 [MASK] 很好
预测：天气
```

#### 3. 下一句预测 (NSP)

判断两个句子是否连续：

```
句子 A: 今天天气很好
句子 B: 我们去公园玩吧
标签：IsNext
```

---

## 代码实战

### 完整的预训练语言模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PretrainingLM(nn.Module):
    """简单的预训练语言模型"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        # x: [batch_size, seq_len]
        embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        out, hidden = self.lstm(embed, hidden)  # [batch_size, seq_len, hidden_dim]
        logits = self.fc(out)  # [batch_size, seq_len, vocab_size]
        return logits, hidden
    
    def pretrain(self, dataloader, epochs=10):
        """预训练"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x, y = batch  # y 是 x 右移一位
                optimizer.zero_grad()
                
                logits, _ = self.forward(x)
                # 计算 MLM loss
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
```

### 使用示例

```python
# 配置
vocab_size = 30000
embed_dim = 512
hidden_dim = 512
num_layers = 4

# 创建模型
model = PretrainingLM(vocab_size, embed_dim, hidden_dim, num_layers)

# 预训练
model.pretrain(train_dataloader, epochs=10)

# 保存预训练权重
torch.save(model.state_dict(), 'pretrained_lm.pth')

# 微调时加载
model.load_state_dict(torch.load('pretrained_lm.pth'))
```

---

## 📊 关键要点

| 模型 | 年份 | 核心创新 | 局限性 |
|------|------|----------|--------|
| Word2Vec | 2013 | 词向量 | 无法处理上下文 |
| RNN | 1986 | 序列建模 | 梯度消失 |
| LSTM | 1997 | 门控机制 | 并行化困难 |
| Pretraining LM | 2018+ | 预训练 + 微调 | 计算资源需求大 |

---

## 🔗 参考文献

1. Mikolov T, et al. "Distributed Representations of Words and Phrases and their Compositionality." NIPS 2013.
2. Hochreiter S, Schmidhuber J. "Long Short-Term Memory." Neural Computation 1997.
3. Devlin J, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.

---

## 💻 实战练习

1. 实现 Word2Vec 的 CBOW 和 Skip-gram
2. 用 LSTM 实现语言模型
3. 在 WikiText-2 数据集上预训练
4. 在下游任务（如情感分析）上微调

---

*下一章：[Transformer 架构](02_transformer/)*
