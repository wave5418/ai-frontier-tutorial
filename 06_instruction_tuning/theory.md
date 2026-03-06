# 第 06 章 指令微调与对齐 (Instruction Tuning & Alignment)

## 1. 指令微调背景与动机

### 1.1 什么是指令微调？

指令微调（Instruction Tuning）是一种让大型语言模型更好地理解和遵循人类指令的微调技术。通过在指令 - 响应对数据上进行监督微调（Supervised Fine-Tuning, SFT），模型能够学会：

- 理解各种类型的任务指令
- 按照指定的格式输出结果
- 在多轮对话中保持一致性
- 遵循人类的偏好和约束

### 1.2 为什么需要指令微调？

**预训练模型的局限性：**

1. **任务理解能力有限**：预训练模型主要通过 next-token prediction 学习语言模式，但不理解"任务"的概念
2. **输出格式不可控**：难以保证模型按照特定格式输出
3. **缺乏对话能力**：预训练是单向的，不具备多轮对话的交互能力
4. **对齐问题**：模型输出可能与人类价值观和偏好不一致

**指令微调带来的改进：**

- 提升 zero-shot 和 few-shot 性能
- 增强任务泛化能力
- 改善对话质量和一致性
- 为后续 RLHF 对齐打下基础

### 1.3 指令微调的发展历史

```
2021: Cross-Task Generalization (Sanh et al.)
      ↓
2022: FLAN (Fine-tuned LAnguage Net) - Google
      ↓
2022: InstructGPT - OpenAI
      ↓
2023: Alpaca - Stanford
      ↓
2023: Vicuna, Koala, Baize 等开源模型
      ↓
2023-2024: 各类指令微调模型爆发式增长
```

---

## 2. 指令数据格式与构建

### 2.1 标准指令数据格式

常见的指令数据格式包含以下字段：

```json
{
  "instruction": "任务指令",
  "input": "可选的输入内容",
  "output": "期望的输出",
  "history": [["轮次 1 用户", "轮次 1 助手"], ...],  // 多轮对话历史
  "system": "系统提示词"  // 可选
}
```

### 2.2 指令数据类型

| 类型 | 描述 | 示例 |
|------|------|------|
| 开放式问答 | 开放性问题回答 | "解释量子力学的基本原理" |
| 封闭式问答 | 有确定答案的问题 | "法国的首都是哪里？" |
| 文本生成 | 创作类任务 | "写一首关于春天的诗" |
| 代码生成 | 编程相关任务 | "用 Python 实现快速排序" |
| 翻译 | 语言间转换 | "将这句话翻译成法语" |
| 摘要 | 文本压缩 | "总结这篇文章的主要观点" |
| 分类 | 文本分类任务 | "判断这条评论的情感倾向" |
| 抽取 | 信息抽取 | "从文中提取所有人名" |

### 2.3 指令数据构建方法

**1. 人工标注**
- 高质量但成本高
- 适用于特定领域
- 示例：OpenAI 的 InstructGPT 数据

**2. 自指令（Self-Instruct）**
- 使用模型自身生成指令数据
- 迭代式扩展
- 示例：Alpaca 数据集构建方法

**3. 现有数据集转换**
- 将 NLP 任务数据集转换为指令格式
- 示例：将 SQuAD 转换为问答指令

**4. 合成数据生成**
- 使用大模型生成训练数据
- 需要质量控制

### 2.4 数据质量要求

- **多样性**：覆盖多种任务类型和领域
- **准确性**：输出内容正确可靠
- **一致性**：格式统一，风格一致
- **安全性**：过滤有害内容

---

## 3. Supervised Fine-Tuning (SFT)

### 3.1 SFT 基本原理

监督微调是在预训练模型基础上，使用标注数据进行有监督训练的过程。

**目标函数：**

$$\mathcal{L}(\theta) = -\sum_{i=1}^{N} \log P(y_i | x_i, \theta)$$

其中：
- $x_i$：输入指令
- $y_i$：期望输出
- $\theta$：模型参数

### 3.2 SFT 训练流程

```
1. 加载预训练模型
       ↓
2. 准备指令数据集
       ↓
3. 数据预处理和 Tokenization
       ↓
4. 定义损失函数和优化器
       ↓
5. 训练循环
       ↓
6. 模型评估和保存
```

### 3.3 关键训练技巧

**1. 学习率调度**
```python
# 常用策略
- Cosine Decay
- Linear Warmup + Decay
- Constant with Warmup
```

**2. 梯度累积**
- 在显存有限时模拟大 batch size
- 每 N 步更新一次参数

**3. 混合精度训练**
- 使用 FP16/BF16 加速训练
- 减少显存占用

**4. 序列打包（Sequence Packing）**
- 将多个短样本拼接成长序列
- 提高训练效率

### 3.4 训练超参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 学习率 | 1e-5 ~ 2e-5 | 较小学习率避免灾难性遗忘 |
| Batch Size | 16-64 | 根据显存调整 |
| Epochs | 2-5 | 过多容易过拟合 |
| Max Length | 512-2048 | 根据任务需求 |
| Warmup Ratio | 0.03-0.1 | 训练初期稳定 |

---

## 4. 指令跟随能力涌现

### 4.1 什么是涌现能力？

涌现能力（Emergent Abilities）指模型在规模达到一定程度后突然展现出的能力，这些能力在小模型中不存在或很弱。

### 4.2 指令跟随的涌现现象

**研究发现：**

1. **规模阈值**：约 60B 参数以下模型指令跟随能力提升缓慢
2. **多任务训练**：跨任务微调促进能力迁移
3. **指令多样性**：数据多样性比数量更重要

### 4.3 评估指标

| 指标 | 描述 |
|------|------|
| Helpfulness | 回答是否有用 |
| Honesty | 信息是否真实 |
| Harmlessness | 内容是否安全 |
| Following Rate | 指令遵循程度 |

### 4.4 能力提升策略

1. **多任务联合训练**：同时训练多种任务类型
2. **课程学习**：从简单到复杂逐步训练
3. **数据增强**：对指令进行改写和扩展
4. **对比学习**：区分好坏回答

---

## 5. Chat 格式与多轮对话

### 5.1 对话格式设计

**常见的对话模板：**

**格式 1 - Alpaca 风格：**
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

**格式 2 - ChatML 风格：**
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_message}<|im_end|>
```

**格式 3 - Llama 风格：**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{assistant_message}<|eot_id|>
```

### 5.2 多轮对话处理

**关键挑战：**

1. **上下文长度限制**：需要管理对话历史
2. **角色一致性**：保持助手人设稳定
3. **指代消解**：理解代词和省略
4. **长期依赖**：记住早期对话内容

**解决方案：**

```python
# 对话历史管理策略
1. 滑动窗口：保留最近 N 轮
2. 摘要压缩：用摘要替代早期对话
3. 关键信息提取：抽取重要事实存储
4. 分层注意力：对不同轮次赋予不同权重
```

### 5.3 系统提示词设计

系统提示词定义助手的行为准则：

```
你是一个有帮助的 AI 助手。请：
1. 提供准确、有用的信息
2. 保持友好、专业的语气
3. 承认知识的局限性
4. 拒绝回答有害问题
5. 用中文回答，除非用户要求其他语言
```

### 5.4 对话质量评估

- **连贯性**：回答是否与上下文相关
- **一致性**：是否自相矛盾
- **信息量**：是否提供足够信息
- **安全性**：是否包含有害内容

---

## 6. 代码实现示例

### 6.1 数据集类实现

```python
class InstructionDataset(Dataset):
    """指令微调数据集类"""
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = load_json(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # 构建对话文本
        text = self.format_conversation(item)
        # Tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0).clone()
        }
```

### 6.2 SFT 训练代码

```python
def train_sft(model, train_loader, optimizer, scheduler, device, epochs=3):
    """监督微调训练循环"""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
```

### 6.3 多轮对话示例

```python
def generate_response(model, tokenizer, conversation_history, system_prompt=""):
    """生成多轮对话回复"""
    # 构建完整对话
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(conversation_history)
    
    # 应用对话模板
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # 生成回复
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
```

---

## 7. 参考文献

1. **Wei, J. et al. (2022).** "Finetuned Language Models are Zero-Shot Learners" (FLAN). arXiv:2109.01652

2. **Ouyang, L. et al. (2022).** "Training language models to follow instructions with human feedback" (InstructGPT). arXiv:2203.02155

3. **Taori, R. et al. (2023).** "Alpaca: A Strong, Replicable Instruction-Following Model". Stanford.

4. **Chiang, W. et al. (2023).** "Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality".

5. **Wang, Y. et al. (2022).** "Self-Instruct: Aligning Language Model with Self Generated Instructions". arXiv:2212.10560

6. **Touvron, H. et al. (2023).** "LLaMA: Open and Efficient Foundation Language Models". arXiv:2302.13971

7. **Bai, Y. et al. (2022).** "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback". arXiv:2204.05862

8. **Longpre, S. et al. (2023).** "The Flan Collection: Designing Data and Methods for Effective Instruction Tuning". ICML.

---

*本章完 - 下一章：高效微调技术 (PEFT)*
