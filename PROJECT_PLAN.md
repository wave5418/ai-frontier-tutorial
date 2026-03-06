# AI 前沿技术教程 - 项目梳理

**创建时间**: 2026-03-06  
**GitHub**: https://github.com/wave5418/ai-frontier-tutorial

---

## 🎯 项目目标

建立一条完整的 AI 前沿技术学习路线，从基础理论到代码实现：

```
Pretraining → LLM → 多模态 → 世界模型 → VLA → Agent → RL → AGI
```

---

## 📚 已完成内容

### ✅ 仓库创建

- [x] GitHub 仓库：https://github.com/wave5418/ai-frontier-tutorial
- [x] README.md - 完整技术路线图
- [x] requirements.txt - 依赖配置
- [x] .gitignore - Git 配置
- [x] 第一章完整内容（理论 + 代码）

### ✅ 第一章：预训练语言模型

**理论部分** (`01_pretraining/theory.md`):
- 语言模型基础（N-gram、概率公式）
- Word2Vec 与词嵌入（CBOW、Skip-gram）
- RNN 与 LSTM（门控机制、公式推导）
- 预训练范式（LM、MLM、NSP）
- 代码实战示例

**代码部分** (`01_pretraining/code/word2vec_lstm.py`):
- Word2Vec 完整实现
- CBOW 模型实现
- LSTM 语言模型
- 预训练语言模型类
- 训练函数
- 使用示例

---

## 📋 待完成章节

### 第二阶段：Transformer 与 LLM

| 章节 | 主题 | 预计完成 |
|------|------|----------|
| 02 | Transformer 架构 | Week 2 |
| 03 | BERT 与双向编码 | Week 2 |
| 04 | GPT 与自回归生成 | Week 2 |
| 05 | LLM 架构演进 | Week 3 |
| 06 | 指令微调与对齐 | Week 3 |
| 07 | RLHF 与人类对齐 | Week 4 |
| 08 | 高效微调技术 | Week 4 |

### 第三阶段：多模态模型

| 章节 | 主题 | 预计完成 |
|------|------|----------|
| 09 | CLIP 与图文对比学习 | Week 5 |
| 10 | 扩散模型与 Stable Diffusion | Week 5 |
| 11 | 多模态大模型 (LMM) | Week 6 |
| 12 | 视觉语言模型 (VLM) | Week 6 |

### 第四阶段：世界模型

| 章节 | 主题 | 预计完成 |
|------|------|----------|
| 13 | 世界模型基础 | Week 7 |
| 14 | 视频预测 | Week 7 |
| 15 | Genie 与环境建模 | Week 8 |

### 第五阶段：VLA

| 章节 | 主题 | 预计完成 |
|------|------|----------|
| 16 | VLA 基础架构 | Week 9 |
| 17 | RT-2 与机器人学习 | Week 9 |
| 18 | OpenVLA 与开源方案 | Week 10 |

### 第六阶段：Agent

| 章节 | 主题 | 预计完成 |
|------|------|----------|
| 19 | Agent 基础架构 | Week 11 |
| 20 | ReAct 与推理行动 | Week 11 |
| 21 | 工具使用与 Function Calling | Week 12 |
| 22 | 多 Agent 协作 | Week 12 |

### 第七阶段：强化学习

| 章节 | 主题 | 预计完成 |
|------|------|----------|
| 23 | RL 基础 | Week 13 |
| 24 | PPO 与策略梯度 | Week 13 |
| 25 | DPO 与直接偏好优化 | Week 14 |
| 26 | RLHF 实战 | Week 14 |

### 第八阶段：AGI 展望

| 章节 | 主题 | 预计完成 |
|------|------|----------|
| 27 | AGI 路径与挑战 | Week 15 |
| 28 | Scaling Law 与未来 | Week 15 |
| 29 | 具身智能与机器人 | Week 16 |
| 30 | 总结与展望 | Week 16 |

---

## 🛠️ 技术栈

### 深度学习框架
- PyTorch 2.0+
- Transformers 4.30+
- Accelerate (可选)

### 数据处理
- NumPy
- Pandas
- Tokenizers

### 可视化
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📖 学习路径

### 零基础路线
```
01 预训练 → 02 Transformer → 03 BERT → 04 GPT → 05 LLM
```

### 进阶路线
```
05 LLM → 06 指令微调 → 07 RLHF → 19 Agent → 23 RL
```

### 专家路线
```
09 CLIP → 10 扩散模型 → 13 世界模型 → 16 VLA → 27 AGI
```

---

## 🚀 下一步计划

### Week 1-2: 基础夯实
- [ ] 完成 02 Transformer 架构
  - Self-Attention 详解
  - Multi-Head Attention
  - Position Encoding
  - 代码实现（从 0 到 1）

- [ ] 完成 03 BERT
  - 双向编码
  - MLM 任务
  - NSP 任务
  - 微调实战

- [ ] 完成 04 GPT
  - 自回归生成
  - Causal Attention
  - 文本生成
  - 温度采样

### Week 3-4: LLM 核心
- [ ] 完成 05-08 章
- [ ] 添加更多代码示例
- [ ] 提供 Colab 笔记本

---

## 📊 项目结构

```
ai-frontier-tutorial/
├── README.md                    # 总览
├── requirements.txt             # 依赖
├── .gitignore                   # Git 配置
│
├── 01_pretraining/              # 第一章：预训练
│   ├── theory.md                # 理论
│   └── code/
│       └── word2vec_lstm.py     # 代码
│
├── 02_transformer/              # 第二章：Transformer
│   ├── theory.md                # (待写)
│   └── code/
│       └── transformer.py       # (待写)
│
├── 03_bert/                     # 第三章：BERT
├── 04_gpt/                      # 第四章：GPT
├── 05_llm_architecture/         # 第五章：LLM 架构
├── ...
└── 30_conclusion/               # 第三十章：总结
```

---

## 🎓 学习建议

1. **理论 + 代码**: 每章都要动手实现
2. **循序渐进**: 按顺序学习，不要跳章
3. **实践项目**: 每章完成后做小项目
4. **讨论交流**: 提 Issue 讨论问题

---

## 📝 更新日志

### 2026-03-06
- ✅ 创建 GitHub 仓库
- ✅ 完成 README 和技术路线
- ✅ 完成第一章（预训练语言模型）
- ✅ 提供完整代码实现
- ✅ 推送到 GitHub

---

*持续更新中... 预计 16 周完成全部内容*
