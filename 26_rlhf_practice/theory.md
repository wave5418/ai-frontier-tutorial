# 第 26 章：RLHF 实战

## 1. RLHF 完整流程回顾

### 1.1 什么是 RLHF？

**RLHF（Reinforcement Learning from Human Feedback，人类反馈强化学习）** 是一种将人类偏好融入大语言模型训练的方法。它是 ChatGPT、Claude 等对齐模型成功的关键技术。

### 1.2 RLHF 三阶段流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      RLHF 完整流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  阶段 1: SFT (Supervised Fine-Tuning)                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 预训练模型 + 指令数据集 → SFT 模型                        │   │
│  │ 目标：让模型学会遵循指令                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  阶段 2: Reward Modeling                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ SFT 模型 + 偏好数据集 → Reward Model                     │   │
│  │ 目标：学习人类偏好，预测回复质量得分                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  阶段 3: PPO Fine-Tuning                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ SFT 模型 + Reward Model → 最终对齐模型                    │   │
│  │ 目标：用强化学习优化策略，最大化奖励同时保持多样性         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 各阶段详解

| 阶段 | 输入 | 输出 | 目标 | 数据需求 |
|------|------|------|------|----------|
| **SFT** | 预训练模型 + (prompt, response) 对 | SFT 模型 | 学会遵循指令 | 10k-100k 指令样本 |
| **Reward** | SFT 模型 + (prompt, response_w, response_l) 偏好对 | Reward Model | 预测人类偏好 | 50k-200k 偏好标注 |
| **PPO** | SFT 模型 + Reward Model | 最终对齐模型 | 最大化奖励 | 无需新数据，用 prompt 集 |

## 2. 数据收集与标注

### 2.1 SFT 数据收集

**数据来源：**
1. **人工编写**：高质量但成本高
2. **现有数据集**：Alpaca、Dolly、OpenOrca 等
3. **自生成**：用强模型生成，弱模型筛选
4. **众包**：Scale AI、Mechanical Turk 等

**数据格式：**
```json
{
  "instruction": "请解释量子力学的基本原理。",
  "input": "",
  "output": "量子力学是描述微观粒子行为的物理学理论...",
  "history": []  // 多轮对话历史（可选）
}
```

**质量要求：**
- 指令多样性（覆盖多种任务类型）
- 回复质量高（准确、有用、无害）
- 长度适中（避免过短或冗长）
- 格式规范（便于处理）

### 2.2 偏好数据标注

**标注流程：**

```
1. 给定 prompt，用 SFT 模型生成多个候选回复（通常 2-4 个）
2. 人工标注员比较回复，选择最佳回复
3. 可选：标注拒绝回复的问题（不安全、无用等）
4. 质量检查：多标注员一致性验证
```

**标注界面示例：**
```
Prompt: 如何学习编程？

回复 A: 多写代码就行了，熟能生巧。
回复 B: 学习编程的建议：1) 选择一门入门语言如 Python；
       2) 理解基本概念；3) 通过项目实践；4) 阅读他人代码。

请选择更好的回复：○ A  ○ B  ○ 一样差  ○ 无法判断
```

**标注指南要点：**
- **有用性**：回复是否解决了用户问题
- **准确性**：信息是否正确
- **安全性**：是否包含有害内容
- **简洁性**：是否冗长或过于简略
- **格式**：是否清晰易读

### 2.3 数据预处理

```python
# 数据清洗步骤
def preprocess_data(raw_data):
    # 1. 去重
    data = deduplicate(raw_data)
    
    # 2. 过滤低质量样本
    data = filter_by_length(data, min_len=20, max_len=2048)
    data = filter_by_quality_score(data, threshold=0.7)
    
    # 3. 标准化格式
    data = normalize_format(data)
    
    # 4. 划分训练/验证集
    train_data, val_data = train_test_split(data, test_size=0.1)
    
    return train_data, val_data
```

## 3. SFT 阶段实践

### 3.1 模型选择

**推荐基座模型：**
- **小模型（<1B）**：Qwen2.5-0.5B, TinyLlama-1.1B
- **中等模型（1-7B）**：Qwen2.5-1.5B/3B/7B, LLaMA-2-7B, Mistral-7B
- **大模型（>13B）**：Qwen2.5-14B/32B, LLaMA-2-13B/70B

**选择考虑：**
- 任务复杂度
- 可用计算资源
- 延迟要求
- 部署环境

### 3.2 SFT 训练配置

```yaml
# SFT 训练配置示例
model_name: "Qwen/Qwen2.5-7B-Instruct"
output_dir: "./sft_output"

# 数据
train_file: "data/sft_train.jsonl"
eval_file: "data/sft_eval.jsonl"
max_seq_length: 1024

# 训练参数
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-5
num_train_epochs: 3
warmup_ratio: 0.1
lr_scheduler_type: "cosine"

# 优化
bf16: true
gradient_checkpointing: true
optim: "adamw_torch"

# 保存
save_strategy: "epoch"
save_total_limit: 3
```

### 3.3 训练技巧

**1. 学习率选择**
- 小模型：1e-4 - 5e-5
- 中等模型：2e-5 - 5e-5
- 大模型：1e-5 - 2e-5

**2. 批大小调整**
- 显存不足时使用梯度累积
- 有效批大小 = per_device_batch × num_devices × accumulation_steps
- 推荐有效批大小：32-128

**3. 序列长度**
- 短任务（问答）：512-1024
- 长任务（写作）：2048-4096
- 注意：更长序列 = 更多显存

**4. 防止过拟合**
- 使用验证集早停
- 添加 dropout（如果模型支持）
- 限制训练轮数（通常 2-3 轮足够）

### 3.4 SFT 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **训练损失** | 交叉熵损失 | 持续下降 |
| **验证损失** | 验证集交叉熵 | 不显著上升 |
| **Perplexity** | 困惑度 | 越低越好 |
| **人工评估** | 指令遵循质量 | 主观评分 |
| **基准测试** | MMLU、GSM8K 等 | 保持或提升 |

## 4. Reward Model 训练技巧

### 4.1 Reward Model 架构

**常见架构选择：**

1. **分类器式（Classifier-style）**
   ```
   [Base Model] → [Pooling] → [Linear] → [Scalar Score]
   ```
   - 使用基座模型 + 分类头
   - 输出单个标量分数

2. **Pairwise 式**
   ```
   [Base Model] 处理 (prompt, response) 对
   输出：P(response_w > response_l)
   ```
   - 直接建模偏好概率
   - 使用 Bradley-Terry 损失

3. **回归式**
   ```
   [Base Model] → [Regression Head] → [Continuous Score]
   ```
   - 输出连续分数（如 1-5 分）
   - 可用于绝对质量评估

### 4.2 训练配置

```yaml
# Reward Model 训练配置
base_model: "Qwen/Qwen2.5-7B"
output_dir: "./reward_model_output"

# 数据
train_file: "data/preference_train.jsonl"
eval_file: "data/preference_eval.jsonl"
max_seq_length: 512

# 训练参数
per_device_train_batch_size: 8
learning_rate: 1e-5
num_train_epochs: 2
warmup_ratio: 0.05

# 特殊设置
pooling: "mean"  # 或 "last", "cls"
loss_type: "pairwise"  # 或 "regression"
```

### 4.3 训练技巧

**1. 数据增强**
- 对同一 prompt 生成多个回复
- 创建更多偏好对
- 注意：避免引入噪声

**2. 处理类别不平衡**
- 如果偏好数据有类别偏差，使用加权损失
- 确保不同任务类型的样本均衡

**3. 防止过拟合**
- Reward Model 容易过拟合标注噪声
- 使用验证集早停
- 考虑 label smoothing

**4. 多 Reward Model 集成**
- 训练多个 Reward Model
- 取平均或投票作为最终分数
- 提高稳定性和泛化能力

### 4.4 Reward Model 评估

**评估方法：**

1. **准确率**
   - 在 held-out 偏好数据上的预测准确率
   - 目标：>70%（取决于任务难度）

2. **一致性检查**
   - 检查传递性：如果 A>B 且 B>C，则应该 A>C
   - 检查标注员间一致性

3. **相关性分析**
   - 与人工评分的相关性（Spearman/Pearson）
   - 目标：>0.6

4. **A/B 测试**
   - 用 Reward Model 指导的模型 vs 基线
   - 人工评估胜率

## 5. PPO 微调实践

### 5.1 PPO 算法回顾

**PPO（Proximal Policy Optimization）** 是一种策略梯度强化学习算法。

**核心思想：**
- 用当前策略生成样本
- 用 Reward Model 评估样本质量
- 更新策略以最大化期望奖励
- 用 KL 散度约束防止偏离太远

**PPO 损失函数：**
```
L_PPO = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)] - β * KL(π || π_ref)

其中：
- r_t = π_new(a|s) / π_old(a|s)  (概率比率)
- A_t 是优势函数估计
- ε 是 clip 范围（通常 0.2）
- β 是 KL 约束系数
```

### 5.2 PPO 训练配置

```yaml
# PPO 训练配置
policy_model: "./sft_output"  # SFT 模型
reward_model: "./reward_model_output"  # Reward Model
ref_model: "./sft_output"  # 参考模型（同 SFT）
value_model: "./sft_output"  # 价值模型（可复用）

# PPO 特定参数
ppo_epochs: 4  # 每批数据的 PPO 更新轮数
mini_batch_size: 128
batch_size: 1024  # 每次 rollout 的 prompt 数
gradient_accumulation_steps: 4

# 采样参数
top_k: 0
top_p: 0.9
temperature: 1.0
max_new_tokens: 256

# PPO 超参数
clip_range: 0.2
kl_coef: 0.2  # KL 约束系数
gamma: 1.0  # 折扣因子
lam: 0.95  # GAE 参数
value_coef: 0.1  # 价值损失系数
ent_coef: 0.0  # 熵正则化

# 训练
total_episodes: 10000
learning_rate: 1e-6
```

### 5.3 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                    PPO 训练循环                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  for episode in range(total_episodes):                      │
│                                                             │
│    1. Rollout 阶段                                         │
│       - 从 prompt 集采样一批 prompt                          │
│       - 用当前策略模型生成回复                              │
│       - 用 Reward Model 计算奖励                            │
│       - 计算优势函数（GAE）                                 │
│                                                             │
│    2. Update 阶段                                          │
│       - 对每批数据进行多轮 PPO 更新                          │
│       - 计算 PPO 损失 + KL 惩罚 + 价值损失                    │
│       - 反向传播，更新策略模型                              │
│                                                             │
│    3. 评估阶段（定期）                                     │
│       - 在验证 prompt 上生成回复                            │
│       - 计算平均奖励、KL 散度                                │
│       - 保存检查点                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 关键技巧

**1. KL 散度控制**
- KL 散度过大 → 模型偏离太多，可能产生胡言乱语
- KL 散度过小 → 优化不足，效果不明显
- 动态调整：根据 KL 散度自动调整 β

```python
# 自适应 KL 控制
if kl > kl_target * 2:
    kl_coef *= 1.5  # 增加约束
elif kl < kl_target / 2:
    kl_coef *= 0.5  # 减少约束
```

**2. 奖励标准化**
- Reward Model 输出可能尺度不稳定
- 每批数据内标准化奖励
- 或使用 running mean/std

**3. 价值函数训练**
- 价值函数估计状态价值 V(s)
- 帮助计算优势函数 A(s,a) = Q(s,a) - V(s)
- 价值函数训练不稳定是 PPO 常见问题
- 技巧：价值函数 clip、价值函数归一化

**4. 早停策略**
- 监控验证集奖励
- 奖励不再提升时停止
- 防止过拟合到 Reward Model

## 6. 常见问题与解决方案

### 6.1 SFT 阶段问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 过拟合 | 数据量小、训练轮数多 | 减少轮数、增加数据、早停 |
| 欠拟合 | 模型太小、学习率低 | 增大模型、提高学习率 |
| 灾难性遗忘 | SFT 数据分布窄 | 混合预训练数据、正则化 |
| 格式错误 | 数据预处理不当 | 检查数据格式、模板 |

### 6.2 Reward Model 阶段问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 准确率低 | 标注质量差、数据少 | 改进标注、增加数据 |
| 过拟合 | 训练轮数多 | 早停、正则化 |
| 奖励黑客 | 模型找到奖励漏洞 | 多样化数据、人工检查 |
| 尺度不稳定 | 训练不稳定 | 奖励标准化、梯度裁剪 |

### 6.3 PPO 阶段问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 训练发散 | 学习率太高、KL 太小 | 降低学习率、增大 KL 约束 |
| 奖励下降 | Reward Model 问题、分布偏移 | 检查 Reward Model、减少更新步数 |
| 生成重复 | 熵太低、探索不足 | 增加熵正则、提高温度 |
| 生成胡言乱语 | KL 约束太弱 | 增大 KL 系数、检查参考模型 |
| 显存不足 | 模型太大、批大小大 | 梯度累积、模型并行、量化 |

### 6.4 调试技巧

1. **从小规模开始**
   - 先用小模型（<1B）验证流程
   - 确认无误后再上大模型

2. **逐步增加复杂度**
   - 先跑通 SFT
   - 再训练 Reward Model
   - 最后做 PPO

3. **详细日志**
   - 记录每步的损失、奖励、KL 散度
   - 定期保存检查点
   - 可视化训练曲线

4. **人工检查**
   - 定期检查生成样本质量
   - 不要完全依赖自动指标

## 7. 开源项目参考

### 7.1 OpenRLHF

**GitHub**: https://github.com/OpenRLHF/OpenRLHF

**特点：**
- 高性能 RLHF 实现
- 支持 DeepSpeed 和 FSDP
- 支持多种模型架构
- 提供完整训练脚本

**使用示例：**
```bash
# SFT
openrlhf train_sft \
  --pretrain Qwen/Qwen2.5-7B \
  --dataset data/sft.jsonl \
  --save_path ./sft_output

# Reward Model
openrlhf train_rm \
  --pretrain Qwen/Qwen2.5-7B \
  --dataset data/preference.jsonl \
  --save_path ./rm_output

# PPO
openrlhf train_ppo \
  --pretrain ./sft_output \
  --reward_pretrain ./rm_output \
  --ref_pretrain ./sft_output \
  --save_path ./ppo_output
```

### 7.2 DeepSpeed-Chat

**GitHub**: https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat

**特点：**
- 微软官方实现
- 集成 DeepSpeed 优化
- 支持多节点训练
- 详细文档和教程

### 7.3 TRL (Transformer Reinforcement Learning)

**GitHub**: https://github.com/huggingface/trl

**特点：**
- Hugging Face 官方库
- 与 transformers 无缝集成
- 支持 DPO、PPO、CPO 等多种算法
- 易于使用和扩展

**使用示例：**
```python
from trl import PPOTrainer, PPOConfig

config = PPOConfig(
    model_name="./sft_output",
    learning_rate=1e-6,
    batch_size=128,
    mini_batch_size=32,
)

trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset)

for batch in dataloader:
    response_tensors = trainer.generate(batch["prompt"])
    rewards = reward_model(response_tensors)
    stats = trainer.step(batch, response_tensors, rewards)
```

### 7.4 LLaMA-Factory

**GitHub**: https://github.com/hiyouga/LLaMA-Factory

**特点：**
- 统一微调框架
- 支持 SFT、DPO、PPO、ORPO 等
- Web UI 界面
- 支持众多模型

## 8. 代码实现示例

详见 `code/rlhf_practice.py`，包含：
- 完整 RLHF Pipeline 实现
- SFT 训练代码
- Reward Model 训练
- PPO 微调
- 端到端示例

## 9. 最佳实践总结

### 9.1 数据方面
1. **质量优先**：1000 条高质量数据 > 10000 条低质量数据
2. **多样性**：覆盖多种任务类型和场景
3. **一致性**：标注指南清晰，标注员培训到位
4. **持续迭代**：根据模型表现补充数据

### 9.2 训练方面
1. **渐进式**：SFT → Reward → PPO，逐步验证
2. **监控**：详细日志，定期检查生成质量
3. **保守**：宁可欠拟合也不要过拟合
4. **备份**：频繁保存检查点

### 9.3 评估方面
1. **多维评估**：自动指标 + 人工评估
2. **A/B 测试**：与基线模型对比
3. **边界测试**：测试极端情况和对抗样本
4. **持续监控**：部署后持续收集反馈

## 10. 参考文献

1. **RLHF 原论文**：Ouyang, L., et al. "Training Language Models to Follow Instructions with Human Feedback." NeurIPS 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

2. **InstructGPT**：Ouyang, L., et al. "Training Language Models to Follow Instructions with Human Feedback." [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

3. **PPO 算法**：Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347

4. **DeepSpeed-Chat**：DeepSpeed Team. "DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training." [GitHub](https://github.com/microsoft/DeepSpeedExamples)

5. **OpenRLHF**：OpenRLHF Team. "OpenRLHF: High Performance RLHF Framework." [GitHub](https://github.com/OpenRLHF/OpenRLHF)

6. **TRL 库**：Hugging Face. "Transformer Reinforcement Learning." [GitHub](https://github.com/huggingface/trl)

7. **LLaMA-Factory**：hiyouga. "LLaMA-Factory: Unified Fine-Tuning Framework." [GitHub](https://github.com/hiyouga/LLaMA-Factory)

8. **Reward Modeling**：Stiennon, N., et al. "Learning to Summarize with Human Feedback." NeurIPS 2020. [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)

---

*教程完。祝学习愉快！*
