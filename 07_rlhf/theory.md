# 第 07 章 RLHF 与人类对齐

## 目录

1. [为什么需要对齐（Alignment Problem）](#为什么需要对齐alignment-problem)
2. [RLHF 三阶段详解](#rlhf 三阶段详解)
3. [PPO 算法原理](#ppo 算法原理)
4. [Reward Model 训练](#reward-model 训练)
5. [DPO（Direct Preference Optimization）](#dpo direct-preference-optimization)
6. [其他对齐方法](#其他对齐方法)
7. [人类反馈数据收集](#人类反馈数据收集)
8. [代码实现示例](#代码实现示例)
9. [参考文献](#参考文献)

---

## 为什么需要对齐（Alignment Problem）

### 1.1 什么是对齐问题

**对齐问题（Alignment Problem）** 指的是如何确保人工智能系统的目标和行为与人类的价值观、意图和利益保持一致。随着 AI 系统变得越来越强大，如果它们的目标与人类不完全对齐，可能会产生严重的负面后果。

### 1.2 为什么需要对齐

#### 1.2.1 能力与目标的错配

现代大语言模型具有强大的能力，但它们的目标函数（通常是下一个 token 预测）与人类的期望并不完全一致：

- **目标函数局限**：预训练模型优化的是语言建模损失，而非人类偏好
- **分布外泛化**：模型在新场景下可能产生不符合预期的行为
- **奖励黑客（Reward Hacking）**：模型可能找到最大化奖励但不符合人类意图的方式

#### 1.2.2 潜在风险

1. **有害内容生成**：模型可能生成偏见、歧视、虚假信息
2. **指令遵循偏差**：模型可能误解或故意曲解用户意图
3. **工具滥用风险**：强大的 AI 可能被用于恶意目的
4. **长期对齐问题**：随着 AI 能力提升，小的对齐偏差可能被放大

#### 1.2.3 实际案例

- **模型幻觉**：自信地生成错误信息
- **提示注入攻击**：被恶意提示操控输出
- **过度顺从**：即使请求有害也尝试完成
- **价值漂移**：在微调过程中丢失原有安全约束

### 1.3 对齐的核心挑战

```
挑战 1: 人类价值观的复杂性
  - 价值观因文化、情境而异
  - 难以形式化表达
  - 存在内在冲突

挑战 2: 规格化问题（Specification Problem）
  - 难以完整描述期望行为
  - 存在未指定的边缘情况
  - 奖励函数可能被钻空子

挑战 3: 泛化问题
  - 训练分布 vs 测试分布差异
  - 新场景下的行为不确定性
  - 能力扩展带来的新风险
```

---

## RLHF 三阶段详解

RLHF（Reinforcement Learning from Human Feedback，人类反馈强化学习）是目前最成功的对齐方法之一，包含三个核心阶段：

### 2.1 第一阶段：监督微调（SFT - Supervised Fine-Tuning）

#### 2.1.1 目标

将预训练模型微调为能够遵循指令的助手模型。

#### 2.1.2 数据准备

收集高质量的人类示范数据：
- 指令 - 响应对（prompt-response pairs）
- 覆盖多种任务类型和场景
- 确保响应质量高、有帮助、无害

#### 2.1.3 训练过程

```python
# SFT 损失函数
L_SFT = -E[(x,y)~D] [log P_θ(y|x)]

# 其中：
# x: 输入提示
# y: 人类示范响应
# θ: 模型参数
# D: 示范数据集
```

#### 2.1.4 关键要点

- **数据质量 > 数据数量**：少量高质量示范优于大量低质量数据
- **多样性**：覆盖广泛的指令类型
- **一致性**：确保示范风格一致

### 2.2 第二阶段：奖励模型训练（Reward Model Training）

#### 2.2.1 目标

训练一个模型来预测人类对模型输出的偏好，作为强化学习的奖励信号。

#### 2.2.2 数据收集

收集人类偏好数据：
- 对同一提示的多个模型响应
- 人类标注者选择更优响应
- 形成偏好对 (prompt, response_A, response_B, label)

#### 2.2.3 奖励模型架构

```
输入：提示 + 响应
  ↓
基础语言模型（共享权重）
  ↓
奖励头（线性层）
  ↓
标量奖励值
```

#### 2.2.4 训练目标

使用 Bradley-Terry 模型建模偏好：

```python
# Bradley-Terry 偏好概率
P(r_A > r_B) = σ(r_A - r_B) = exp(r_A) / (exp(r_A) + exp(r_B))

# 奖励模型损失
L_RM = -E[(x,y_w,y_l)~D] [log σ(r_φ(x,y_w) - r_φ(x,y_l))]

# 其中：
# y_w: 被偏好的响应（winner）
# y_l: 不被偏好的响应（loser）
# r_φ: 奖励模型
# σ: sigmoid 函数
```

### 2.3 第三阶段：强化学习优化（RL Optimization）

#### 2.3.1 目标

使用奖励模型作为信号，通过强化学习进一步优化策略模型。

#### 2.3.2 核心组件

- **策略模型（Policy）**：生成响应的语言模型
- **奖励模型（Reward）**：评估响应质量
- **参考模型（Reference）**：SFT 模型，用于 KL 散度约束

#### 2.3.3 优化目标

```python
# 带 KL 约束的强化学习目标
max_θ E[x~D, y~π_θ] [r_φ(x,y) - β * KL(π_θ(y|x) || π_ref(y|x))]

# 其中：
# π_θ: 当前策略
# π_ref: 参考策略（SFT 模型）
# β: KL 惩罚系数
# r_φ: 奖励模型
```

#### 2.3.4 为什么需要 KL 约束

- 防止策略偏离太远，保持语言流畅性
- 避免奖励黑客行为
- 保持模型的通用能力

---

## PPO 算法原理

PPO（Proximal Policy Optimization，近端策略优化）是 RLHF 中最常用的强化学习算法。

### 3.1 PPO 的核心思想

PPO 通过限制策略更新的幅度，确保训练稳定性：

```
核心洞察：
- 大的策略更新可能导致性能崩溃
- 小的、保守的更新更稳定
- 使用截断的重要性采样来限制更新幅度
```

### 3.2 重要性采样比率

```python
# 重要性采样比率
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

# 含义：新策略相对于旧策略的概率比率
# 作用：用旧策略采集的数据来评估新策略
```

### 3.3 截断的代理目标

```python
# PPO 截断目标
L_CLIP(θ) = E_t [min(
    r_t(θ) * Â_t,
    clip(r_t(θ), 1-ε, 1+ε) * Â_t
)]

# 其中：
# Â_t: 优势函数估计
# ε: 截断范围（通常 0.2）
# clip: 将比率限制在 [1-ε, 1+ε] 范围内
```

### 3.4 优势函数估计

```python
# GAE（Generalized Advantage Estimation）
Â_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

# TD 残差
δ_t = r_t + γV(s_{t+1}) - V(s_t)

# 其中：
# γ: 折扣因子
# λ: GAE 参数（平衡偏差和方差）
# V: 价值函数
```

### 3.5 PPO 完整目标函数

```python
# PPO 总损失
L_PPO(θ) = L_CLIP(θ) - c1 * L_VF(θ) + c2 * S[π_θ]

# 组成部分：
# L_CLIP: 策略损失（截断）
# L_VF: 价值函数损失
# S: 策略熵（鼓励探索）
# c1, c2: 系数
```

### 3.6 PPO 在 RLHF 中的特殊处理

```python
# RLHF 中的 PPO 目标
L = E[(x,y)~D] [
    min(
        r_t(θ) * A(x,y),
        clip(r_t(θ), 1-ε, 1+ε) * A(x,y)
    )
] - β * KL(π_θ || π_ref)

# 其中优势函数：
A(x,y) = r_φ(x,y) - V(x)  # 奖励减基线
```

### 3.7 PPO 训练流程

```
1. 从当前策略采样轨迹
2. 计算优势函数（使用 GAE）
3. 多次优化 PPO 目标（小批量）
4. 更新旧策略为当前策略
5. 重复直到收敛
```

---

## Reward Model 训练

### 4.1 数据格式

```python
# 偏好数据示例
preference_data = [
    {
        "prompt": "如何学习编程？",
        "chosen": "学习编程的好方法是...",  # 人类偏好的响应
        "rejected": "编程很难，你不应该学..."  # 人类不偏好的响应
    },
    # ... 更多样本
]
```

### 4.2 模型架构

```python
# 奖励模型架构
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model  # 预训练语言模型
        self.reward_head = nn.Linear(hidden_size, 1)  # 奖励头
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask)
        # 使用最后一个 token 的隐藏状态
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward.squeeze()
```

### 4.3 训练细节

#### 4.3.1 损失函数

```python
# Bradley-Terry 损失
def reward_loss(reward_chosen, reward_rejected):
    # 计算偏好概率
    logits = reward_chosen - reward_rejected
    # 二元交叉熵损失
    loss = -log_sigmoid(logits).mean()
    return loss
```

#### 4.3.2 训练技巧

1. **学习率调度**：使用较小的学习率（1e-5 到 5e-5）
2. **批大小**：较大的批大小有助于稳定训练
3. **正则化**：防止过拟合偏好数据
4. **早停**：在验证集上监控性能

#### 4.3.3 评估指标

```python
# 准确率：预测正确偏好的比例
accuracy = (predicted_winner == actual_winner).mean()

# 边际：奖励差异的均值
margin = (reward_chosen - reward_rejected).mean()
```

### 4.4 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 奖励坍塌 | 模型给所有输出相似奖励 | 增加温度、调整学习率 |
| 过拟合 | 在训练偏好上过拟合 | 数据增强、正则化 |
| 分布偏移 | 与策略模型分布不同 | 在线数据收集、混合数据 |

---

## DPO（Direct Preference Optimization）

### 5.1 DPO 的核心思想

DPO 是一种**无需显式奖励模型**的对齐方法，直接在偏好数据上优化策略。

```
传统 RLHF:
  偏好数据 → 奖励模型 → RL 优化 → 策略

DPO:
  偏好数据 → 直接优化策略
```

### 5.2 理论基础

#### 5.2.1 从 RLHF 到 DPO 的推导

RLHF 的最优策略可以解析表达：

```python
# RLHF 最优策略（带 KL 约束）
π*(y|x) = (1/Z(x)) * π_ref(y|x) * exp(r(x,y)/β)

# 其中 Z(x) 是配分函数
# 这个关系允许我们直接从偏好学习，无需显式奖励
```

#### 5.2.2 DPO 损失函数

```python
# DPO 损失
L_DPO = -E[(x,y_w,y_l)~D] [
    log σ(
        β * log(π_θ(y_w|x)/π_ref(y_w|x)) 
        - β * log(π_θ(y_l|x)/π_ref(y_l|x))
    )
]

# 简化理解：
# 直接优化策略，使偏好响应的（对数概率 - 参考概率）差值最大化
```

### 5.3 DPO vs RLHF

| 特性 | RLHF | DPO |
|------|------|-----|
| 奖励模型 | 需要 | 不需要 |
| 训练复杂度 | 高（三阶段） | 低（单阶段） |
| 稳定性 | 需要仔细调参 | 更稳定 |
| 计算成本 | 高 | 较低 |
| 性能 | 优秀 | 相当或更好 |

### 5.4 DPO 实现要点

```python
# DPO 训练关键步骤
1. 计算当前策略的对数概率：log π_θ(y|x)
2. 计算参考策略的对数概率：log π_ref(y|x)
3. 计算隐式奖励：r = β * (log π_θ - log π_ref)
4. 应用 DPO 损失优化
```

### 5.5 DPO 的变体

#### 5.5.1 IPO（Identity Preference Optimization）

```python
# IPO 损失 - 使用平方损失代替 sigmoid
L_IPO = E[(x,y_w,y_l)~D] [
    (log(π_θ(y_w|x)/π_ref(y_w|x)) 
     - log(π_θ(y_l|x)/π_ref(y_l|x)) 
     - τ)²
]

# τ: 目标边际参数
```

#### 5.5.2 KTO（Kahneman-Tversky Optimization）

```python
# KTO 损失 - 基于前景理论
L_KTO = E[(x,y)~D] [
    λ_desirable * (1 - σ(r(x,y) - δ_desirable))  # 期望响应
    + λ_undesirable * (1 - σ(δ_undesirable - r(x,y)))  # 不期望响应
]

# 特点：不需要成对偏好，只需好坏标签
```

---

## 其他对齐方法

### 6.1 基于人类反馈的方法

#### 6.1.1 RLAIF（AI 反馈强化学习）

```
使用 AI（如更强的模型）代替人类提供反馈
优点：可扩展、成本低
缺点：可能继承上游模型的偏见
```

#### 6.1.2 宪法 AI（Constitutional AI）

```
核心思想：
- 定义一组原则（"宪法"）
- 模型根据原则自我批评和改进
- 无需人类偏好数据

流程：
1. 生成初始响应
2. 根据宪法原则批评
3. 修订响应
4. 微调模型学习修订行为
```

### 6.2 基于约束的方法

#### 6.2.1 约束 RL

```python
# 带约束的优化
max_θ E[r(x,y)]
s.t. E[c_i(x,y)] ≤ d_i  # 约束条件

# 使用拉格朗日乘子法
L = E[r] - Σ λ_i * (E[c_i] - d_i)
```

#### 6.2.2 安全层（Safety Layer）

```
在输出层添加过滤器：
- 检测有害内容
- 拦截或修改不安全输出
- 可与 RLHF 结合使用
```

### 6.3 基于提示的方法

#### 6.3.1 系统提示工程

```
精心设计的系统提示可以：
- 设定行为准则
- 定义拒绝策略
- 引导模型遵循特定价值观
```

#### 6.3.2 Few-shot 对齐

```
在上下文中提供对齐示例：
- 展示期望的响应风格
- 演示如何处理敏感请求
- 无需参数更新
```

### 6.4 方法比较

| 方法 | 数据需求 | 计算成本 | 灵活性 | 推荐场景 |
|------|----------|----------|--------|----------|
| RLHF | 高（偏好 + 示范） | 高 | 高 | 生产级对齐 |
| DPO | 中（偏好） | 中 | 高 | 快速迭代 |
| IPO | 中（偏好） | 中 | 中 | 稳定训练 |
| KTO | 低（好坏标签） | 低 | 中 | 数据有限 |
| 宪法 AI | 低（原则） | 中 | 高 | 透明对齐 |

---

## 人类反馈数据收集

### 7.1 数据类型

#### 7.1.1 示范数据（Demonstrations）

```python
# 用于 SFT
{
    "prompt": "写一首关于春天的诗",
    "response": "春风拂面花自开，...",
    "metadata": {
        "annotator_id": "human_1",
        "quality_score": 5,
        "task_type": "creative"
    }
}
```

#### 7.1.2 偏好数据（Preferences）

```python
# 用于奖励模型/DPO
{
    "prompt": "解释量子力学",
    "response_A": "...",  # 更详细
    "response_B": "...",  # 较简洁
    "preference": "A",    # 人类选择
    "reason": "更准确完整"
}
```

#### 7.1.3 评分数据（Ratings）

```python
# 用于回归式奖励
{
    "prompt": "...",
    "response": "...",
    "rating": 4,  # 1-5 分
    "dimensions": {
        "helpfulness": 4,
        "honesty": 5,
        "harmlessness": 5
    }
}
```

### 7.2 数据收集流程

```
1. 任务设计
   ↓
2. 标注者招募与培训
   ↓
3. 数据标注
   ↓
4. 质量控制
   ↓
5. 数据清洗与格式化
   ↓
6. 数据集划分（训练/验证/测试）
```

### 7.3 质量控制

#### 7.3.1 标注者筛选

- 资格测试
- 试用期评估
- 持续监控

#### 7.3.2 一致性检查

```python
# 标注者间一致性（Inter-annotator Agreement）
# Cohen's Kappa
κ = (p_o - p_e) / (1 - p_e)

# 目标：κ > 0.6（实质性一致）
```

#### 7.3.3 黄金样本

- 插入已知答案的测试样本
- 监控标注者准确率
- 自动过滤低质量标注

### 7.4 伦理考虑

```
1. 标注者权益
   - 公平报酬
   - 心理健康支持（避免接触有害内容）
   - 知情同意

2. 数据隐私
   - 匿名化处理
   - 安全存储
   - 使用限制

3. 偏见控制
   - 多样化标注者群体
   - 偏见检测
   - 平衡数据集
```

### 7.5 数据规模建议

| 阶段 | 最小规模 | 推荐规模 | 高质量规模 |
|------|----------|----------|------------|
| SFT | 1K | 10K | 50K+ |
| 奖励模型 | 5K 对 | 50K 对 | 200K+ 对 |
| RLHF | - | 与 RM 相同 | - |
| DPO | 5K 对 | 30K 对 | 100K+ 对 |

---

## 代码实现示例

详细的代码实现请参见 `code/rlhf.py` 文件，包含：

- PPO 算法实现（简化版）
- Reward Model 实现
- DPO 损失函数实现
- 完整 RLHF 训练流程示例
- SFT 微调示例

---

## 参考文献

### 核心论文

1. **RLHF 基础**
   - Christiano et al. "Deep Reinforcement Learning from Human Preferences" (2017)
   - Stiennon et al. "Learning to Summarize with Human Feedback" (2020)
   - Ouyang et al. "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT, 2022)

2. **PPO 算法**
   - Schulman et al. "Proximal Policy Optimization Algorithms" (2017)

3. **DPO 及相关**
   - Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)
   - Azar et al. "A General Theoretical Paradigm to Understand Learning from Human Preferences" (IPO, 2023)
   - Ethayarajh et al. "KTO: Model Alignment as Prospect Theoretic Optimization" (2024)

4. **宪法 AI**
   - Bai et al. "Constitutional AI: Harmlessness from AI Feedback" (2022)

### 实践资源

5. **开源实现**
   - Hugging Face TRL (Transformer Reinforcement Learning)
   - OpenAI Spinning Up in Deep RL
   - CleanRL PPO Implementation

6. **教程与课程**
   - Hugging Face RLHF Course
   - Stanford CS324: Large Language Models
   - DeepLearning.AI AI Alignment Courses

### 进一步阅读

7. **对齐研究**
   - Arbital Alignment Problem
   - LessWrong Alignment Forum
   - Anthropic Alignment Research

8. **安全与评估**
   - Hendrycks et al. "Measuring Massive Multitask Language Understanding" (MMLU)
   - Perez et al. "Red Teaming Language Models to Reduce Harms"

---

*本章完*

最后更新：2026 年 3 月
