# 第 07 章 RLHF 与人类对齐

## 文件结构

```
07_rlhf/
├── README.md          # 本文件
├── theory.md          # 理论详解
└── code/
    └── rlhf.py        # 代码实现
```

## 内容概览

### theory.md - 理论部分

包含以下核心内容：

1. **为什么需要对齐（Alignment Problem）**
   - 对齐问题的定义
   - 能力与目标的错配
   - 潜在风险与实际案例

2. **RLHF 三阶段详解**
   - SFT（监督微调）
   - Reward Model（奖励模型）训练
   - RL（强化学习）优化

3. **PPO 算法原理**
   - 核心思想与重要性采样
   - 截断的代理目标
   - GAE 优势估计

4. **Reward Model 训练**
   - Bradley-Terry 模型
   - 训练技巧与评估指标

5. **DPO（Direct Preference Optimization）**
   - 核心思想与理论推导
   - DPO vs RLHF 对比
   - 实现要点

6. **其他对齐方法**
   - IPO、KTO
   - RLAIF、宪法 AI
   - 方法比较

7. **人类反馈数据收集**
   - 数据类型与格式
   - 收集流程与质量控制
   - 伦理考虑

8. **参考文献**
   - 核心论文
   - 实践资源

### code/rlhf.py - 代码实现

包含以下完整实现：

1. **PPO 算法（简化版）**
   - `PPOConfig`: 配置类
   - `ActorCritic`: Actor-Critic 网络
   - `PPOTrainer`: PPO 训练器

2. **Reward Model 实现**
   - `RewardModel`: 奖励模型架构
   - `RewardModelTrainer`: 训练器

3. **DPO 损失函数**
   - `DPOTrainer`: DPO 训练器
   - 完整的损失函数实现

4. **其他对齐方法**
   - `IPOTrainer`: IPO 实现
   - `KTOTrainer`: KTO 实现

5. **完整 RLHF 流程**
   - `RLHFPipeline`: 三阶段完整流程
   - 数据准备工具函数

6. **使用示例**
   - `example_usage()`: 完整使用演示

## 运行要求

```bash
# 安装依赖
pip install torch transformers numpy

# 运行示例
python code/rlhf.py
```

## 学习建议

1. **先读理论**：阅读 `theory.md` 理解核心概念
2. **再看代码**：对照 `rlhf.py` 理解实现细节
3. **动手实践**：修改参数，尝试不同配置
4. **深入阅读**：查阅参考文献中的原始论文

## 关键公式

### Bradley-Terry 偏好模型
```
P(r_A > r_B) = σ(r_A - r_B)
```

### DPO 损失
```
L_DPO = -E[log σ(β * (log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l))))]
```

### PPO 截断目标
```
L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
```

## 下一步

完成本章后，建议继续学习：

- 第 08 章：多模态大模型
- 第 09 章：Agent 与工具使用
- 第 10 章：高效微调技术（LoRA、QLoRA）

---

*AI 前沿技术教程 - 第 07 章*
