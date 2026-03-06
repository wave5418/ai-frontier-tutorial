# 第 23 章 强化学习基础 (Reinforcement Learning Basics)

## 1. 强化学习定义与核心概念

### 1.1 什么是强化学习？

强化学习（Reinforcement Learning, RL）是机器学习的一个分支，关注智能体（Agent）如何在与环境（Environment）的交互中学习最优行为策略，以最大化累积奖励。

**核心特点：**
- **试错学习**：通过尝试不同动作，观察结果来学习
- **延迟奖励**：当前动作的影响可能在未来才显现
- **序列决策**：需要考虑动作的长期后果

### 1.2 核心概念

| 概念 | 符号 | 说明 |
|------|------|------|
| **状态 (State)** | $s$ | 环境在某一时刻的情况描述 |
| **动作 (Action)** | $a$ | 智能体可以执行的行为 |
| **奖励 (Reward)** | $r$ | 环境对动作的即时反馈 |
| **策略 (Policy)** | $\pi$ | 状态到动作的映射规则 |
| **价值函数 (Value Function)** | $V(s), Q(s,a)$ | 评估状态或状态 - 动作对的好坏 |
| **折扣因子 (Discount Factor)** | $\gamma$ | 未来奖励的折现率 (0≤γ≤1) |

### 1.3 强化学习的基本框架

```
智能体 (Agent) ←→ 环境 (Environment)
    ↓ 动作 a         ↓ 状态 s, 奖励 r
```

**交互流程：**
1. 智能体观察当前状态 $s_t$
2. 根据策略选择动作 $a_t$
3. 环境返回新状态 $s_{t+1}$ 和奖励 $r_t$
4. 重复直到任务完成

---

## 2. 马尔可夫决策过程 (MDP)

### 2.1 MDP 定义

马尔可夫决策过程是强化学习的数学基础，定义为五元组 $(S, A, P, R, \gamma)$：

- $S$：状态空间
- $A$：动作空间
- $P$：状态转移概率 $P(s'|s,a)$
- $R$：奖励函数 $R(s,a,s')$
- $\gamma$：折扣因子

### 2.2 马尔可夫性质

**马尔可夫性质**：下一状态只依赖于当前状态和动作，与历史无关
$$P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1}|s_t, a_t)$$

### 2.3 回报与价值函数

**累积回报 (Return)**：
$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

**状态价值函数**：
$$V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s]$$

**动作价值函数 (Q 函数)**：
$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]$$

### 2.4 贝尔曼方程

**贝尔曼期望方程**：
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$$

**贝尔曼最优方程**：
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$$

---

## 3. 价值迭代与策略迭代

### 3.1 价值迭代 (Value Iteration)

**核心思想**：通过迭代更新价值函数，直到收敛到最优价值函数。

**算法步骤**：
1. 初始化 $V(s)$ 为任意值（如 0）
2. 对每个状态 $s$，更新：
   $$V(s) \leftarrow \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V(s')]$$
3. 重复步骤 2 直到收敛
4. 提取最优策略：$\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$

### 3.2 策略迭代 (Policy Iteration)

**核心思想**：交替进行策略评估和策略改进。

**算法步骤**：
1. 初始化任意策略 $\pi$
2. **策略评估**：计算当前策略的价值函数 $V^\pi$
3. **策略改进**：$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$
4. 如果策略不再改变，结束；否则返回步骤 2

### 3.3 两种方法对比

| 特性 | 价值迭代 | 策略迭代 |
|------|----------|----------|
| 收敛速度 | 较慢 | 较快 |
| 每步计算量 | 小 | 大 |
| 适用场景 | 状态空间大 | 状态空间适中 |

---

## 4. Q-Learning 与 SARSA

### 4.1 Q-Learning（异策略学习）

**核心思想**：直接学习最优动作价值函数 $Q^*(s,a)$，无需环境模型。

**更新公式**：
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)]$$

**特点**：
- 异策略 (Off-policy)：学习最优策略，但可以用其他策略探索
- 贪婪更新：使用下一状态的最大 Q 值
- 可能高估 Q 值

### 4.2 SARSA（同策略学习）

**核心思想**：学习当前策略下的动作价值函数。

**更新公式**：
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$

**特点**：
- 同策略 (On-policy)：学习和执行同一策略
- 使用实际选择的下一动作 $a_{t+1}$
- 更保守，考虑探索成本

### 4.3 Q-Learning vs SARSA

| 特性 | Q-Learning | SARSA |
|------|------------|-------|
| 策略类型 | 异策略 | 同策略 |
| 更新目标 | $\max_a Q(s_{t+1}, a)$ | $Q(s_{t+1}, a_{t+1})$ |
| 探索影响 | 不考虑 | 考虑 |
| 收敛 | 最优策略 | 当前策略 |

---

## 5. 深度强化学习 (DQN)

### 5.1 DQN 核心思想

将深度学习与 Q-Learning 结合，用神经网络近似 Q 函数：
$$Q(s,a) \approx Q(s,a; \theta)$$

### 5.2 关键技术

**1. 经验回放 (Experience Replay)**
- 存储转移 $(s,a,r,s')$ 到回放缓冲区
- 随机采样打破数据相关性
- 提高数据利用率

**2. 目标网络 (Target Network)**
- 使用独立的目标网络计算 TD 目标
- 定期更新目标网络参数
- 提高训练稳定性

**3. TD 目标**：
$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

**4. 损失函数**：
$$L(\theta) = \mathbb{E}[(y - Q(s,a;\theta))^2]$$

### 5.3 DQN 算法流程

```
初始化回放缓冲区 D
初始化 Q 网络参数θ，目标网络参数θ^- = θ

for episode = 1 to N:
    初始化状态 s
    for t = 1 to T:
        以ε-贪婪策略选择动作 a
        执行动作 a，观察 r, s'
        存储 (s,a,r,s') 到 D
        从 D 随机采样小批量
        计算 TD 目标 y
        更新θ以最小化损失
        每 C 步更新θ^- = θ
        s = s'
```

---

## 6. 策略梯度方法基础

### 6.1 策略梯度定理

直接优化策略参数 $\theta$，最大化期望回报：
$$J(\theta) = \mathbb{E}_\pi[G]$$

**策略梯度**：
$$\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a)]$$

### 6.2 REINFORCE 算法

**蒙特卡洛策略梯度**：
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

**更新规则**：
$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

### 6.3 基线 (Baseline)

引入基线 $b(s)$ 减少方差，不影响期望：
$$\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta \log \pi_\theta(a|s) \cdot (Q^\pi(s,a) - b(s))]$$

常用基线：状态价值 $V(s)$

**优势函数 (Advantage)**：
$$A(s,a) = Q(s,a) - V(s)$$

---

## 7. 代码实现示例

详见 `code/rl_basics.py`，包含：
- MDP 环境类
- 价值迭代与策略迭代
- Q-Learning 与 SARSA
- DQN 简化实现
- 策略梯度示例

---

## 8. 参考文献

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
3. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3), 229-256.
4. Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3), 279-292.
5. Bertsekas, D. P. (2012). *Dynamic Programming and Optimal Control*. Athena Scientific.

---

**下一章预告**：第 24 章将深入讲解 PPO（Proximal Policy Optimization）与高级策略梯度方法。
