# 第 24 章 PPO 与策略梯度 (PPO and Policy Gradient Methods)

## 1. 策略梯度方法回顾

### 1.1 策略梯度的基本思想

策略梯度方法直接优化策略参数 $\theta$，最大化期望回报：
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

其中 $\tau = (s_0, a_0, s_1, a_1, ...)$ 是轨迹，$R(\tau)$ 是轨迹的总回报。

### 1.2 策略梯度定理

**定理**：策略梯度的形式为
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)]$$

**推导要点**：
1. 利用似然比技巧 (Likelihood Ratio Trick)
2. $\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)$
3. 将梯度转化为期望形式

### 1.3 REINFORCE 算法

**更新规则**：
$$\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

**优点**：
- 简单易懂
- 适用于连续和离散动作空间
- 同策略，保证收敛

**缺点**：
- 高方差
- 样本效率低
- 学习率敏感

### 1.4 优势函数 (Advantage Function)

引入基线 $b(s)$ 减少方差：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)]$$

其中优势函数：
$$A(s,a) = Q(s,a) - V(s)$$

**常用估计方法**：
- 蒙特卡洛：$A(s_t, a_t) = G_t - V(s_t)$
- TD 误差：$A(s_t, a_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$
- GAE：多步优势估计（见后文）

---

## 2. TRPO (Trust Region Policy Optimization)

### 2.1 核心思想

TRPO 通过限制策略更新幅度，保证单调改进：

$$\max_\theta \mathbb{E}_{s \sim \rho_{\theta_{old}}, a \sim \pi_{\theta_{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A_{\theta_{old}}(s,a) \right]$$

**约束条件**：
$$\mathbb{E}_{s \sim \rho_{\theta_{old}}} [D_{KL}(\pi_{\theta_{old}}(\cdot|s) || \pi_\theta(\cdot|s))] \leq \delta$$

### 2.2 关键组件

**1. 重要性采样 (Importance Sampling)**
$$\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]$$

在 TRPO 中用于复用旧策略采样的数据。

**2. KL 散度约束**
$$D_{KL}(P || Q) = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$$

限制新旧策略的差异，防止更新过大。

**3. 共轭梯度法**
用于高效求解带约束的优化问题。

### 2.3 TRPO 算法流程

```
初始化策略参数θ
for iteration = 1 to N:
    用π_θ_old 采样轨迹
    计算优势函数 A
    
    计算梯度 g = ∇_θ L(θ)
    计算 Hessian-vector 乘积 Hv
    
    用共轭梯度法求解 Hx = g
    计算步长 α 满足 KL 约束
    
    更新 θ = θ + αx
    线搜索确保目标函数提升
```

### 2.4 TRPO 的局限性

- 实现复杂（需要共轭梯度、线搜索）
- 计算开销大
- 难以与某些网络结构兼容（如 Dropout、BatchNorm）
- 不能直接使用自适应优化器（如 Adam）

---

## 3. PPO 原理详解

### 3.1 PPO 的核心创新

PPO (Proximal Policy Optimization) 是 TRPO 的简化版本，通过**截断目标函数**实现策略更新限制。

### 3.2 截断目标函数 (Clipped Objective)

**PPO 目标函数**：
$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是概率比
- $\hat{A}_t$ 是优势函数估计
- $\epsilon$ 是截断参数（通常 0.1-0.3）

### 3.3 截断机制详解

**情况 1：优势为正 ($\hat{A}_t > 0$)**
- 希望增加动作概率（$r_t(\theta) > 1$）
- 但限制最大增加到 $1+\epsilon$
- 防止策略变化过大

**情况 2：优势为负 ($\hat{A}_t < 0$)**
- 希望减少动作概率（$r_t(\theta) < 1$）
- 但限制最小减少到 $1-\epsilon$
- 防止完全丢弃动作

**可视化**：
```
目标值
  ↑
  |         / 未截断
  |        /
  |-------/---- 截断后 (1+ε)
  |      /
  |-----/------ (1-ε) 截断后
  |    /
  |   /
  +----------------→ 概率比 r(θ)
      1-ε  1  1+ε
```

### 3.4 完整 PPO 目标函数

$$L^{PPO}(\theta) = \mathbb{E}_t [L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t)]$$

其中：
- $L^{CLIP}$：截断策略目标
- $L^{VF}$：价值函数损失 $= (V_\theta(s_t) - V_t^{target})^2$
- $S[\pi_\theta]$：策略熵正则项，鼓励探索
- $c_1, c_2$：系数

---

## 4. PPO 算法变体

### 4.1 PPO-Clip

**特点**：使用截断目标函数
**优势**：实现简单，效果稳定
**适用**：大多数场景的首选

**伪代码**：
```
for iteration = 1 to N:
    用π_θ_old 采样 T 步
    计算优势函数 {Â_1, ..., Â_T}
    
    for epoch = 1 to K:
        打乱数据
        for mini-batch:
            计算 r(θ) = π_θ(a|s) / π_θ_old(a|s)
            计算 L^CLIP(θ)
            更新θ以最大化 L^CLIP
```

### 4.2 PPO-Penalty

**特点**：在目标函数中添加 KL 惩罚项
**公式**：
$$L(\theta) = \mathbb{E}[r_t(\theta) \hat{A}_t] - \beta \cdot D_{KL}(\pi_{\theta_{old}} || \pi_\theta)$$

**自适应 KL 系数**：
- 如果 KL > target_KL，增加β
- 如果 KL < target_KL，减少β

**对比 PPO-Clip**：
| 特性 | PPO-Clip | PPO-Penalty |
|------|----------|-------------|
| 实现难度 | 简单 | 中等 |
| 超参数 | ε | β, target_KL |
| 稳定性 | 高 | 依赖β调节 |
| 推荐使用 | ✓ | 特定场景 |

### 4.3 其他变体

**PPO-λ**：结合 GAE(λ) 的优势估计
**PPO-RNN**：使用循环神经网络处理序列决策
**PPO-MultiHead**：多任务共享表示

---

## 5. 超参数与训练技巧

### 5.1 关键超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| **学习率 (lr)** | 3e-4 | 策略和价值网络 |
| **截断参数 (ε)** | 0.1-0.3 | 限制策略更新幅度 |
| **GAE λ** | 0.95 | 优势估计的衰减 |
| **折扣因子 (γ)** | 0.99 | 未来奖励折现 |
| **价值系数 (c1)** | 0.5 | 价值函数损失权重 |
| **熵系数 (c2)** | 0.01 | 探索鼓励强度 |
| **PPO 轮数 (K)** | 10 | 每次采样的更新轮数 |
| **小批量大小** | 64-256 | 梯度更新批次 |
| **最大梯度范数** | 0.5 | 梯度裁剪 |

### 5.2 训练技巧

**1. 优势归一化**
```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```
- 稳定训练
- 加速收敛

**2. 价值函数裁剪**
```python
v_pred_clipped = v_old + (v_pred - v_old).clamp(-clip_range, clip_range)
```
- 防止价值函数更新过大

**3. 学习率衰减**
```python
lr = initial_lr * (1 - iteration / max_iterations)
```
- 后期精细调优

**4. 早停 (Early Stopping)**
- 监控 KL 散度
- 如果 KL > 2 * target_KL，提前结束当前轮

**5. 奖励归一化/裁剪**
- RunningNorm 归一化奖励
- 或直接裁剪到 [-10, 10]

### 5.3 常见问题与解决

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 训练不稳定 | 学习率过大 | 降低学习率，增加截断ε |
| 收敛慢 | 探索不足 | 增加熵系数，调整ε |
| 性能下降 | 策略崩溃 | 减小截断ε，增加 KL 惩罚 |
| 价值函数发散 | 价值学习率过高 | 降低 c1，使用价值裁剪 |

---

## 6. 应用场景

### 6.1 机器人控制

**任务类型**：
- 行走/跑步（Humanoid, Ant）
- 机械臂操作（Reach, Push, Pick-and-Place）
- 四足机器人（Unitree, Boston Dynamics 仿真）

**PPO 优势**：
- 连续动作空间友好
- 样本效率较高
- 稳定可靠

**典型案例**：
- OpenAI Gym: Humanoid-v3
- MuJoCo 机器人任务
- 真实机器人 Sim-to-Real 迁移

### 6.2 游戏 AI

**经典案例**：
- **OpenAI Five** (Dota 2)：PPO + LSTM
- **AlphaStar** (星际争霸 2)：PPO 变体
- **ProcGen**：程序化生成游戏基准

**特点**：
- 高维观测（图像）
- 长期规划
- 多智能体协作

### 6.3 其他应用

**自然语言处理**：
- 对话系统优化
- 文本生成强化学习

**推荐系统**：
- 序列推荐
- 用户长期满意度优化

**金融交易**：
- 投资组合优化
- 高频交易策略

**自动驾驶**：
- 决策规划
- 行为预测

---

## 7. 代码实现示例

详见 `code/ppo.py`，包含：

### 7.1 核心组件
- Actor-Critic 网络架构
- GAE 优势估计
- PPO-Clip 算法实现
- 连续/离散动作空间支持

### 7.2 功能特性
- 经验回放与多轮更新
- 优势归一化
- 梯度裁剪
- 学习率调度
- 训练可视化

### 7.3 使用示例
```python
# 创建环境
env = gym.make('CartPole-v1')

# 初始化 PPO
agent = PPO(env, hidden_size=64, lr=3e-4)

# 训练
agent.train(total_timesteps=100000)

# 测试
agent.test(render=True)
```

---

## 8. 参考文献

1. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
2. Schulman, J., et al. (2015). Trust Region Policy Optimization. *ICML*.
3. Sutton, R. S., et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation. *NIPS*.
4. Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. *Machine Learning*.
5. Mnih, V., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. *ICML*.
6. Lillicrap, T. P., et al. (2016). Continuous Control with Deep Reinforcement Learning. *ICLR*.
7. Huang, S., et al. (2022). What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study. *arXiv:2203.07814*.

---

## 附录：PPO 与相关算法对比

| 算法 | 策略类型 | 更新方式 | 样本效率 | 稳定性 | 实现难度 |
|------|----------|----------|----------|--------|----------|
| **REINFORCE** | 同策略 | MC 梯度 | 低 | 低 | 简单 |
| **A3C** | 同策略 | 异步 Actor-Critic | 中 | 中 | 中等 |
| **TRPO** | 同策略 | 二阶优化 | 高 | 高 | 复杂 |
| **PPO** | 同策略 | 一阶优化 + 截断 | 高 | 高 | 简单 |
| **DDPG** | 异策略 | Actor-Critic + DQN | 高 | 中 | 中等 |
| **SAC** | 异策略 | 最大熵 RL | 高 | 高 | 中等 |

**PPO 的定位**：在稳定性和实现难度之间取得最佳平衡，是目前最常用的 on-policy 算法。

---

**本章总结**：PPO 通过截断目标函数巧妙地限制了策略更新幅度，既保证了训练稳定性，又避免了 TRPO 的复杂实现。配合 GAE 优势估计和一系列工程技巧，PPO 成为当前强化学习应用的首选算法之一。
