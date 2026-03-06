# 第 25 章：DPO 与直接偏好优化

## 1. DPO 背景与动机

### 1.1 为什么需要 DPO？

在大语言模型的对齐（Alignment）领域，传统的 RLHF（Reinforcement Learning from Human Feedback）流程虽然有效，但存在以下问题：

1. **复杂度高**：需要训练独立的 Reward Model，然后使用 PPO 等强化学习算法进行优化
2. **训练不稳定**：PPO 训练过程中需要调调多个超参数，容易发散
3. **计算资源消耗大**：需要同时维护多个模型（Policy、Reference、Reward、Value）
4. **实现复杂**：工程实现难度大，调试困难

**DPO（Direct Preference Optimization）** 由 Rafailov 等人于 2023 年提出，其核心思想是：
> **绕过显式的 Reward Model 和强化学习，直接从偏好数据中优化策略模型。**

### 1.2 DPO 的核心洞察

DPO 的关键洞察是：**最优策略可以解析地表示为 Reward 函数的形式**。这意味着我们可以：
- 直接从偏好数据优化策略
- 无需显式训练 Reward Model
- 将强化学习问题转化为监督学习问题

## 2. Bradley-Terry 偏好模型

### 2.1 偏好建模基础

Bradley-Terry 模型是成对比较的经典统计模型。在语言模型对齐中，给定一个提示 $x$ 和两个候选回复 $y_1, y_2$，人类偏好可以表示为：

$$P(y_1 \succ y_2 | x) = \frac{\exp(r(x, y_1))}{\exp(r(x, y_1)) + \exp(r(x, y_2))} = \sigma(r(x, y_1) - r(x, y_2))$$

其中：
- $r(x, y)$ 是奖励函数
- $\sigma$ 是 sigmoid 函数
- $y_1 \succ y_2$ 表示 $y_1$ 优于 $y_2$

### 2.2 从偏好到奖励

给定偏好数据集 $\mathcal{D} = \{(x_i, y_i^w, y_i^l)\}$，其中 $y_i^w$ 是偏好回复（winner），$y_i^l$ 是拒绝回复（loser），最大似然估计为：

$$\max_r \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}[\log \sigma(r(x, y_w) - r(x, y_l))]$$

## 3. DPO 损失函数推导

### 3.1 最优策略的解析形式

在带 KL 散度约束的 RLHF 中，优化目标为：

$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(y|x)}[r(x, y)] - \beta \mathbb{D}_{KL}[\pi(y|x) || \pi_{ref}(y|x)]$$

其中 $\pi_{ref}$ 是参考策略（通常是 SFT 模型）。

**关键定理**：上述优化问题的最优解为：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

其中 $Z(x) = \sum_y \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ 是配分函数。

### 3.2 从策略反推奖励

从上述公式可以解出奖励函数：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

### 3.3 DPO 损失函数

将奖励表达式代入 Bradley-Terry 模型：

$$P(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

注意 $\log Z(x)$ 项被消去了！

**DPO 损失函数**：

$$\mathcal{L}_{DPO}(\pi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

### 3.4 损失函数的直观理解

定义 **隐式奖励**：$r_\pi(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)}$

DPO 损失等价于：
- 最大化偏好回复的隐式奖励
- 最小化拒绝回复的隐式奖励
- 通过 $\beta$ 控制与参考策略的偏离程度

## 4. 与 RLHF/PPO 对比

| 特性 | RLHF (PPO) | DPO |
|------|-----------|-----|
| **Reward Model** | 需要独立训练 | 隐式，无需训练 |
| **优化算法** | PPO（强化学习） | 交叉熵（监督学习） |
| **模型数量** | 4 个（Policy, Ref, Reward, Value） | 2 个（Policy, Ref） |
| **训练稳定性** | 较难调参，易发散 | 稳定，类似 SFT |
| **计算开销** | 高 | 中等 |
| **实现复杂度** | 高 | 低 |
| **内存占用** | 高 | 中等 |

### 4.1 优势总结

1. **简单**：只需修改损失函数，无需复杂的 RL 基础设施
2. **稳定**：避免了 PPO 的训练不稳定问题
3. **高效**：训练速度更快，资源消耗更少
4. **效果相当**：多项研究表明 DPO 能达到与 RLHF 相当甚至更好的效果

## 5. DPO 变体

### 5.1 IPO（Identity Preference Optimization）

IPO 使用不同的损失形式，避免了对 $\beta$ 的敏感性：

$$\mathcal{L}_{IPO}(\pi) = \mathbb{E}\left[\left(\log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)} - \tau\right)^2\right]$$

其中 $\tau$ 是目标 margin。

### 5.2 KTO（Kahneman-Tversky Optimization）

KTO 基于前景理论（Prospect Theory），处理非配对偏好数据：

$$\mathcal{L}_{KTO} = \mathbb{E}\left[\lambda_w \cdot \mathbb{I}(y \text{ is desired}) \cdot (1 - \sigma(r(x, y) - \delta)) + \lambda_l \cdot \mathbb{I}(y \text{ is undesired}) \cdot (1 - \sigma(\delta - r(x, y)))\right]$$

### 5.3 SLiC（Sequence Likelihood Calibration）

SLiC 使用合页损失（hinge loss）进行偏好优化：

$$\mathcal{L}_{SLiC} = \mathbb{E}\left[\max(0, \gamma - \log \pi(y_w|x) + \log \pi(y_l|x))\right]$$

### 5.4 其他变体

- **ORPO**：Odds Ratio Preference Optimization，结合 SFT 和偏好优化
- **SimPO**：Simple Preference Optimization，使用平均 token 概率
- **CPO**：Contrastive Preference Optimization，对比学习视角

## 6. 实践技巧与超参数

### 6.1 关键超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| $\beta$ | 0.1 - 0.5 | KL 约束强度，越大越接近参考策略 |
| learning_rate | 1e-6 - 5e-6 | 学习率，通常比 SFT 小 |
| batch_size | 16 - 128 | 根据显存调整 |
| max_length | 512 - 2048 | 序列长度 |

### 6.2 数据准备

1. **数据质量**：偏好数据质量直接影响效果
2. **数据多样性**：覆盖多种场景和任务
3. **数据平衡**：确保不同类别的偏好样本均衡
4. **去重**：去除重复或高度相似的样本

### 6.3 训练技巧

1. **参考策略选择**：使用高质量的 SFT 模型作为 $\pi_{ref}$
2. **冻结参考模型**：训练时 $\pi_{ref}$ 不更新梯度
3. **梯度累积**：显存不足时使用梯度累积
4. **混合精度**：使用 FP16/BF16 加速训练
5. **早停策略**：监控验证集损失，防止过拟合

### 6.4 常见问题

**问题 1：训练损失下降但效果变差**
- 可能是 $\beta$ 太小，导致过度优化
- 解决方案：增大 $\beta$，加强 KL 约束

**问题 2：训练不稳定**
- 检查学习率是否过大
- 确保参考模型正确加载且冻结

**问题 3：生成质量下降**
- 可能是偏好数据质量问题
- 检查数据标注一致性

## 7. 代码实现示例

详见 `code/dpo.py`，包含：
- DPOLoss 实现
- DPOTrainer 类
- 完整训练流程
- 与 PPO 对比示例

## 8. 参考文献

1. **DPO 原论文**：Rafailov, R., et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

2. **RLHF 基础**：Ouyang, L., et al. "Training Language Models to Follow Instructions with Human Feedback." NeurIPS 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

3. **IPO**：Azar, M. G., et al. "A General Theory of Preference Optimization." arXiv:2310.12962

4. **KTO**：Ethayarajh, K., et al. "Kahneman-Tversky Optimization." arXiv:2402.01306

5. **ORPO**：Hong, J., et al. "Reference-Free Monolithic Preference Optimization." arXiv:2403.07691

6. **SimPO**：Meng, Y., et al. "Simple Preference Optimization." arXiv:2405.14734

7. **TRL 库**：Hugging Face TRL (Transformer Reinforcement Learning) - https://github.com/huggingface/trl

---

*下一章：第 26 章 RLHF 实战*
