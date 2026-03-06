# 第 13 章 世界模型基础

## 13.1 世界模型定义与动机

### 什么是世界模型？

**世界模型（World Model）** 是一种能够理解和预测环境动态的人工智能系统。它通过构建环境的内部表示（internal representation），使智能体能够在"想象"中模拟未来状态，而无需在真实环境中执行动作。

```
┌─────────────────────────────────────────────────────────┐
│                    世界模型架构                          │
├─────────────────────────────────────────────────────────┤
│  观察 (Observation) → 编码器 → 潜在状态 (Latent State)  │
│                          ↓                              │
│  动作 (Action) ───→ 动态模型 → 预测下一状态              │
│                          ↓                              │
│                    解码器 → 预测观察                     │
└─────────────────────────────────────────────────────────┘
```

### 为什么需要世界模型？

1. **样本效率**：在真实环境中收集数据成本高，世界模型允许在"想象"中训练
2. **安全性**：危险动作可以在模拟中测试，避免真实世界的风险
3. **规划能力**：通过模拟多个未来轨迹，选择最优行动序列
4. **泛化能力**：学习环境的本质规律，而非记忆特定场景

### 核心思想

世界模型的核心思想源于人类认知：我们在做决策前，会在脑海中模拟可能的结果。例如：
- 下棋时，棋手会思考"如果我走这一步，对手会如何回应"
- 开车时，司机会预测"如果现在变道，是否安全"

```python
# 世界模型的基本接口
class WorldModel:
    def encode(self, observation):
        """将观察编码为潜在状态"""
        pass
    
    def predict(self, latent_state, action):
        """预测下一状态"""
        pass
    
    def decode(self, latent_state):
        """从潜在状态解码为观察"""
        pass
```

## 13.2 预测未来状态的能力

### 状态预测的数学形式

给定当前状态 $s_t$ 和动作 $a_t$，世界模型学习预测下一状态 $s_{t+1}$：

$$s_{t+1} = f(s_t, a_t) + \epsilon$$

其中 $f$ 是学习到的动态函数，$\epsilon$ 是环境的不确定性。

### 确定性 vs 随机性预测

**确定性模型**：
- 输出单一的预测结果
- 适用于确定性环境
- 简单但无法处理不确定性

**随机性模型**：
- 输出状态的概率分布
- 能够表示环境的随机性
- 更适合真实世界的复杂环境

```python
import torch
import torch.nn as nn

class DeterministicPredictor(nn.Module):
    """确定性状态预测器"""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state, action):
        # 拼接状态和动作
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class StochasticPredictor(nn.Module):
    """随机性状态预测器（输出均值和方差）"""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, state_dim)
        self.logvar_head = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        h = self.shared(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar
    
    def sample(self, state, action):
        mean, logvar = self.forward(state, action)
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(mean)
        return mean + std * noise
```

### 多步预测

世界模型可以递归地进行多步预测：

```python
def multi_step_predict(model, initial_state, actions, num_steps):
    """
    多步状态预测
    
    Args:
        model: 世界模型
        initial_state: 初始状态
        actions: 动作序列 [a_0, a_1, ..., a_{T-1}]
        num_steps: 预测步数
    
    Returns:
        预测的状态序列
    """
    states = [initial_state]
    current_state = initial_state
    
    for t in range(num_steps):
        action = actions[t] if t < len(actions) else torch.zeros_like(actions[0])
        next_state = model.predict(current_state, action)
        states.append(next_state)
        current_state = next_state
    
    return states
```

## 13.3 Dreamer 架构详解

### Dreamer 概述

**Dreamer** 是由 Hafner 等人提出的基于世界模型的强化学习算法。它的核心创新是：
1. 使用变分自编码器（VAE）学习紧凑的状态表示
2. 使用循环状态空间模型（RSSM）预测未来
3. 在潜在空间中进行"想象"训练

### RSSM（Recurrent State Space Model）

RSSM 是 Dreamer 的核心组件，结合了确定性 RNN 和随机潜在变量：

```
┌─────────────────────────────────────────────────────────┐
│              RSSM 状态转移                               │
├─────────────────────────────────────────────────────────┤
│  h_{t-1} (确定性隐藏状态)                                │
│     ↓                                                   │
│  GRU → h_t^prior (先验隐藏状态)                         │
│     ↓                                                   │
│  与 z_{t-1} (随机状态) 结合 → s_t (完整状态)            │
│     ↓                                                   │
│  生成先验分布 p(z_t | s_t)                              │
│     ↓                                                   │
│  与观察 o_t 结合 → 后验分布 q(z_t | s_t, o_t)           │
└─────────────────────────────────────────────────────────┘
```

```python
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

class RSSM(nn.Module):
    """Recurrent State Space Model"""
    def __init__(self, obs_dim, action_dim, latent_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # 编码器：观察 → 随机状态
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean 和 logvar
        )
        
        # 动态模型：(隐藏状态，随机状态，动作) → 新隐藏状态
        self.gru = nn.GRUCell(hidden_dim + latent_dim + action_dim, hidden_dim)
        
        # 先验头：隐藏状态 → 随机状态先验
        self.prior_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # 解码器：(隐藏状态，随机状态) → 观察重建
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    
    def encode_obs(self, obs):
        """将观察编码为随机状态的均值和方差"""
        params = self.obs_encoder(obs)
        mean, logvar = torch.chunk(params, 2, dim=-1)
        return mean, logvar
    
    def get_prior(self, hidden_state):
        """从隐藏状态获得先验分布"""
        params = self.prior_head(hidden_state)
        mean, logvar = torch.chunk(params, 2, dim=-1)
        return mean, logvar
    
    def get_posterior(self, hidden_state, obs):
        """从隐藏状态和观察获得后验分布"""
        return self.encode_obs(obs)
    
    def forward(self, obs, action, prev_hidden, prev_stochastic):
        """
        RSSM 前向传播
        
        Returns:
            hidden_state: 新的确定性隐藏状态
            stochastic_mean: 随机状态均值
            stochastic_logvar: 随机状态对数方差
            prior_mean: 先验均值
            prior_logvar: 先验对数方差
        """
        # 1. 计算先验
        prior_mean, prior_logvar = self.get_prior(prev_hidden)
        
        # 2. 采样随机状态（训练时用后验，推理时用先验）
        post_mean, post_logvar = self.get_posterior(prev_hidden, obs)
        std = torch.exp(0.5 * post_logvar)
        noise = torch.randn_like(post_mean)
        stochastic = post_mean + std * noise
        
        # 3. 更新隐藏状态
        gru_input = torch.cat([prev_hidden, prev_stochastic, action], dim=-1)
        hidden_state = self.gru(gru_input, prev_hidden)
        
        return hidden_state, post_mean, post_logvar, prior_mean, prior_logvar
    
    def decode(self, hidden_state, stochastic):
        """从状态解码观察"""
        state = torch.cat([hidden_state, stochastic], dim=-1)
        return self.decoder(state)
    
    def compute_kl_loss(self, post_mean, post_logvar, prior_mean, prior_logvar):
        """计算 KL 散度损失"""
        posterior = Normal(post_mean, torch.exp(0.5 * post_logvar))
        prior = Normal(prior_mean, torch.exp(0.5 * prior_logvar))
        kl = kl_divergence(posterior, prior).sum(dim=-1)
        return kl.mean()
```

### Dreamer 训练流程

```python
class Dreamer:
    """Dreamer 算法实现"""
    def __init__(self, config):
        self.rssm = RSSM(
            obs_dim=config['obs_dim'],
            action_dim=config['action_dim'],
            latent_dim=config['latent_dim'],
            hidden_dim=config['hidden_dim']
        )
        self.actor = ActorNetwork(config)
        self.critic = CriticNetwork(config)
        self.config = config
    
    def train(self, batch):
        """
        训练 Dreamer
        
        Args:
            batch: 经验回放批次 {obs, action, reward, done}
        """
        obs = batch['obs']  # [T, B, obs_dim]
        action = batch['action']  # [T, B, action_dim]
        reward = batch['reward']  # [T, B, 1]
        
        T, B = obs.shape[0], obs.shape[1]
        
        # 1. RSSM 前向传播
        hidden = torch.zeros(B, self.rssm.hidden_dim)
        stochastic = torch.zeros(B, self.rssm.latent_dim)
        
        kl_losses = []
        recon_losses = []
        
        for t in range(T):
            hidden, post_mean, post_logvar, prior_mean, prior_logvar = \
                self.rssm(obs[t], action[t], hidden, stochastic)
            
            # KL 损失
            kl = self.rssm.compute_kl_loss(
                post_mean, post_logvar, prior_mean, prior_logvar
            )
            kl_losses.append(kl)
            
            # 重建损失
            recon = self.rssm.decode(hidden, post_mean)
            recon_loss = ((recon - obs[t]) ** 2).mean()
            recon_losses.append(recon_loss)
            
            stochastic = post_mean  # 使用均值作为随机状态
        
        # 2. 在潜在空间中"想象"未来
        imagined_states = self.imagine(hidden, stochastic, action)
        
        # 3. 训练 Actor 和 Critic
        actor_loss = self.train_actor(imagined_states)
        critic_loss = self.train_critic(imagined_states, reward)
        
        # 总损失
        total_loss = (
            sum(kl_losses) / T +
            sum(recon_losses) / T +
            actor_loss +
            critic_loss
        )
        
        return total_loss
    
    def imagine(self, initial_hidden, initial_stochastic, initial_action, horizon=15):
        """
        在潜在空间中想象未来轨迹
        
        Returns:
            imagined_states: 想象的状态序列
        """
        hidden = initial_hidden
        stochastic = initial_stochastic
        
        imagined = []
        for _ in range(horizon):
            # 使用 Actor 选择动作
            action = self.actor(hidden, stochastic)
            
            # RSSM 预测下一状态（只用先验，没有观察）
            prior_mean, prior_logvar = self.rssm.get_prior(hidden)
            std = torch.exp(0.5 * prior_logvar)
            stochastic = prior_mean + std * torch.randn_like(prior_mean)
            
            gru_input = torch.cat([hidden, stochastic, action], dim=-1)
            hidden = self.rssm.gru(gru_input, hidden)
            
            imagined.append({
                'hidden': hidden,
                'stochastic': stochastic,
                'action': action
            })
        
        return imagined
```

## 13.4 潜在空间建模

### 为什么使用潜在空间？

1. **降维**：高维观察（如图像）压缩为低维表示
2. **去噪**：过滤无关信息，保留关键特征
3. **平滑**：连续潜在空间便于插值和优化
4. **抽象**：捕捉环境的本质结构

### 变分自编码器（VAE）

VAE 是学习潜在空间的常用方法：

```python
class VAE(nn.Module):
    """变分自编码器"""
    def __init__(self, obs_dim, latent_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    
    def encode(self, obs):
        h = self.encoder(obs)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mean)
        return mean + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, obs):
        mean, logvar = self.encode(obs)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar
    
    def compute_loss(self, obs, recon, mean, logvar, beta=1.0):
        """VAE 损失 = 重建损失 + KL 散度"""
        # 重建损失（MSE）
        recon_loss = ((obs - recon) ** 2).mean()
        
        # KL 散度
        kl_loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).mean()
        
        return recon_loss + beta * kl_loss
```

### β-VAE 与解耦表示

通过调整 KL 散度的权重β，可以学习更解耦的表示：

```python
class BetaVAE(VAE):
    """β-VAE：增加 KL 权重以获得更好的解耦"""
    def __init__(self, obs_dim, latent_dim, hidden_dim, beta=4.0):
        super().__init__(obs_dim, latent_dim, hidden_dim)
        self.beta = beta
    
    def compute_loss(self, obs, recon, mean, logvar):
        return super().compute_loss(obs, recon, mean, logvar, beta=self.beta)
```

### 潜在空间可视化

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_latent_space(latent_codes, labels=None):
    """
    可视化潜在空间
    
    Args:
        latent_codes: 潜在编码 [N, latent_dim]
        labels: 可选的类别标签
    """
    # 使用 PCA 降维到 2D
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_codes)
    
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        scatter = plt.scatter(
            latent_2d[:, 0], latent_2d[:, 1],
            c=labels, cmap='viridis', alpha=0.6
        )
        plt.colorbar(scatter)
    else:
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6)
    
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('潜在空间可视化 (PCA)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return latent_2d


def interpolate_latent(vae, z1, z2, num_steps=10):
    """
    在潜在空间中插值
    
    Args:
        vae: 训练好的 VAE
        z1, z2: 两个潜在向量
        num_steps: 插值步数
    
    Returns:
        插值生成的观察序列
    """
    interpolations = []
    
    for alpha in torch.linspace(0, 1, num_steps):
        z = (1 - alpha) * z1 + alpha * z2
        recon = vae.decode(z)
        interpolations.append(recon)
    
    return torch.stack(interpolations)
```

## 13.5 基于模型的 RL

### 基于模型 vs 无模型 RL

| 特性 | 基于模型 RL | 无模型 RL |
|------|------------|----------|
| 样本效率 | 高 | 低 |
| 计算复杂度 | 高（需要规划） | 低 |
| 渐近性能 | 可能受限 | 通常更高 |
| 实现难度 | 复杂 | 相对简单 |

### 世界模型 + RL 的两种范式

**范式 1：模型辅助策略学习**
```
收集数据 → 训练世界模型 → 在模型中训练策略 → 部署到真实环境
```

**范式 2：基于模型的规划**
```
收集数据 → 训练世界模型 → 在线规划（MPC） → 执行最优动作
```

### DreamerV2/V3 的 Actor-Critic 架构

```python
class ActorNetwork(nn.Module):
    """Dreamer 的 Actor 网络"""
    def __init__(self, hidden_dim, action_dim, action_dist='tanh_normal'):
        super().__init__()
        self.action_dist = action_dist
        
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        if action_dist == 'tanh_normal':
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.logstd_head = nn.Linear(hidden_dim, action_dim)
        else:
            self.logits_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, hidden_state, stochastic_state):
        state = torch.cat([hidden_state, stochastic_state], dim=-1)
        h = self.network(state)
        
        if self.action_dist == 'tanh_normal':
            mean = self.mean_head(h)
            logstd = self.logstd_head(h)
            logstd = torch.clamp(logstd, -5, 2)
            return mean, logstd
        else:
            logits = self.logits_head(h)
            return logits
    
    def get_action(self, hidden_state, stochastic_state, deterministic=False):
        if self.action_dist == 'tanh_normal':
            mean, logstd = self.forward(hidden_state, stochastic_state)
            if deterministic:
                return torch.tanh(mean)
            std = torch.exp(logstd)
            normal = Normal(mean, std)
            action = torch.tanh(normal.rsample())
            return action
        else:
            logits = self.forward(hidden_state, stochastic_state)
            if deterministic:
                return torch.argmax(logits, dim=-1)
            dist = Categorical(logits=logits)
            return dist.sample()


class CriticNetwork(nn.Module):
    """Dreamer 的 Critic 网络（状态值函数）"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, hidden_state, stochastic_state):
        state = torch.cat([hidden_state, stochastic_state], dim=-1)
        return self.network(state)
```

### 模型预测控制（MPC）

```python
def mpc_planning(world_model, current_state, horizon=10, num_samples=100):
    """
    使用 MPC 进行规划
    
    Args:
        world_model: 训练好的世界模型
        current_state: 当前状态
        horizon: 规划视界
        num_samples: 采样的动作序列数量
    
    Returns:
        最优动作序列中的第一个动作
    """
    best_return = -float('inf')
    best_action_sequence = None
    
    for _ in range(num_samples):
        # 随机采样动作序列
        action_sequence = torch.randn(horizon, action_dim)
        
        # 在世界模型中模拟
        state = current_state
        total_reward = 0
        
        for t in range(horizon):
            next_state = world_model.predict(state, action_sequence[t])
            reward = world_model.get_reward(next_state)  # 假设的奖励模型
            total_reward += reward * (0.99 ** t)  # 折扣
            state = next_state
        
        if total_reward > best_return:
            best_return = total_reward
            best_action_sequence = action_sequence
    
    # 执行第一个动作
    return best_action_sequence[0]
```

## 13.6 世界模型与规划

### 规划的基本思想

规划是在执行前"思考"的过程。世界模型使智能体能够：
1. 模拟不同动作序列的结果
2. 评估每个序列的预期回报
3. 选择最优的动作序列

### 蒙特卡洛树搜索（MCTS）与世界模型

```python
class MCTSNode:
    """MCTS 节点"""
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # 到达此节点的动作
        self.children = []
        self.visits = 0
        self.value = 0.0
    
    def uct_score(self, exploration_constant=1.41):
        """UCT 选择分数"""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_constant * torch.sqrt(
            torch.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration


def mcts_with_world_model(world_model, initial_state, num_iterations=100, max_depth=10):
    """
    使用世界模型的 MCTS 规划
    
    Args:
        world_model: 世界模型
        initial_state: 初始状态
        num_iterations: MCTS 迭代次数
        max_depth: 最大搜索深度
    
    Returns:
        最优动作
    """
    root = MCTSNode(initial_state)
    
    for _ in range(num_iterations):
        # 1. 选择：从根节点选择叶子节点
        node = root
        while node.children:
            node = max(node.children, key=lambda n: n.uct_score())
        
        # 2. 扩展：如果未终止，扩展子节点
        if node.visits > 0 and len(node.children) < num_actions:
            # 采样新动作
            action = sample_untried_action(node)
            next_state = world_model.predict(node.state, action)
            child = MCTSNode(next_state, parent=node, action=action)
            node.children.append(child)
            node = child
        
        # 3. 模拟：从当前节点 rollout
        state = node.state
        rollout_reward = 0
        for _ in range(max_depth - get_depth(node)):
            action = world_model.policy(state)  # 默认策略
            reward = world_model.get_reward(state)
            rollout_reward += reward
            state = world_model.predict(state, action)
        
        # 4. 回溯：更新路径上所有节点的统计
        total_reward = rollout_reward
        while node:
            node.visits += 1
            node.value += total_reward
            node = node.parent
    
    # 选择访问次数最多的动作
    best_child = max(root.children, key=lambda n: n.visits)
    return best_child.action
```

### 想象式规划（Imagination-based Planning）

Dreamer 的核心创新是在潜在空间中进行规划：

```python
def imagination_planning(dreamer, initial_hidden, initial_stochastic, horizon=15):
    """
    在潜在空间中想象并选择最优动作
    
    这是 Dreamer 的核心：Actor 直接在想象的轨迹上优化
    """
    # Dreamer 的 Actor 通过梯度下降优化，而不是搜索
    # 这比 MCTS 更高效
    
    imagined_trajectory = dreamer.imagine(
        initial_hidden, initial_stochastic, horizon=horizon
    )
    
    # 计算想象轨迹的总奖励
    total_value = 0
    for step in imagined_trajectory:
        value = dreamer.critic(step['hidden'], step['stochastic'])
        total_value += value
    
    # Actor 的梯度会最大化这个总价值
    return imagined_trajectory[0]['action']
```

## 13.7 应用案例

### 游戏：Atari 和 Minecraft

**Atari 游戏**：
- DreamerV2 在 Atari 基准测试中超越了许多无模型方法
- 关键优势：样本效率高，200k 步就能达到良好性能

**Minecraft（DreamerV3）**：
- 学习复杂的长期任务（如收集钻石）
- 从像素输入直接学习，无需手工特征
- 能够规划多步骤的任务序列

### 机器人控制

**视觉导航**：
```python
class VisualNavigationAgent:
    """使用世界模型的视觉导航机器人"""
    def __init__(self):
        self.world_model = WorldModel(
            obs_dim=64*64*3,  # 相机图像
            action_dim=2,  # 线速度和角速度
            latent_dim=256
        )
        self.policy = PolicyNetwork()
    
    def navigate_to_goal(self, current_image, goal_description):
        """
        导航到目标位置
        
        使用世界模型预测不同动作序列的结果
        """
        # 编码当前观察
        latent = self.world_model.encode(current_image)
        
        # 在潜在空间中规划
        best_action = None
        best_predicted_outcome = -float('inf')
        
        for action_sequence in self.generate_action_sequences():
            predicted_states = self.world_model.predict_sequence(
                latent, action_sequence
            )
            # 评估是否接近目标
            outcome_score = self.evaluate_goal_progress(
                predicted_states, goal_description
            )
            
            if outcome_score > best_predicted_outcome:
                best_predicted_outcome = outcome_score
                best_action = action_sequence[0]
        
        return best_action
```

**机械臂操作**：
- 学习物体的物理特性（质量、摩擦力）
- 预测抓取和操纵的结果
- 在模拟中训练，迁移到真实机器人

### 自动驾驶

世界模型在自动驾驶中的应用：
1. **预测其他车辆的行为**
2. **模拟不同驾驶决策的后果**
3. **处理罕见但危险的场景**

```python
class AutonomousDrivingWorldModel:
    """自动驾驶世界模型"""
    def __init__(self):
        # 多智能体世界模型
        self.ego_model = EgoVehicleModel()
        self.other_agents_model = OtherAgentsModel()
        self.environment_model = EnvironmentModel()
    
    def predict_future(self, current_state, ego_action, horizon=30):
        """
        预测未来场景
        
        Args:
            current_state: 当前场景状态（所有车辆、行人、道路）
            ego_action: 自车计划动作
            horizon: 预测时间步（如 30 步 = 3 秒）
        
        Returns:
            预测的未来场景序列
        """
        predictions = []
        state = current_state
        
        for t in range(horizon):
            # 预测其他智能体的行为
            other_actions = self.other_agents_model.predict(state)
            
            # 更新场景状态
            next_state = self.environment_model.step(
                state, ego_action, other_actions
            )
            
            predictions.append(next_state)
            state = next_state
            
            # 自车动作可以来自规划器
            ego_action = self.planner.plan(state)
        
        return predictions
    
    def evaluate_safety(self, predicted_scenes):
        """评估预测场景的安全性"""
        for scene in predicted_scenes:
            # 检查碰撞
            if self.check_collision(scene):
                return False
            # 检查交通规则
            if not self.check_traffic_rules(scene):
                return False
        return True
```

## 13.8 代码实现示例

### 完整的世界模型训练循环

```python
"""
world_model_training.py
完整的世界模型训练示例
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from collections import deque

class SimpleWorldModel(nn.Module):
    """
    简化的世界模型用于教学
    包含：编码器、动态模型、解码器
    """
    def __init__(self, obs_dim, action_dim, latent_dim, hidden_dim):
        super().__init__()
        
        # 编码器：观察 → 潜在状态
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 动态模型：(潜在状态，动作) → 下一潜在状态
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 解码器：潜在状态 → 观察重建
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        # 奖励预测
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def encode(self, obs):
        return self.encoder(obs)
    
    def predict_next_latent(self, latent, action):
        x = torch.cat([latent, action], dim=-1)
        return self.dynamics(x)
    
    def decode(self, latent):
        return self.decoder(latent)
    
    def predict_reward(self, latent, action):
        x = torch.cat([latent, action], dim=-1)
        return self.reward_head(x)
    
    def forward(self, obs, action):
        """单步前向传播"""
        latent = self.encode(obs)
        next_latent = self.predict_next_latent(latent, action)
        recon = self.decode(next_latent)
        reward = self.predict_reward(latent, action)
        return recon, reward, latent, next_latent


class WorldModelTrainer:
    """世界模型训练器"""
    def __init__(self, config):
        self.model = SimpleWorldModel(
            obs_dim=config['obs_dim'],
            action_dim=config['action_dim'],
            latent_dim=config['latent_dim'],
            hidden_dim=config['hidden_dim']
        )
        
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config['learning_rate']
        )
        
        self.replay_buffer = deque(maxlen=config['buffer_size'])
        self.config = config
    
    def add_experience(self, obs, action, next_obs, reward, done):
        """添加经验到回放缓冲区"""
        self.replay_buffer.append({
            'obs': obs,
            'action': action,
            'next_obs': next_obs,
            'reward': reward,
            'done': done
        })
    
    def sample_batch(self, batch_size):
        """从回放缓冲区采样"""
        indices = np.random.choice(len(self.replay_buffer), batch_size)
        batch = {
            'obs': torch.FloatTensor([self.replay_buffer[i]['obs'] for i in indices]),
            'action': torch.FloatTensor([self.replay_buffer[i]['action'] for i in indices]),
            'next_obs': torch.FloatTensor([self.replay_buffer[i]['next_obs'] for i in indices]),
            'reward': torch.FloatTensor([self.replay_buffer[i]['reward'] for i in indices]),
        }
        return batch
    
    def train_step(self, batch_size=32):
        """单步训练"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        batch = self.sample_batch(batch_size)
        
        self.optimizer.zero_grad()
        
        # 前向传播
        recon, reward_pred, latent, next_latent = self.model(
            batch['obs'], batch['action']
        )
        
        # 编码下一观察
        next_latent_target = self.model.encode(batch['next_obs'])
        
        # 损失计算
        # 1. 重建损失
        recon_loss = nn.MSELoss()(recon, batch['next_obs'])
        
        # 2. 动态模型损失（潜在空间预测）
        dynamics_loss = nn.MSELoss()(next_latent, next_latent_target)
        
        # 3. 奖励预测损失
        reward_loss = nn.MSELoss()(reward_pred.squeeze(), batch['reward'])
        
        # 总损失
        total_loss = recon_loss + dynamics_loss + 0.1 * reward_loss
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'dynamics_loss': dynamics_loss.item(),
            'reward_loss': reward_loss.item()
        }
    
    def train(self, num_steps, batch_size=32, log_interval=100):
        """训练循环"""
        for step in range(num_steps):
            metrics = self.train_step(batch_size)
            
            if metrics and step % log_interval == 0:
                print(f"Step {step}:")
                print(f"  Total Loss: {metrics['total_loss']:.4f}")
                print(f"  Recon Loss: {metrics['recon_loss']:.4f}")
                print(f"  Dynamics Loss: {metrics['dynamics_loss']:.4f}")
                print(f"  Reward Loss: {metrics['reward_loss']:.4f}")


# 使用示例
if __name__ == '__main__':
    config = {
        'obs_dim': 10,
        'action_dim': 4,
        'latent_dim': 32,
        'hidden_dim': 128,
        'learning_rate': 1e-3,
        'buffer_size': 10000
    }
    
    trainer = WorldModelTrainer(config)
    
    # 模拟收集一些数据
    for _ in range(1000):
        obs = np.random.randn(config['obs_dim'])
        action = np.random.randn(config['action_dim'])
        next_obs = obs + 0.1 * action + np.random.randn(config['obs_dim']) * 0.01
        reward = np.random.randn(1)
        trainer.add_experience(obs, action, next_obs, reward, done=False)
    
    # 训练
    trainer.train(num_steps=1000, batch_size=32)
```

### 潜在空间可视化完整示例

```python
"""
latent_visualization.py
潜在空间可视化和分析工具
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def analyze_latent_space(model, data_loader, device='cpu'):
    """
    分析训练好的世界模型的潜在空间
    
    Args:
        model: 训练好的世界模型
        data_loader: 数据加载器
        device: 计算设备
    
    Returns:
        latent_codes: 所有潜在编码
        recon_errors: 重建误差
    """
    model.eval()
    latent_codes = []
    recon_errors = []
    
    with torch.no_grad():
        for batch in data_loader:
            obs = batch['obs'].to(device)
            
            # 编码
            latent = model.encode(obs)
            latent_codes.append(latent.cpu().numpy())
            
            # 重建
            recon = model.decode(latent)
            error = ((obs - recon) ** 2).mean(dim=1)
            recon_errors.append(error.cpu().numpy())
    
    latent_codes = np.concatenate(latent_codes, axis=0)
    recon_errors = np.concatenate(recon_errors, axis=0)
    
    return latent_codes, recon_errors


def plot_latent_distribution(latent_codes, title='潜在空间分布'):
    """绘制潜在空间的分布"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 每个维度的直方图
    for i, ax in enumerate(axes[0]):
        if i < latent_codes.shape[1]:
            ax.hist(latent_codes[:, i], bins=50, alpha=0.7)
            ax.set_xlabel(f'维度 {i}')
            ax.set_ylabel('频率')
    
    # 2. 维度间的相关性热力图
    if latent_codes.shape[1] <= 10:
        corr = np.corrcoef(latent_codes.T)
        im = axes[0, 1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, 1].set_title('维度相关性')
        plt.colorbar(im, ax=axes[0, 1])
    
    # 3. PCA 降维可视化
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_codes)
    axes[1, 0].scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.5)
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[1, 0].set_title('PCA 降维')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. t-SNE 可视化
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    latent_tsne = tsne.fit_transform(latent_codes[:1000])  # 限制样本数
    axes[1, 1].scatter(latent_tsne[:, 0], latent_tsne[:, 1], alpha=0.5)
    axes[1, 1].set_xlabel('t-SNE 1')
    axes[1, 1].set_ylabel('t-SNE 2')
    axes[1, 1].set_title('t-SNE 降维')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_latent_interpolation(model, obs1, obs2, num_steps=10):
    """
    可视化潜在空间插值
    
    Args:
        model: 训练好的世界模型
        obs1, obs2: 两个观察样本
        num_steps: 插值步数
    """
    model.eval()
    
    with torch.no_grad():
        # 编码
        z1 = model.encode(obs1.unsqueeze(0))
        z2 = model.encode(obs2.unsqueeze(0))
        
        # 插值
        interpolations = []
        for alpha in torch.linspace(0, 1, num_steps):
            z = (1 - alpha) * z1 + alpha * z2
            recon = model.decode(z)
            interpolations.append(recon.squeeze())
        
        # 可视化
        fig, axes = plt.subplots(2, num_steps, figsize=(20, 4))
        
        for i, recon in enumerate(interpolations):
            # 这里假设是图像数据，需要根据实际情况调整
            if len(recon.shape) == 1:
                axes[0, i].plot(recon.cpu().numpy())
            else:
                axes[0, i].imshow(recon.cpu().numpy())
            axes[0, i].set_title(f'α={alpha.item():.2f}')
            axes[0, i].axis('off')
        
        # 显示原始图像
        axes[1, 0].imshow(obs1.cpu().numpy())
        axes[1, 0].set_title('原始 1')
        axes[1, 0].axis('off')
        axes[1, -1].imshow(obs2.cpu().numpy())
        axes[1, -1].set_title('原始 2')
        axes[1, -1].axis('off')
        
        plt.tight_layout()
        plt.show()
```

## 13.9 参考文献

### 核心论文

1. **World Models** (Ha & Schmidhuber, 2018)
   - 首次提出"世界模型"概念
   - 使用 VAE + RNN + Controller 架构
   - 在赛车游戏中展示强大能力

2. **Dream to Control** (Hafner et al., 2020)
   - DreamerV1：基于 RSSM 的世界模型
   - 在潜在空间中进行"想象"训练
   - Atari 基准测试上的突破

3. **DreamerV2** (Hafner et al., 2021)
   - 改进的架构和训练方法
   - 在 Atari 上超越人类水平
   - 更高效的样本利用

4. **DreamerV3** (Hafner et al., 2023)
   - 通用且强大的世界模型
   - 在 Minecraft 等复杂任务上的应用
   - 跨领域的泛化能力

### 相关研究

5. **SimPLe** (Kaiser et al., 2019)
   - 使用视频预测模型进行基于模型的 RL
   - 在 Atari 上的早期成功案例

6. **MuZero** (Schrittwieser et al., 2019)
   - DeepMind 的基于模型的规划算法
   - 在围棋、象棋等游戏中的突破
   - 学习隐式的世界模型

7. **PlaNet** (Hafner et al., 2019)
   - Dreamer 的前身
   - 纯基于模型的 RL 方法

### 综述与教程

8. **"Model-Based Reinforcement Learning: A Survey"** (Moerland et al., 2022)
   - 全面的基于模型 RL 综述
   - 涵盖各种方法和基准

9. **"World Models" 教程** (Various)
   - 实践导向的实现教程
   - 代码示例和最佳实践

---

*第 13 章完。下一章将学习视频预测技术。*
