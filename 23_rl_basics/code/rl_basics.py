#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 23 章 强化学习基础 - 代码实现
包含：MDP 环境、价值迭代、策略迭代、Q-Learning、SARSA、DQN、策略梯度
"""

import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# ============================================================
# 1. MDP 环境类
# ============================================================

class GridWorld:
    """
    简单的网格世界环境
    智能体在网格中移动，目标是到达终点，避开陷阱
    """
    
    def __init__(self, size=5, seed=42):
        """
        初始化网格世界
        
        参数:
            size: 网格大小 (size x size)
            seed: 随机种子
        """
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # 上、下、左、右
        self.action_names = ['上', '下', '左', '右']
        
        # 定义状态转移和奖励
        self.goal = size * size - 1  # 右下角为终点
        self.traps = [size * 2, size * 2 + 1]  # 陷阱位置
        
        # 状态转移概率 P[s'][s][a]
        self.P = self._build_transition_matrix()
        # 奖励函数 R[s][a]
        self.R = self._build_reward_matrix()
        
        self.reset()
    
    def _build_transition_matrix(self):
        """构建状态转移概率矩阵"""
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        for s in range(self.n_states):
            row, col = s // self.size, s % self.size
            
            for a in range(self.n_actions):
                # 计算动作后的新位置
                if a == 0:  # 上
                    new_row, new_col = max(0, row - 1), col
                elif a == 1:  # 下
                    new_row, new_col = min(self.size - 1, row + 1), col
                elif a == 2:  # 左
                    new_row, new_col = row, max(0, col - 1)
                else:  # 右
                    new_row, new_col = row, min(self.size - 1, col + 1)
                
                new_s = new_row * self.size + new_col
                P[s, a, new_s] = 1.0
        
        return P
    
    def _build_reward_matrix(self):
        """构建奖励矩阵"""
        R = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # 计算新状态
                row, col = s // self.size, s % self.size
                if a == 0:
                    new_row, new_col = max(0, row - 1), col
                elif a == 1:
                    new_row, new_col = min(self.size - 1, row + 1), col
                elif a == 2:
                    new_row, new_col = row, max(0, col - 1)
                else:
                    new_row, new_col = row, min(self.size - 1, col + 1)
                
                new_s = new_row * self.size + new_col
                
                # 设置奖励
                if new_s == self.goal:
                    R[s, a] = 100  # 到达终点
                elif new_s in self.traps:
                    R[s, a] = -50  # 掉入陷阱
                else:
                    R[s, a] = -1  # 每步代价
        
        return R
    
    def reset(self):
        """重置环境到初始状态"""
        self.current_state = 0
        self.done = False
        return self.current_state
    
    def step(self, action):
        """
        执行动作
        
        参数:
            action: 动作索引 (0-3)
        
        返回:
            next_state, reward, done, info
        """
        # 根据转移概率采样下一状态
        probs = self.P[self.current_state, action]
        next_state = np.random.choice(self.n_states, p=probs)
        
        # 获取奖励
        reward = self.R[self.current_state, action]
        
        # 检查是否结束
        done = (next_state == self.goal or next_state in self.traps)
        
        self.current_state = next_state
        self.done = done
        
        return next_state, reward, done, {}
    
    def render(self):
        """可视化当前状态"""
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:] = '·'
        
        row, col = self.current_state // self.size, self.current_state % self.size
        grid[row, col] = 'A'  # 智能体
        
        goal_row, goal_col = self.goal // self.size, self.goal % self.size
        grid[goal_row, goal_col] = 'G'  # 终点
        
        for trap in self.traps:
            t_row, t_col = trap // self.size, trap % self.size
            grid[t_row, t_col] = 'X'  # 陷阱
        
        print(grid)


# ============================================================
# 2. 价值迭代 (Value Iteration)
# ============================================================

class ValueIteration:
    """价值迭代算法"""
    
    def __init__(self, env, gamma=0.95, theta=1e-6):
        """
        初始化价值迭代
        
        参数:
            env: MDP 环境
            gamma: 折扣因子
            theta: 收敛阈值
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(env.n_states)
        self.policy = np.zeros(env.n_states, dtype=int)
    
    def iterate(self, max_iterations=1000):
        """
        执行价值迭代
        
        返回:
            迭代次数
        """
        for i in range(max_iterations):
            delta = 0
            
            # 更新每个状态的价值
            for s in range(self.env.n_states):
                v = self.V[s]
                
                # 计算所有动作的价值
                action_values = []
                for a in range(self.env.n_actions):
                    q = sum(self.env.P[s, a, s_next] * 
                           (self.env.R[s, a] + self.gamma * self.V[s_next])
                           for s_next in range(self.env.n_states))
                    action_values.append(q)
                
                # 更新状态价值为最大动作价值
                self.V[s] = max(action_values)
                delta = max(delta, abs(v - self.V[s]))
            
            # 检查收敛
            if delta < self.theta:
                print(f"价值迭代在 {i+1} 次迭代后收敛")
                break
        
        # 提取最优策略
        self._extract_policy()
        return i + 1
    
    def _extract_policy(self):
        """从价值函数提取最优策略"""
        for s in range(self.env.n_states):
            action_values = []
            for a in range(self.env.n_actions):
                q = sum(self.env.P[s, a, s_next] * 
                       (self.env.R[s, a] + self.gamma * self.V[s_next])
                       for s_next in range(self.env.n_states))
                action_values.append(q)
            self.policy[s] = np.argmax(action_values)
    
    def get_policy(self):
        """获取策略"""
        return self.policy


# ============================================================
# 3. 策略迭代 (Policy Iteration)
# ============================================================

class PolicyIteration:
    """策略迭代算法"""
    
    def __init__(self, env, gamma=0.95, theta=1e-6):
        """
        初始化策略迭代
        
        参数:
            env: MDP 环境
            gamma: 折扣因子
            theta: 收敛阈值
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(env.n_states)
        self.policy = np.zeros(env.n_states, dtype=int)
    
    def iterate(self, max_iterations=1000):
        """
        执行策略迭代
        
        返回:
            迭代次数
        """
        # 随机初始化策略
        self.policy = np.random.randint(0, self.env.n_actions, self.env.n_states)
        
        for i in range(max_iterations):
            # 策略评估
            self._policy_evaluation()
            
            # 策略改进
            policy_stable = self._policy_improvement()
            
            if policy_stable:
                print(f"策略迭代在 {i+1} 次迭代后收敛")
                break
        
        return i + 1
    
    def _policy_evaluation(self):
        """评估当前策略的价值函数"""
        while True:
            delta = 0
            
            for s in range(self.env.n_states):
                v = self.V[s]
                
                # 计算当前策略下的价值
                a = self.policy[s]
                self.V[s] = sum(self.env.P[s, a, s_next] * 
                               (self.env.R[s, a] + self.gamma * self.V[s_next])
                               for s_next in range(self.env.n_states))
                
                delta = max(delta, abs(v - self.V[s]))
            
            if delta < self.theta:
                break
    
    def _policy_improvement(self):
        """改进策略"""
        policy_stable = True
        
        for s in range(self.env.n_states):
            old_action = self.policy[s]
            
            # 计算所有动作的价值
            action_values = []
            for a in range(self.env.n_actions):
                q = sum(self.env.P[s, a, s_next] * 
                       (self.env.R[s, a] + self.gamma * self.V[s_next])
                       for s_next in range(self.env.n_states))
                action_values.append(q)
            
            # 选择最优动作
            self.policy[s] = np.argmax(action_values)
            
            if old_action != self.policy[s]:
                policy_stable = False
        
        return policy_stable
    
    def get_policy(self):
        """获取策略"""
        return self.policy


# ============================================================
# 4. Q-Learning
# ============================================================

class QLearning:
    """Q-Learning 算法（异策略）"""
    
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1):
        """
        初始化 Q-Learning
        
        参数:
            env: 环境
            alpha: 学习率
            gamma: 折扣因子
            epsilon: 探索率
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.n_states, env.n_actions))
    
    def choose_action(self, state):
        """ε-贪婪策略选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.env.n_actions - 1)
        else:
            return np.argmax(self.Q[state])
    
    def train(self, n_episodes=1000, max_steps=100):
        """
        训练 Q-Learning 智能体
        
        参数:
            n_episodes: 训练回合数
            max_steps: 每回合最大步数
        
        返回:
            每回合的奖励列表
        """
        rewards_per_episode = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # 选择动作
                action = self.choose_action(state)
                
                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                
                # Q-Learning 更新
                best_next_action = np.argmax(self.Q[next_state])
                td_target = reward + self.gamma * self.Q[next_state, best_next_action]
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error
                
                state = next_state
                
                if done:
                    break
            
            rewards_per_episode.append(total_reward)
            
            # 衰减探索率
            self.epsilon = max(0.01, self.epsilon * 0.999)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                print(f"回合 {episode+1}/{n_episodes}, 平均奖励：{avg_reward:.2f}")
        
        return rewards_per_episode
    
    def get_policy(self):
        """获取贪婪策略"""
        return np.argmax(self.Q, axis=1)


# ============================================================
# 5. SARSA
# ============================================================

class SARSA:
    """SARSA 算法（同策略）"""
    
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1):
        """
        初始化 SARSA
        
        参数:
            env: 环境
            alpha: 学习率
            gamma: 折扣因子
            epsilon: 探索率
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.n_states, env.n_actions))
    
    def choose_action(self, state):
        """ε-贪婪策略选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.env.n_actions - 1)
        else:
            return np.argmax(self.Q[state])
    
    def train(self, n_episodes=1000, max_steps=100):
        """
        训练 SARSA 智能体
        
        参数:
            n_episodes: 训练回合数
            max_steps: 每回合最大步数
        
        返回:
            每回合的奖励列表
        """
        rewards_per_episode = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            total_reward = 0
            
            for step in range(max_steps):
                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                
                # 选择下一动作
                next_action = self.choose_action(next_state)
                
                # SARSA 更新
                td_target = reward + self.gamma * self.Q[next_state, next_action]
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error
                
                state = next_state
                action = next_action
                
                if done:
                    break
            
            rewards_per_episode.append(total_reward)
            
            # 衰减探索率
            self.epsilon = max(0.01, self.epsilon * 0.999)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                print(f"回合 {episode+1}/{n_episodes}, 平均奖励：{avg_reward:.2f}")
        
        return rewards_per_episode
    
    def get_policy(self):
        """获取贪婪策略"""
        return np.argmax(self.Q, axis=1)


# ============================================================
# 6. DQN (简化实现)
# ============================================================

class DQN:
    """深度 Q 网络（简化实现，使用表格近似）"""
    
    def __init__(self, env, alpha=0.001, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995, 
                 buffer_size=10000, batch_size=32):
        """
        初始化 DQN
        
        参数:
            env: 环境
            alpha: 学习率
            gamma: 折扣因子
            epsilon: 初始探索率
            epsilon_min: 最小探索率
            epsilon_decay: 探索率衰减
            buffer_size: 回放缓冲区大小
            batch_size: 批次大小
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        # Q 网络（简化为表格）
        self.Q = np.zeros((env.n_states, env.n_actions))
        self.target_Q = np.zeros((env.n_states, env.n_actions))
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=buffer_size)
        
        # 目标网络更新频率
        self.target_update_freq = 10
        self.step_count = 0
    
    def choose_action(self, state):
        """ε-贪婪策略选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.env.n_actions - 1)
        else:
            return np.argmax(self.Q[state])
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """从回放缓冲区采样并训练"""
        if len(self.memory) < self.batch_size:
            return
        
        # 随机采样
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            # 计算 TD 目标
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.target_Q[next_state])
            
            # 更新 Q 值
            td_error = target - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error
        
        # 更新目标网络
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_Q = self.Q.copy()
    
    def train(self, n_episodes=1000, max_steps=100):
        """
        训练 DQN 智能体
        
        返回:
            每回合的奖励列表
        """
        rewards_per_episode = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # 选择动作
                action = self.choose_action(state)
                
                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                
                # 存储经验
                self.remember(state, action, reward, next_state, done)
                
                # 训练
                self.replay()
                
                state = next_state
                
                if done:
                    break
            
            rewards_per_episode.append(total_reward)
            
            # 衰减探索率
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                print(f"回合 {episode+1}/{n_episodes}, 平均奖励：{avg_reward:.2f}, ε={self.epsilon:.3f}")
        
        return rewards_per_episode
    
    def get_policy(self):
        """获取贪婪策略"""
        return np.argmax(self.Q, axis=1)


# ============================================================
# 7. 策略梯度 (REINFORCE)
# ============================================================

class PolicyGradient:
    """策略梯度算法 (REINFORCE)"""
    
    def __init__(self, env, alpha=0.01, gamma=0.95):
        """
        初始化策略梯度
        
        参数:
            env: 环境
            alpha: 学习率
            gamma: 折扣因子
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        
        # 策略参数 (每个状态下各动作的偏好)
        self.theta = np.zeros((env.n_states, env.n_actions))
    
    def softmax_policy(self, state):
        """
        计算 softmax 策略
        
        参数:
            state: 当前状态
        
        返回:
            动作概率分布
        """
        preferences = self.theta[state]
        # 减去最大值防止数值溢出
        preferences = preferences - np.max(preferences)
        exp_prefs = np.exp(preferences)
        return exp_prefs / np.sum(exp_prefs)
    
    def choose_action(self, state):
        """根据策略采样动作"""
        probs = self.softmax_policy(state)
        return np.random.choice(self.env.n_actions, p=probs)
    
    def train(self, n_episodes=1000, max_steps=100):
        """
        训练策略梯度智能体
        
        返回:
            每回合的奖励列表
        """
        rewards_per_episode = []
        
        for episode in range(n_episodes):
            # 收集轨迹
            states = []
            actions = []
            rewards = []
            
            state = self.env.reset()
            
            for step in range(max_steps):
                # 选择动作
                action = self.choose_action(state)
                
                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                
                # 存储轨迹
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state
                
                if done:
                    break
            
            # 计算回报 G_t
            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.append(G)
            returns = list(reversed(returns))
            
            # 更新策略参数
            for t in range(len(states)):
                state = states[t]
                action = actions[t]
                G_t = returns[t]
                
                # 策略梯度更新
                # ∇θ J(θ) ≈ α * G_t * ∇θ log π(a|s)
                # 对于 softmax: ∇θ log π(a|s) = one_hot(a) - π(·|s)
                probs = self.softmax_policy(state)
                gradient = np.zeros(self.env.n_actions)
                gradient[action] = 1 - probs[action]
                for a in range(self.env.n_actions):
                    if a != action:
                        gradient[a] = -probs[a]
                
                self.theta[state] += self.alpha * G_t * gradient
            
            total_reward = sum(rewards)
            rewards_per_episode.append(total_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                print(f"回合 {episode+1}/{n_episodes}, 平均奖励：{avg_reward:.2f}")
        
        return rewards_per_episode
    
    def get_policy(self):
        """获取贪婪策略"""
        return np.argmax(self.theta, axis=1)


# ============================================================
# 8. 测试与可视化
# ============================================================

def test_algorithms():
    """测试所有强化学习算法"""
    print("=" * 60)
    print("强化学习算法测试")
    print("=" * 60)
    
    # 创建环境
    env = GridWorld(size=5)
    print(f"\n环境：{env.size}x{env.size} 网格世界")
    print(f"状态数：{env.n_states}, 动作数：{env.n_actions}")
    print(f"终点：{env.goal}, 陷阱：{env.traps}")
    
    # 1. 价值迭代
    print("\n" + "=" * 60)
    print("1. 价值迭代 (Value Iteration)")
    print("=" * 60)
    vi = ValueIteration(env, gamma=0.95)
    vi.iterate()
    print(f"最优策略：{vi.get_policy()}")
    
    # 2. 策略迭代
    print("\n" + "=" * 60)
    print("2. 策略迭代 (Policy Iteration)")
    print("=" * 60)
    pi = PolicyIteration(env, gamma=0.95)
    pi.iterate()
    print(f"最优策略：{pi.get_policy()}")
    
    # 3. Q-Learning
    print("\n" + "=" * 60)
    print("3. Q-Learning")
    print("=" * 60)
    ql = QLearning(env, alpha=0.1, gamma=0.95, epsilon=0.1)
    ql_rewards = ql.train(n_episodes=500, max_steps=100)
    print(f"Q-Learning 策略：{ql.get_policy()}")
    
    # 4. SARSA
    print("\n" + "=" * 60)
    print("4. SARSA")
    print("=" * 60)
    sarsa = SARSA(env, alpha=0.1, gamma=0.95, epsilon=0.1)
    sarsa_rewards = sarsa.train(n_episodes=500, max_steps=100)
    print(f"SARSA 策略：{sarsa.get_policy()}")
    
    # 5. DQN
    print("\n" + "=" * 60)
    print("5. DQN (简化版)")
    print("=" * 60)
    dqn = DQN(env, alpha=0.001, gamma=0.95, epsilon=1.0)
    dqn_rewards = dqn.train(n_episodes=500, max_steps=100)
    print(f"DQN 策略：{dqn.get_policy()}")
    
    # 6. 策略梯度
    print("\n" + "=" * 60)
    print("6. 策略梯度 (REINFORCE)")
    print("=" * 60)
    pg = PolicyGradient(env, alpha=0.01, gamma=0.95)
    pg_rewards = pg.train(n_episodes=500, max_steps=100)
    print(f"策略梯度策略：{pg.get_policy()}")
    
    # 可视化结果
    try:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 3, 1)
        plt.plot(ql_rewards)
        plt.title('Q-Learning')
        plt.xlabel('回合')
        plt.ylabel('奖励')
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(sarsa_rewards)
        plt.title('SARSA')
        plt.xlabel('回合')
        plt.ylabel('奖励')
        plt.grid(True)
        
        plt.subplot(2, 3, 3)
        plt.plot(dqn_rewards)
        plt.title('DQN')
        plt.xlabel('回合')
        plt.ylabel('奖励')
        plt.grid(True)
        
        plt.subplot(2, 3, 4)
        plt.plot(pg_rewards)
        plt.title('策略梯度 (REINFORCE)')
        plt.xlabel('回合')
        plt.ylabel('奖励')
        plt.grid(True)
        
        plt.subplot(2, 3, 5)
        plt.plot(np.cumsum(ql_rewards) / np.arange(1, len(ql_rewards) + 1))
        plt.title('Q-Learning 累积平均')
        plt.xlabel('回合')
        plt.ylabel('平均奖励')
        plt.grid(True)
        
        plt.subplot(2, 3, 6)
        plt.plot(np.cumsum(pg_rewards) / np.arange(1, len(pg_rewards) + 1))
        plt.title('策略梯度 累积平均')
        plt.xlabel('回合')
        plt.ylabel('平均奖励')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('rl_comparison.png', dpi=150)
        print("\n可视化结果已保存到 rl_comparison.png")
    except Exception as e:
        print(f"\n可视化失败：{e}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_algorithms()
