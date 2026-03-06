"""
第 29 章 具身智能与机器人 - 代码实现示例
embodied.py

本文件包含：
1. 简单机器人控制环境
2. 感知 - 决策 - 行动循环实现
3. PPO 强化学习训练示例

运行方式：
    python embodied.py

依赖：
    pip install numpy gymnasium torch stable-baselines3
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


# ============================================================
# 1. 简单机器人控制环境
# ============================================================

class SimpleRobotEnv(gym.Env):
    """
    简单的机器人控制环境
    
    任务：控制机械臂末端到达目标位置
    
    状态空间：
        - 关节角度（3 个关节）
        - 关节角速度
        - 末端执行器位置
        - 目标位置
    
    动作空间：
        - 3 个关节的力矩控制
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 200,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # 机器人参数（简化模型）
        self.n_joints = 3  # 3 个关节
        self.joint_limits = np.array([
            [-np.pi, np.pi],   # 关节 1
            [-np.pi/2, np.pi/2],  # 关节 2
            [-np.pi/2, np.pi/2],  # 关节 3
        ])
        
        # 状态空间
        # [关节角度 (3), 关节角速度 (3), 末端位置 (3), 目标位置 (3)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )
        
        # 动作空间：关节力矩
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_joints,),
            dtype=np.float32
        )
        
        # 初始化状态
        self.joint_angles = np.zeros(self.n_joints)
        self.joint_velocities = np.zeros(self.n_joints)
        self.target_position = self._random_target()
        self.end_effector_pos = self._forward_kinematics(self.joint_angles)
    
    def _random_target(self) -> np.ndarray:
        """生成随机目标位置"""
        # 在工作空间内随机生成目标
        return np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(0.1, 0.8),
        ])
    
    def _forward_kinematics(self, angles: np.ndarray) -> np.ndarray:
        """
        简化的正向运动学计算
        
        假设三连杆机械臂，每段长度 0.3 米
        """
        L1, L2, L3 = 0.3, 0.3, 0.3
        
        # 基座旋转（关节 1）
        base_x = np.cos(angles[0]) * (
            L2 * np.cos(angles[1]) + 
            L3 * np.cos(angles[1] + angles[2])
        )
        base_y = np.sin(angles[0]) * (
            L2 * np.cos(angles[1]) + 
            L3 * np.cos(angles[1] + angles[2])
        )
        base_z = L1 + L2 * np.sin(angles[1]) + L3 * np.sin(angles[1] + angles[2])
        
        return np.array([base_x, base_y, base_z])
    
    def _compute_reward(self) -> float:
        """
        计算奖励函数
        
        奖励组成：
        - 距离奖励：末端到目标的距离（负值）
        - 稀疏奖励：到达目标时额外奖励
        - 平滑奖励：鼓励平滑运动
        - 能量奖励：惩罚过大的力矩
        """
        # 距离奖励（负的距离）
        distance = np.linalg.norm(self.end_effector_pos - self.target_position)
        distance_reward = -distance
        
        # 稀疏奖励（到达目标）
        sparse_reward = 10.0 if distance < 0.05 else 0.0
        
        # 平滑奖励（惩罚速度突变）
        smoothness_reward = -np.sum(np.square(self.joint_velocities)) * 0.1
        
        # 能量奖励（惩罚大力矩）
        # 注意：action 是归一化的力矩
        energy_reward = -np.sum(np.square(self.action_space.sample())) * 0.01
        
        return distance_reward + sparse_reward + smoothness_reward + energy_reward
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 随机初始化关节角度
        self.joint_angles = np.array([
            np.random.uniform(low, high) 
            for low, high in self.joint_limits
        ])
        self.joint_velocities = np.zeros(self.n_joints)
        self.target_position = self._random_target()
        self.end_effector_pos = self._forward_kinematics(self.joint_angles)
        self.current_step = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观测"""
        return np.concatenate([
            self.joint_angles,
            self.joint_velocities,
            self.end_effector_pos,
            self.target_position,
        ]).astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        环境步进
        
        参数：
            action: 关节力矩 [3,]
        
        返回：
            observation: 新观测
            reward: 奖励
            terminated: 是否终止（成功）
            truncated: 是否截断（超时）
            info: 额外信息
        """
        # 限制动作在有效范围内
        action = np.clip(action, -1.0, 1.0)
        
        # 简化的动力学模拟
        # 角加速度 = 力矩 / 惯性（简化为 1）
        angular_acceleration = action * 10.0  # 放大系数
        
        # 更新角速度
        self.joint_velocities += angular_acceleration * 0.02  # dt = 0.02s
        self.joint_velocities *= 0.95  # 阻尼
        
        # 更新关节角度
        self.joint_angles += self.joint_velocities * 0.02
        
        # 限制在关节范围内
        for i in range(self.n_joints):
            low, high = self.joint_limits[i]
            self.joint_angles[i] = np.clip(self.joint_angles[i], low, high)
            # 边界反弹
            if self.joint_angles[i] <= low or self.joint_angles[i] >= high:
                self.joint_velocities[i] *= -0.5
        
        # 更新末端执行器位置
        self.end_effector_pos = self._forward_kinematics(self.joint_angles)
        
        # 计算奖励
        reward = self._compute_reward()
        
        # 检查是否成功
        distance = np.linalg.norm(self.end_effector_pos - self.target_position)
        terminated = distance < 0.05
        
        # 检查是否超时
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # 获取新观测
        observation = self._get_observation()
        
        # 额外信息
        info = {
            "distance": distance,
            "success": terminated,
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """渲染环境（简化版本）"""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}")
            print(f"关节角度：{np.degrees(self.joint_angles)}")
            print(f"末端位置：{self.end_effector_pos}")
            print(f"目标位置：{self.target_position}")
            print(f"距离：{np.linalg.norm(self.end_effector_pos - self.target_position):.4f}")
            print("-" * 40)


# ============================================================
# 2. 感知 - 决策 - 行动循环实现
# ============================================================

class PerceptionModule(nn.Module):
    """
    感知模块：处理原始传感器数据
    
    输入：原始观测（关节状态 + 目标）
    输出：特征表示
    """
    
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DecisionModule(nn.Module):
    """
    决策模块：基于感知特征生成动作
    
    使用 Actor-Critic 架构
    """
    
    def __init__(self, feature_dim: int = 64, action_dim: int = 3):
        super().__init__()
        
        # Actor 网络（策略）
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),  # 输出归一化到 [-1, 1]
        )
        
        # Critic 网络（值函数）
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        action = self.actor(features)
        value = self.critic(features)
        return action, value


class ActionModule:
    """
    行动模块：执行动作并处理物理约束
    """
    
    def __init__(self, env: SimpleRobotEnv):
        self.env = env
        self.action_history = []
    
    def execute(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行动作
        
        包括：
        - 动作平滑（低通滤波）
        - 安全限制检查
        - 实际执行
        """
        # 动作平滑
        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            action = 0.8 * action + 0.2 * last_action
        
        self.action_history.append(action.copy())
        if len(self.action_history) > 10:
            self.action_history.pop(0)
        
        # 执行动作
        return self.env.step(action)


class PerceptionActionLoop:
    """
    完整的感知 - 决策 - 行动循环
    """
    
    def __init__(self, env: SimpleRobotEnv, device: str = "cpu"):
        self.env = env
        self.device = device
        self.device_type = torch.device(device)
        
        # 初始化模块
        self.perception = PerceptionModule().to(self.device_type)
        self.decision = DecisionModule().to(self.device_type)
        self.action = ActionModule(env)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.perception.parameters()) + list(self.decision.parameters()),
            lr=3e-4
        )
    
    def run_episode(self, max_steps: int = 200, render: bool = False) -> float:
        """
        运行一个完整回合
        
        返回：累计奖励
        """
        obs, _ = self.env.reset()
        total_reward = 0.0
        
        for step in range(max_steps):
            # 1. 感知：处理观测
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device_type)
            features = self.perception(obs_tensor)
            
            # 2. 决策：生成动作
            with torch.no_grad():
                action, value = self.decision(features)
            action_np = action.cpu().numpy()[0]
            
            # 3. 行动：执行动作
            next_obs, reward, terminated, truncated, info = self.action.execute(action_np)
            
            total_reward += reward
            
            if render:
                self.env.render()
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        return total_reward


# ============================================================
# 3. PPO 强化学习训练示例
# ============================================================

def train_ppo(
    env: gym.Env,
    total_timesteps: int = 100000,
    verbose: int = 1,
) -> PPO:
    """
    使用 PPO 算法训练机器人控制策略
    
    参数：
        env: 训练环境
        total_timesteps: 总训练步数
        verbose: 日志详细程度
    
    返回：
        训练好的 PPO 模型
    """
    print("=" * 60)
    print("开始 PPO 训练")
    print(f"环境：{env.unwrapped.__class__.__name__}")
    print(f"总步数：{total_timesteps:,}")
    print("=" * 60)
    
    # 创建 PPO 模型
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # 熵系数（鼓励探索）
        vf_coef=0.5,    # 值函数系数
        max_grad_norm=0.5,
        tensorboard_log="./ppo_robot_tensorboard/",
        verbose=verbose,
    )
    
    # 训练
    model.learn(total_timesteps=total_timesteps)
    
    print("=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    return model


def evaluate_model(model: PPO, env: gym.Env, n_episodes: int = 10) -> Dict:
    """
    评估训练好的模型
    
    返回：
        评估统计信息
    """
    print("\n开始模型评估...")
    
    rewards = []
    successes = []
    distances = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        rewards.append(episode_reward)
        successes.append(info.get("success", False))
        distances.append(info.get("distance", float("inf")))
    
    stats = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "success_rate": np.mean(successes),
        "mean_distance": np.mean(distances),
    }
    
    print(f"\n评估结果（{n_episodes} 个回合）:")
    print(f"  平均奖励：{stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  成功率：{stats['success_rate']*100:.1f}%")
    print(f"  平均距离：{stats['mean_distance']:.4f}")
    
    return stats


def demo_training():
    """
    演示完整的训练流程
    """
    print("\n" + "=" * 60)
    print("具身智能机器人控制 - 训练演示")
    print("=" * 60 + "\n")
    
    # 1. 创建环境
    env = SimpleRobotEnv(render_mode=None)
    
    # 2. 检查环境
    print("检查环境合规性...")
    check_env(env, warn=True)
    print("✓ 环境检查通过\n")
    
    # 3. 训练模型
    model = train_ppo(env, total_timesteps=50000, verbose=1)
    
    # 4. 保存模型
    model.save("ppo_robot_model")
    print("\n模型已保存为：ppo_robot_model.zip")
    
    # 5. 评估模型
    eval_stats = evaluate_model(model, env, n_episodes=20)
    
    # 6. 可视化示例
    print("\n" + "=" * 60)
    print("运行可视化演示（5 个回合）")
    print("=" * 60)
    
    for episode in range(5):
        obs, _ = env.reset()
        total_reward = 0.0
        step = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            if terminated or truncated:
                status = "✓ 成功" if info.get("success") else "✗ 超时"
                print(f"回合 {episode+1}: {status} | 步数：{step} | "
                      f"奖励：{total_reward:.2f} | 距离：{info['distance']:.4f}")
                break
    
    return model, env


# ============================================================
# 4. 进阶示例：Sim-to-Real 迁移
# ============================================================

class DomainRandomizedEnv(SimpleRobotEnv):
    """
    领域随机化环境
    
    用于 Sim-to-Real 迁移训练
    在仿真中随机化各种参数以提高泛化能力
    """
    
    def __init__(self, render_mode: Optional[str] = None, randomize: bool = True):
        super().__init__(render_mode)
        self.randomize = randomize
        
        # 随机化参数范围
        self.mass_range = (0.8, 1.2)  # 质量随机化
        self.friction_range = (0.3, 0.7)  # 摩擦随机化
        self.motor_strength_range = (0.9, 1.1)  # 电机强度随机化
        
        # 当前随机化参数
        self.current_params = self._sample_domain_params()
    
    def _sample_domain_params(self) -> Dict:
        """采样领域参数"""
        return {
            "mass_multiplier": np.random.uniform(*self.mass_range),
            "friction_coef": np.random.uniform(*self.friction_range),
            "motor_strength": np.random.uniform(*self.motor_strength_range),
            "observation_noise": np.random.uniform(0.0, 0.05),
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置时重新采样领域参数"""
        if self.randomize:
            self.current_params = self._sample_domain_params()
        
        obs, info = super().reset(seed, options)
        
        # 添加观测噪声
        if self.randomize:
            noise = np.random.normal(0, self.current_params["observation_noise"], obs.shape)
            obs = obs + noise
        
        info["domain_params"] = self.current_params
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """步进时应用随机化参数"""
        if self.randomize:
            # 应用电机强度随机化
            action = action * self.current_params["motor_strength"]
        
        return super().step(action)


def train_with_domain_randomization(total_timesteps: int = 100000):
    """
    使用领域随机化训练模型（Sim-to-Real）
    """
    print("\n" + "=" * 60)
    print("Sim-to-Real 训练 - 领域随机化")
    print("=" * 60 + "\n")
    
    # 创建领域随机化环境
    env = DomainRandomizedEnv(randomize=True)
    
    # 训练
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )
    
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_robot_sim2real")
    
    print("\nSim-to-Real 模型已保存为：ppo_robot_sim2real.zip")
    
    return model


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    # 设置随机种子以保证可复现性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 运行演示
    model, env = demo_training()
    
    # 可选：运行 Sim-to-Real 训练
    # sim2real_model = train_with_domain_randomization(total_timesteps=50000)
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n下一步建议：")
    print("1. 增加训练步数以获得更好的性能")
    print("2. 尝试不同的奖励函数设计")
    print("3. 使用更复杂的机器人模型（如 URDF 加载）")
    print("4. 添加视觉观测（RGB-D 相机）")
    print("5. 尝试 Sim-to-Real 迁移到真实机器人")
    print("\n参考资源：")
    print("- Stable Baselines3 文档：https://stable-baselines3.readthedocs.io/")
    print("- Gymnasium 文档：https://gymnasium.farama.org/")
    print("- PyTorch 教程：https://pytorch.org/tutorials/")
