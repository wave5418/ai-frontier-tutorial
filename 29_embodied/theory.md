# 第 29 章 具身智能与机器人

## 29.1 具身智能定义与背景

### 29.1.1 什么是具身智能

**具身智能（Embodied AI）** 是指智能体通过身体与物理环境进行交互，从而获得认知和智能的能力。与传统 AI 不同，具身智能强调：

- **身体性（Embodiment）**：智能依赖于物理身体及其传感器和执行器
- **环境交互（Environmental Interaction）**：通过与环境的实时交互学习和适应
- **感知 - 行动循环（Perception-Action Loop）**：认知产生于感知和行动的持续循环中

### 29.1.2 发展历程

```
1950s-1980s: 符号主义 AI（离身认知）
    ↓
1990s: 行为主义机器人学（Brooks 的包容式架构）
    ↓
2000s: 发展机器人学（Developmental Robotics）
    ↓
2010s: 深度学习 + 机器人学融合
    ↓
2020s: 大规模具身智能（大模型 + 机器人）
```

### 29.1.3 核心挑战

1. **高维连续控制**：机器人关节空间和任务空间的复杂性
2. **样本效率**：真实世界数据收集成本高昂
3. **泛化能力**：从一个任务/环境迁移到新场景
4. **安全约束**：物理交互中的安全性保证

---

## 29.2 感知 - 决策 - 行动闭环

### 29.2.1 感知（Perception）

机器人通过传感器获取环境信息：

```python
# 传感器类型
sensor_types = {
    "视觉": ["RGB 相机", "深度相机", "事件相机"],
    "触觉": ["力/扭矩传感器", "触觉阵列"],
    "本体感知": ["编码器", "IMU", "关节力矩传感器"],
    "其他": ["激光雷达", "超声波", "麦克风阵列"]
}
```

**关键任务：**
- 目标检测与识别
- 姿态估计（物体/人体）
- 场景理解与语义分割
- 深度估计与 3D 重建

### 29.2.2 决策（Decision）

基于感知信息进行规划和决策：

```
感知输入 → 状态估计 → 任务规划 → 运动规划 → 控制指令
              ↓           ↓           ↓
         世界模型    符号/神经    轨迹优化
```

**决策层次：**
1. **任务级规划**：高层任务分解（如"拿起杯子"）
2. **运动级规划**：路径和轨迹生成
3. **控制级执行**：低层关节力矩/位置控制

### 29.2.3 行动（Action）

通过执行器与环境交互：

```python
# 执行器类型
actuator_types = {
    "电动": ["直流电机", "步进电机", "无刷电机"],
    "液压": ["液压缸", "液压马达"],
    "气动": ["气动肌肉", "气缸"],
    "新型": ["软体执行器", "形状记忆合金"]
}
```

---

## 29.3 机器人学习基础

### 29.3.1 强化学习（Reinforcement Learning）

**核心公式：**
```
状态 s_t → 策略π → 动作 a_t → 环境 → 奖励 r_t + 新状态 s_{t+1}
```

**常用算法：**
- **策略梯度**：REINFORCE, PPO, TRPO
- **值函数**：DQN, DDQN, Rainbow
- **演员 - 评论家**：A3C, SAC, TD3

### 29.3.2 模仿学习（Imitation Learning）

```python
# 行为克隆（Behavioral Cloning）
# 最小化专家动作与策略输出的差异
loss = MSE(π(s), a_expert)

# 逆强化学习（Inverse RL）
# 从专家演示中推断奖励函数
```

### 29.3.3 自监督学习

- **世界模型学习**：预测环境动态
- **技能发现**：无奖励的内在动机学习
- **多任务预训练**：跨任务知识迁移

---

## 29.4 仿真环境

### 29.4.1 Isaac Gym（NVIDIA）

**特点：**
- GPU 并行仿真数千个环境
- 与 PyTorch 原生集成
- 支持机器人、刚体、软体仿真

```python
# Isaac Gym 基本使用示例
import isaacgym
from isaacgym import gymapi, gymtorch
import torch

# 初始化 Gym
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.dt = 1/60

# 创建场景
scene = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
```

### 29.4.2 MuJoCo

**特点：**
- 高精度物理仿真
- 广泛应用于强化学习研究
- 支持接触动力学和摩擦模型

```python
# MuJoCo 使用示例
import mujoco
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

# 仿真步进
mujoco.mj_step(model, data)
print(f"关节位置：{data.qpos}")
```

### 29.4.3 PyBullet

**特点：**
- 开源免费
- Python 接口友好
- 支持 URDF 机器人模型

```python
import pybullet as p
import pybullet_data

# 连接仿真
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 加载机器人
robot_id = p.loadURDF("kuka.urdf")
```

### 29.4.4 其他仿真平台

| 平台 | 特点 | 适用场景 |
|------|------|----------|
| Gazebo | ROS 集成，传感器丰富 | 移动机器人 |
| Webots | 教育友好，多语言支持 | 教学与研究 |
| CoppeliaSim | 可视化强，脚本灵活 | 快速原型 |
| Habitat | 室内导航专用 | 家庭机器人 |

---

## 29.5 Sim-to-Real 迁移

### 29.5.1 现实差距（Reality Gap）

仿真与现实之间的差异来源：

1. **模型不准确**：质量、摩擦、刚度参数误差
2. **传感器噪声**：真实传感器存在噪声和延迟
3. **执行器动态**：电机响应、 backlash 等未建模
4. **环境变化**：光照、纹理、物体属性的变化

### 29.5.2 迁移技术

**领域随机化（Domain Randomization）：**
```python
# 在仿真中随机化参数
domain_randomization = {
    "质量": uniform(0.8, 1.2),
    "摩擦": uniform(0.3, 0.7),
    "视觉": randomize_texture_and_lighting(),
    "动力学": randomize_motor_strength()
}
```

**系统辨识（System Identification）：**
- 从真实数据估计模型参数
- 在线自适应调整

**元学习（Meta-Learning）：**
- 学习快速适应新环境的能力
- MAML、RL²等算法

### 29.5.3 成功案例

- **OpenAI Dactyl**：仿真训练的机械手完成魔方还原
- **Boston Dynamics**：仿真辅助的腿式机器人控制
- **Tesla Optimus**：大规模仿真训练人形机器人

---

## 29.6 人形机器人进展

### 29.6.1 Tesla Optimus

**技术特点：**
- 端到端神经网络控制
- 视觉为主的感知系统
- 大规模数据收集（人类演示）

**进展时间线：**
```
2022: 原型机亮相
2023: 行走、抓取演示
2024: 工厂试点应用
2025+: 规模化生产
```

### 29.6.2 Figure AI

**Figure 01/02 特点：**
- 与 OpenAI 合作开发 VLA（Vision-Language-Action）模型
- 端到端语言指令执行
- 商业化部署（BMW 工厂）

### 29.6.3 其他人形机器人

| 公司 | 型号 | 特点 |
|------|------|------|
| Boston Dynamics | Atlas | 液压驱动，高动态运动 |
| Agility Robotics | Digit | 物流场景优化 |
| Unitree | H1 | 开源，高性价比 |
| Fourier | GR-1 | 中国自主研发 |
| Xiaomi | CyberOne | 全栈自研 |

### 29.6.4 技术趋势

1. **大模型赋能**：VLA（Vision-Language-Action）模型
2. **端到端学习**：从像素到动作的直接映射
3. **数据飞轮**：真实部署→数据收集→模型改进
4. **成本控制**：执行器和传感器成本下降

---

## 29.7 代码实现示例

详见 `code/embodied.py`，包含：

1. 简单机器人控制环境
2. 感知 - 决策 - 行动循环实现
3. PPO 强化学习训练示例

---

## 29.8 参考文献

### 经典论文

1. Brooks, R. A. (1991). "Intelligence without representation". *Artificial Intelligence*.
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
3. Levine, S., et al. (2016). "End-to-End Training of Deep Visuomotor Policies". *JMLR*.

### 近期工作

4. Akkaya, I., et al. (2019). "Solving Rubik's Cube with a Robot Hand". *arXiv*.
5. Brohan, A., et al. (2023). "RT-2: Vision-Language-Action Models". *arXiv*.
6. Black, K., et al. (2024). "Open X-Embodiment: Robotic Learning Datasets". *ICRA*.

### 开源项目

7. Isaac Gym: https://developer.nvidia.com/isaac-gym
8. MuJoCo: https://github.com/google-deepmind/mujoco
9. Stable Baselines3: https://github.com/DLR-RM/stable-baselines3
10. ROS 2: https://docs.ros.org/

### 学习资源

11. CS285: Deep Reinforcement Learning (UC Berkeley)
12. Robotics: Estimation and Learning (Coursera)
13. 具身智能前沿讲座（各大高校公开课）

---

*本章完*
