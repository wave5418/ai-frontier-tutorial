# AI 前沿技术完整教程

**从 Pretraining 到 AGI 的完整技术路线**

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.8+-red.svg)](https://pytorch.org/)

---

## 🎯 教程目标

本教程完整梳理 AI 前沿技术发展路线，从基础理论到代码实现：

```
Pretraining → LLM → 多模态 → 世界模型 → VLA → Agent → RL → AGI
```

---

## 📚 技术路线

### 第一阶段：语言模型基础

| 章节 | 主题 | 理论 | 代码 | 状态 |
|------|------|------|------|------|
| 01 | [预训练语言模型](01_pretraining/) | [理论](01_pretraining/theory.md) | [代码](01_pretraining/code/) | 📝 编写中 |
| 02 | [Transformer 架构](02_transformer/) | [理论](02_transformer/theory.md) | [代码](02_transformer/code/) | ⏳ 待写 |
| 03 | [BERT 与双向编码](03_bert/) | [理论](03_bert/theory.md) | [代码](03_bert/code/) | ⏳ 待写 |
| 04 | [GPT 与自回归生成](04_gpt/) | [理论](04_gpt/theory.md) | [代码](04_gpt/code/) | ⏳ 待写 |

### 第二阶段：大语言模型（LLM）

| 章节 | 主题 | 理论 | 代码 | 状态 |
|------|------|------|------|------|
| 05 | [LLM 架构演进](05_llm_architecture/) | [理论](05_llm_architecture/theory.md) | [代码](05_llm_architecture/code/) | ⏳ 待写 |
| 06 | [指令微调与对齐](06_instruction_tuning/) | [理论](06_instruction_tuning/theory.md) | [代码](06_instruction_tuning/code/) | ⏳ 待写 |
| 07 | [RLHF 与人类对齐](07_rlhf/) | [理论](07_rlhf/theory.md) | [代码](07_rlhf/code/) | ⏳ 待写 |
| 08 | [高效微调技术](08_peft/) | [理论](08_peft/theory.md) | [代码](08_peft/code/) | ⏳ 待写 |

### 第三阶段：多模态模型

| 章节 | 主题 | 理论 | 代码 | 状态 |
|------|------|------|------|------|
| 09 | [CLIP 与图文对比学习](09_clip/) | [理论](09_clip/theory.md) | [代码](09_clip/code/) | ⏳ 待写 |
| 10 | [扩散模型与 Stable Diffusion](10_diffusion/) | [理论](10_diffusion/theory.md) | [代码](10_diffusion/code/) | ⏳ 待写 |
| 11 | [多模态大模型 (LMM)](11_lmm/) | [理论](11_lmm/theory.md) | [代码](11_lmm/code/) | ⏳ 待写 |
| 12 | [视觉语言模型 (VLM)](12_vlm/) | [理论](12_vlm/theory.md) | [代码](12_vlm/code/) | ⏳ 待写 |

### 第四阶段：世界模型

| 章节 | 主题 | 理论 | 代码 | 状态 |
|------|------|------|------|------|
| 13 | [世界模型基础](13_world_model/) | [理论](13_world_model/theory.md) | [代码](13_world_model/code/) | ⏳ 待写 |
| 14 | [JeSS 与视频预测](14_video_prediction/) | [理论](14_video_prediction/theory.md) | [代码](14_video_prediction/code/) | ⏳ 待写 |
| 15 | [Genie 与环境建模](15_genie/) | [理论](15_genie/theory.md) | [代码](15_genie/code/) | ⏳ 待写 |

### 第五阶段：VLA (Vision-Language-Action)

| 章节 | 主题 | 理论 | 代码 | 状态 |
|------|------|------|------|------|
| 16 | [VLA 基础架构](16_vla_intro/) | [理论](16_vla_intro/theory.md) | [代码](16_vla_intro/code/) | ⏳ 待写 |
| 17 | [RT-2 与机器人学习](17_rt2/) | [理论](17_rt2/theory.md) | [代码](17_rt2/code/) | ⏳ 待写 |
| 18 | [OpenVLA 与开源方案](18_openvla/) | [理论](18_openvla/theory.md) | [代码](18_openvla/code/) | ⏳ 待写 |

### 第六阶段：Agent 系统

| 章节 | 主题 | 理论 | 代码 | 状态 |
|------|------|------|------|------|
| 19 | [Agent 基础架构](19_agent_intro/) | [理论](19_agent_intro/theory.md) | [代码](19_agent_intro/code/) | ⏳ 待写 |
| 20 | [ReAct 与推理行动](20_react/) | [理论](20_react/theory.md) | [代码](20_react/code/) | ⏳ 待写 |
| 21 | [工具使用与 Function Calling](21_tool_use/) | [理论](21_tool_use/theory.md) | [代码](21_tool_use/code/) | ⏳ 待写 |
| 22 | [多 Agent 协作](22_multi_agent/) | [理论](22_multi_agent/theory.md) | [代码](22_multi_agent/code/) | ⏳ 待写 |

### 第七阶段：强化学习

| 章节 | 主题 | 理论 | 代码 | 状态 |
|------|------|------|------|------|
| 23 | [RL 基础](23_rl_basics/) | [理论](23_rl_basics/theory.md) | [代码](23_rl_basics/code/) | ⏳ 待写 |
| 24 | [PPO 与策略梯度](24_ppo/) | [理论](24_ppo/theory.md) | [代码](24_ppo/code/) | ⏳ 待写 |
| 25 | [DPO 与直接偏好优化](25_dpo/) | [理论](25_dpo/theory.md) | [代码](25_dpo/code/) | ⏳ 待写 |
| 26 | [RLHF 实战](26_rlhf_practice/) | [理论](26_rlhf_practice/theory.md) | [代码](26_rlhf_practice/code/) | ⏳ 待写 |

### 第八阶段：AGI 展望

| 章节 | 主题 | 理论 | 代码 | 状态 |
|------|------|------|------|------|
| 27 | [AGI 路径与挑战](27_agi_path/) | [理论](27_agi_path/theory.md) | - | ⏳ 待写 |
| 28 | [Scaling Law 与未来](28_scaling/) | [理论](28_scaling/theory.md) | - | ⏳ 待写 |
| 29 | [具身智能与机器人](29_embodied/) | [理论](29_embodied/theory.md) | [代码](29_embodied/code/) | ⏳ 待写 |
| 30 | [总结与展望](30_conclusion/) | [理论](30_conclusion/theory.md) | - | ⏳ 待写 |

---

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/wave5418/ai-frontier-tutorial.git
cd ai-frontier-tutorial

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 学习路径

1. **零基础**: 从 [01 预训练语言模型](01_pretraining/) 开始
2. **有基础**: 直接跳到感兴趣的章节
3. **实战派**: 每个章节都有代码实现

---

## 📖 核心概念

### Pretraining（预训练）
在大规模无标注数据上训练语言模型，学习语言的基本规律。

### LLM（大语言模型）
基于 Transformer 的超大规模语言模型，具备涌现能力。

### 多模态
同时处理文本、图像、音频等多种模态信息。

### 世界模型
学习物理世界的运行规律，能够预测未来状态。

### VLA（视觉 - 语言 - 动作）
将视觉理解、语言推理和物理动作结合，控制机器人完成任务。

### Agent（智能体）
能够自主规划、使用工具、完成复杂任务的 AI 系统。

### RL（强化学习）
通过与环境交互学习最优策略，用于对齐和优化。

---

## 🛠️ 代码实现

每个章节包含：
- **理论讲解**: Markdown 文档
- **代码实现**: Python + PyTorch
- **实战案例**: 可运行的示例
- **参考文献**: 经典论文链接

---

## 📝 更新计划

| 时间 | 内容 |
|------|------|
| Week 1-2 | 01-04: 语言模型基础 |
| Week 3-4 | 05-08: 大语言模型 |
| Week 5-6 | 09-12: 多模态模型 |
| Week 7-8 | 13-15: 世界模型 |
| Week 9-10 | 16-18: VLA |
| Week 11-12 | 19-22: Agent |
| Week 13-14 | 23-26: 强化学习 |
| Week 15-16 | 27-30: AGI 展望 |

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

MIT License

---

## 🔗 相关链接

- [GitHub 仓库](https://github.com/wave5418/ai-frontier-tutorial)
- [作者主页](https://github.com/wave5418)

---

*持续更新中... 从 Pretraining 到 AGI 的完整技术路线*
