# 第 20 章 ReAct 框架

## 1. ReAct 框架原理（Reasoning + Acting）

### 1.1 什么是 ReAct

**ReAct**（Reasoning + Acting）是一种将**推理**（Reasoning）和**行动**（Acting）相结合的语言模型 Agent 框架。该框架由 Yao 等人在 2023 年提出，核心思想是让 LLM 在执行任务时交替进行推理和行动。

**核心公式：**
```
ReAct = Reasoning Traces + Task-Specific Actions
```

### 1.2 设计动机

传统 LLM 方法存在两个主要局限：

**1. 纯推理方法的局限（如 Chain of Thought）**
- 只能进行内部推理，无法与外部环境交互
- 无法获取最新信息或执行实际操作
- 容易产生"幻觉"，缺乏事实核查机制

**2. 纯行动方法的局限**
- 缺乏深度推理能力
- 难以处理复杂的多步骤任务
- 行动选择缺乏理论依据

**ReAct 的解决方案：**
- 将推理和行动交织在一起
- 推理指导行动选择
- 行动结果反馈给推理过程
- 形成闭环的 Thought-Action-Observation 循环

### 1.3 ReAct 的核心思想

```
┌──────────────────────────────────────────────────────────┐
│                    ReAct 循环                             │
│                                                          │
│   ┌─────────┐     ┌─────────┐     ┌─────────────┐       │
│   │ Thought │ ──► │ Action  │ ──► │ Observation │       │
│   │  (思考)  │     │  (行动)  │     │   (观察)    │       │
│   └─────────┘     └─────────┘     └─────────────┘       │
│         ▲                                      │         │
│         │                                      │         │
│         └──────────────────────────────────────┘         │
│                      反馈循环                             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**关键洞察：**
1. **推理帮助行动**：通过思考，Agent 可以更好地规划行动序列
2. **行动增强推理**：行动结果提供新信息，帮助更准确的推理
3. **协同效应**：推理和行动的交替进行产生 1+1>2 的效果

## 2. Thought-Action-Observation 循环

### 2.1 循环详解

**Thought（思考）：**
- Agent 分析当前状态
- 推理下一步应该做什么
- 形成行动计划

**Action（行动）：**
- 执行具体操作
- 调用工具或 API
- 与环境交互

**Observation（观察）：**
- 获取行动结果
- 更新当前状态认知
- 为下一步推理提供信息

### 2.2 循环示例

```
任务：计算 2024 年奥运会举办城市的人口

Thought 1: 我需要先找出 2024 年奥运会的举办城市
Action 1: search("2024 年奥运会举办城市")
Observation 1: 2024 年奥运会在巴黎举办

Thought 2: 现在我需要查询巴黎的人口
Action 2: search("巴黎人口 2024")
Observation 2: 巴黎人口约为 210 万

Thought 3: 我已经获得了所需信息，可以给出答案了
Action 3: finish("2024 年奥运会举办城市巴黎的人口约为 210 万")
Observation 3: 任务完成
```

### 2.3 Prompt 模板

ReAct 使用特殊的 Prompt 模板来引导 LLM 进行推理和行动：

```
你是一个智能助手，请通过思考和行动来解决用户的问题。

你可以使用以下工具：
{tool_descriptions}

请按照以下格式回答：

Thought: 你当前的思考
Action: 要采取的行动（从 {tool_names} 中选择）
Action Input: 行动的输入参数
Observation: 行动的结果
...（重复 Thought/Action/Observation 多次）
Thought: 我现在知道最终答案了
Final Answer: 对用户的最终回答

开始！

问题：{input}
```

## 3. 与 Chain of Thought 对比

### 3.1 Chain of Thought（CoT）

**CoT 的核心：**
- 通过展示推理步骤来提高 LLM 的推理能力
- 格式：`Thought 1 → Thought 2 → ... → Answer`
- 仅在模型内部进行推理

**示例：**
```
问题：小明有 5 个苹果，吃了 2 个，又买了 3 个，现在有几个？

Thought 1: 小明最初有 5 个苹果
Thought 2: 吃了 2 个后，剩下 5 - 2 = 3 个
Thought 3: 又买了 3 个，现在有 3 + 3 = 6 个
Answer: 6 个
```

### 3.2 ReAct vs CoT

| 特性 | Chain of Thought | ReAct |
|------|------------------|-------|
| **推理方式** | 纯内部推理 | 推理 + 行动 |
| **外部交互** | 无 | 有（工具调用） |
| **信息获取** | 依赖训练数据 | 可获取实时信息 |
| **事实核查** | 无 | 通过观察验证 |
| **适用场景** | 数学推理、逻辑题 | 问答、决策、任务执行 |
| **幻觉风险** | 较高 | 较低 |

### 3.3 对比示例

**任务：查询 2024 年诺贝尔文学奖得主**

**CoT 方法：**
```
Thought: 我需要回忆 2024 年诺贝尔文学奖的信息
Thought: 根据我的训练数据...
Answer: [可能给出错误或过时的信息]
```

**ReAct 方法：**
```
Thought: 我需要查询 2024 年诺贝尔文学奖的最新信息
Action: search("2024 年诺贝尔文学奖得主")
Observation: 2024 年诺贝尔文学奖得主是韩国作家韩江
Thought: 我现在知道了答案
Final Answer: 2024 年诺贝尔文学奖得主是韩国作家韩江
```

**优势分析：**
- ReAct 通过实际行动获取最新信息
- 避免了依赖过时训练数据的问题
- 结果更准确、更可靠

## 4. ReAct 应用场景

### 4.1 问答系统（Question Answering）

**特点：**
- 需要获取外部知识
- 问题可能涉及最新事件
- 需要多步骤信息收集

**示例任务：**
```
问题：特斯拉 CEO 埃隆·马斯克在 2024 年的净资产是多少？

Thought: 我需要查询马斯克的最新净资产信息
Action: search("Elon Musk net worth 2024")
Observation: 根据 Forbes 2024 年数据，马斯克净资产约 2500 亿美元
Thought: 我已获得所需信息
Final Answer: 根据 2024 年 Forbes 数据，埃隆·马斯克的净资产约为 2500 亿美元
```

### 4.2 决策支持（Decision Making）

**特点：**
- 需要综合考虑多个因素
- 可能需要查询多种信息
- 需要推理和权衡

**示例任务：**
```
问题：我应该去北京还是上海出差？

Thought: 我需要比较两个城市的天气情况
Action: weather("北京")
Observation: 北京：晴，25°C
Action: weather("上海")
Observation: 上海：多云，28°C
Thought: 两地天气都不错，北京稍微凉爽一些
Thought: 我还需要考虑交通便利性
Action: search("北京上海高铁时间")
Observation: 高铁约 4.5 小时
Thought: 综合考虑，北京天气更好
Final Answer: 建议去北京，天气更凉爽（25°C vs 28°C）
```

### 4.3 任务自动化（Task Automation）

**特点：**
- 涉及多个步骤
- 需要调用不同工具
- 需要状态跟踪

**示例任务：**
```
任务：帮我安排明天的会议

Thought: 我需要先查看明天的日程
Action: check_calendar("tomorrow")
Observation: 明天上午 10-12 点有空
Thought: 现在需要预订会议室
Action: book_room("会议室 A", "2024-03-07 10:00")
Observation: 会议室预订成功
Thought: 最后发送会议邀请
Action: send_invite("team@company.com", "明天 10 点会议")
Observation: 邀请已发送
Thought: 所有步骤完成
Final Answer: 会议已安排：明天上午 10 点，会议室 A，邀请已发送给团队
```

### 4.4 数据分析（Data Analysis）

**特点：**
- 需要处理数值计算
- 可能需要数据查询
- 需要逻辑推理

**示例任务：**
```
问题：计算公司 Q1 的平均月增长率

Thought: 我需要获取 Q1 各月的数据
Action: query_sales("2024-01")
Observation: 1 月销售额：100 万
Action: query_sales("2024-02")
Observation: 2 月销售额：120 万
Action: query_sales("2024-03")
Observation: 3 月销售额：150 万
Thought: 计算月增长率
Action: calculator("(120-100)/100")
Observation: 0.2 (20%)
Action: calculator("(150-120)/120")
Observation: 0.25 (25%)
Action: calculator("(0.2+0.25)/2")
Observation: 0.225 (22.5%)
Thought: 已计算平均增长率
Final Answer: Q1 平均月增长率为 22.5%
```

## 5. 变体与改进

### 5.1 ReAct + 强化学习（ReAct + RL）

**动机：**
- 标准 ReAct 依赖 LLM 的零样本能力
- 通过 RL 可以优化行动策略
- 学习更高效的推理 - 行动模式

**方法：**
```
1. 定义奖励函数
   - 任务完成奖励
   - 步骤效率奖励
   - 准确性奖励

2. 训练策略
   - 使用 PPO 或 DQN
   - 优化 Thought 和 Action 的选择

3. 效果
   - 减少不必要的步骤
   - 提高任务完成率
```

### 5.2 ReAct + 自我反思（ReAct + Self-Reflection）

**动机：**
- 标准 ReAct 可能陷入错误循环
- 添加反思机制可以纠正错误

**实现：**
```
Thought: 我应该搜索 X
Action: search(X)
Observation: 结果不符合预期
Reflection: 我的搜索策略可能有问题
Thought: 让我换一种方式搜索 Y
Action: search(Y)
...
```

### 5.3 ReAct + 规划（ReAct + Planning）

**动机：**
- 复杂任务需要预先规划
- 结合规划可以提高效率

**方法：**
```
1. 先进行高层规划
   Plan: [步骤 1, 步骤 2, 步骤 3]

2. 执行每个步骤时使用 ReAct
   步骤 1: Thought → Action → Observation
   步骤 2: Thought → Action → Observation
   步骤 3: Thought → Action → Observation

3. 根据执行反馈调整规划
```

### 5.4 多 Agent ReAct（Multi-Agent ReAct）

**动机：**
- 复杂任务需要多个专业 Agent 协作
- 不同 Agent 可以负责不同子任务

**架构：**
```
┌─────────────────────────────────────────┐
│           Coordinator Agent             │
│         (协调员 Agent)                   │
└───────────────┬─────────────────────────┘
                │
    ┌───────────┼───────────┐
    │           │           │
    ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Search  │ │Calculate│ │  Write  │
│ Agent   │ │ Agent   │ │ Agent   │
└─────────┘ └─────────┘ └─────────┘
```

### 5.5 其他改进方向

| 改进方向 | 描述 | 效果 |
|----------|------|------|
| **Few-shot ReAct** | 提供示例演示 | 提高任务理解 |
| **Tool Learning** | 学习新工具使用 | 扩展能力边界 |
| **Memory Augmented** | 添加长期记忆 | 支持长程任务 |
| **Human-in-the-loop** | 人工干预机制 | 提高可靠性 |

## 6. 代码实现示例

完整 ReAct Agent 实现请参考 `code/react.py` 文件。

**核心组件：**
- `ReActAgent`：ReAct Agent 主类
- `ThoughtActionObservation` 循环实现
- 工具集成与调用
- 示例任务演示

**使用示例：**
```python
# 创建 ReAct Agent
agent = ReActAgent()

# 注册工具
agent.register_tool(CalculatorTool())
agent.register_tool(SearchTool())

# 执行任务
result = agent.run("计算 2024 除以 4 的结果")
print(result)
```

## 7. 参考文献

1. **ReAct: Synergizing Reasoning and Acting in Language Models**
   - Yao, S., Zhao, J., Yu, D., et al.
   - ICLR 2023
   - [论文链接](https://arxiv.org/abs/2210.03629)
   - [GitHub](https://github.com/ysymyth/ReAct)

2. **Chain of Thought Prompting Elicits Reasoning in Large Language Models**
   - Wei, J., Wang, X., Schuurmans, D., et al.
   - NeurIPS 2022
   - [论文链接](https://arxiv.org/abs/2201.11903)

3. **Toolformer: Language Models Can Teach Themselves to Use Tools**
   - Schick, T., Dwivedi-Yu, J., Dessì, R., et al.
   - NeurIPS 2023
   - [论文链接](https://arxiv.org/abs/2302.04761)

4. **Reflexion: Language Agents with Verbal Reinforcement Learning**
   - Shinn, N., Cassano, F., Gopinath, A., et al.
   - NeurIPS 2023
   - [论文链接](https://arxiv.org/abs/2303.11366)

5. **Language Models Can Teach Themselves to Program Better**
   - Haluptzok, P., Bowers, M., Smith, M.
   - [论文链接](https://arxiv.org/abs/2207.14502)

6. **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**
   - Shen, Y., Song, K., Tan, X., et al.
   - [论文链接](https://arxiv.org/abs/2303.17580)

7. **AutoGPT: An Experimental Open-Source Application**
   - Significant Gravitas
   - [GitHub](https://github.com/Significant-Gravitas/Auto-GPT)

---

**上一章：第 19 章 Agent 基础**
**下一章：继续深入 Agent 高级主题**
