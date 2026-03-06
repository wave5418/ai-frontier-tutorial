# 第 21-22 章：工具使用与多 Agent 协作

本目录包含 AI 前沿技术教程的第 21 章和第 22 章内容。

## 目录结构

```
21_tool_use/          # 第 21 章：工具使用与 Function Calling
├── theory.md         # 理论内容
└── code/
    └── tool_use.py   # 代码实现

22_multi_agent/       # 第 22 章：多 Agent 协作
├── theory.md         # 理论内容
└── code/
    └── multi_agent.py # 代码实现
```

## 第 21 章：工具使用与 Function Calling

### 理论要点

- **Function Calling 原理**：LLM 与外部工具交互的核心机制
- **OpenAI Function Calling API**：标准的工具调用接口
- **工具定义与注册**：如何设计和注册工具
- **工具选择与调用**：LLM 如何选择合适的工具
- **复杂任务分解**：多步骤任务的工具调用链
- **代码解释器与插件**：执行代码和扩展能力
- **主流方案对比**：OpenAI、LangChain、AutoGen 等

### 代码实现

`code/tool_use.py` 包含：

- `FunctionCall` 类：表示一次函数调用
- `ToolSchema` 类：工具的 JSON Schema 定义
- `ToolRegistry` 类：工具注册表
- 示例工具：计算器、搜索、代码执行、天气、时间
- `Agent` 类：完整的 Agent + Tools 实现

### 运行示例

```bash
cd 21_tool_use/code
python tool_use.py
```

## 第 22 章：多 Agent 协作

### 理论要点

- **多 Agent 系统架构**：去中心化、中心化、分层架构
- **Agent 间通信机制**：消息传递、发布订阅、黑板系统
- **协作模式**：串行、并行、分层协作
- **竞争与博弈**：拍卖、协商、博弈论基础
- **共识与协调**：投票、分布式共识、冲突解决
- **应用场景**：辩论、协作编程、客户服务
- **主流框架**：AutoGen、CrewAI、LangGraph

### 代码实现

`code/multi_agent.py` 包含：

- `Message` 类：Agent 间通信消息
- `Agent` 基类：Agent 抽象
- 示例 Agent：计算器、搜索、写作、审核、辩论、裁判
- `MultiAgentSystem` 类：多 Agent 系统核心
- `SequentialCollaboration`：串行协作
- `ParallelCollaboration`：并行协作
- `DebateSystem`：辩论系统示例

### 运行示例

```bash
cd 22_multi_agent/code
python multi_agent.py
```

## 学习目标

完成这两章学习后，你将能够：

1. ✅ 理解 Function Calling 的原理和工作流程
2. ✅ 设计和实现自定义工具
3. ✅ 构建基于工具的 Agent 系统
4. ✅ 理解多 Agent 系统的架构模式
5. ✅ 实现 Agent 间的通信与协作
6. ✅ 应用主流多 Agent 框架

## 关键代码片段

### Function Calling 基本用法

```python
from tool_use import registry, Agent

# 注册工具
@registry.register(
    name="my_tool",
    description="我的工具",
    parameters={...}
)
def my_tool(param1: str) -> str:
    return f"结果：{param1}"

# 创建 Agent
agent = Agent()
response = agent.chat("请帮我...")
```

### 多 Agent 系统基本用法

```python
from multi_agent import MultiAgentSystem, Agent, Message

# 创建系统
system = MultiAgentSystem("MySystem")

# 注册 Agent
system.register_agent(my_agent)

# 发送消息
msg = Message(
    msg_type=MessageType.QUERY,
    sender_id="user",
    receiver_id=my_agent.agent_id,
    content="任务内容"
)
system.send_message(msg)
system.run_until_idle()
```

## 延伸阅读

- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- AutoGen: https://microsoft.github.io/autogen/
- CrewAI: https://docs.crewai.com/
- LangChain Agents: https://python.langchain.com/docs/modules/agents/

---

**作者**: AI Frontier Tutorial  
**日期**: 2026  
**版本**: 1.0
