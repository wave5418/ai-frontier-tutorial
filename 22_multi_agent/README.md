# 第 22 章：多 Agent 协作

## 本章内容

多 Agent 系统（Multi-Agent System, MAS）是 AI 前沿技术的重要组成部分，本章深入讲解多 Agent 协作的原理、架构和实现。

## 核心概念

### 1. 多 Agent 系统架构

- **去中心化架构**：Agent 平等通信，无中心节点
- **中心化架构**：有中央控制器协调
- **分层架构**：多层级管理结构

### 2. 通信机制

- **消息传递**：点对点、广播、发布订阅
- **黑板系统**：共享数据空间
- **通信协议**：FIPA-ACL、自定义协议

### 3. 协作模式

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| 串行 | Agent 按顺序处理 | 依赖关系强的任务 |
| 并行 | Agent 同时处理 | 可独立处理的子任务 |
| 分层 | Manager 分配任务 | 复杂大型任务 |

### 4. 竞争与博弈

- 拍卖机制
- 协商谈判
- 博弈论基础（囚徒困境、纳什均衡）

### 5. 共识与协调

- 投票机制
- 分布式共识
- 冲突解决

## 代码结构

```
code/
└── multi_agent.py
    ├── Message           # 消息类
    ├── Agent             # Agent 基类
    ├── MultiAgentSystem  # 多 Agent 系统核心
    ├── CalculatorAgent   # 计算器 Agent
    ├── SearchAgent       # 搜索 Agent
    ├── WriterAgent       # 写作 Agent
    ├── ReviewerAgent     # 审核 Agent
    ├── DebateAgent       # 辩论 Agent
    ├── JudgeAgent        # 裁判 Agent
    ├── ManagerAgent      # 管理 Agent
    ├── SequentialCollaboration  # 串行协作
    ├── ParallelCollaboration    # 并行协作
    └── DebateSystem      # 辩论系统
```

## 快速开始

### 1. 基本多 Agent 系统

```python
from multi_agent import MultiAgentSystem, CalculatorAgent, Message, MessageType

# 创建系统
system = MultiAgentSystem("Demo")

# 注册 Agent
calc = CalculatorAgent()
system.register_agent(calc)

# 发送消息
msg = Message(
    msg_type=MessageType.QUERY,
    sender_id="user",
    receiver_id=calc.agent_id,
    content="100 + 200"
)
system.send_message(msg)
system.run_until_idle()
```

### 2. 串行协作

```python
from multi_agent import SequentialCollaboration

collaboration = SequentialCollaboration(
    system,
    [writer.agent_id, reviewer.agent_id]
)

result = collaboration.execute("主题")
```

### 3. 并行协作

```python
from multi_agent import ParallelCollaboration

collaboration = ParallelCollaboration(
    system,
    ["agent_1", "agent_2", "agent_3"]
)

results = collaboration.execute(task)
```

### 4. 辩论系统

```python
from multi_agent import DebateSystem

debate = DebateSystem("人工智能是否应该取代人类工作")
result = debate.run_debate(rounds=3)
print(result)
```

## 运行示例

```bash
cd code
python multi_agent.py
```

示例输出包含：
- 基本多 Agent 系统演示
- 串行协作演示（内容创作流程）
- 并行协作演示（多 Agent 同时搜索）
- 辩论系统演示
- 自定义 Agent 演示

## 自定义 Agent

创建自己的 Agent 只需继承 `Agent` 基类：

```python
class MyAgent(Agent):
    def __init__(self):
        super().__init__("my_id", "我的 Agent", "描述")
    
    def process_message(self, message: Message) -> Optional[Message]:
        # 处理消息逻辑
        if message.msg_type == MessageType.QUERY:
            result = self.do_something(message.content)
            return message.create_reply(result)
        return None
    
    def get_capabilities(self) -> List[str]:
        return ["能力 1", "能力 2"]
```

## 最佳实践

1. **单一职责**：每个 Agent 专注一个领域
2. **清晰接口**：定义明确的输入输出
3. **错误处理**：优雅处理异常
4. **日志记录**：追踪消息流转
5. **超时控制**：避免无限等待

## 与其他框架对比

| 特性 | 本章实现 | AutoGen | CrewAI | LangGraph |
|------|----------|---------|--------|-----------|
| 易用性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 灵活性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 代码执行 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 学习曲线 | 平缓 | 中等 | 平缓 | 陡峭 |

## 扩展阅读

- [AutoGen 文档](https://microsoft.github.io/autogen/)
- [CrewAI 文档](https://docs.crewai.com/)
- [LangGraph 文档](https://python.langchain.com/docs/langgraph)
- [多 Agent 系统导论](https://www.amazon.com/Multi-Agent-Systems-Introduction-Weiss/dp/026273141X)

---

**提示**：代码中的中文注释便于理解，实际生产环境建议使用英文注释以提高兼容性。
