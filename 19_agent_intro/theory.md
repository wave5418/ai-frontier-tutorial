# 第 19 章 Agent 基础

## 1. Agent 定义与发展历程

### 1.1 什么是 Agent

**Agent（智能体）** 是指能够感知环境、进行推理决策并执行行动以实现目标的自主系统。在 AI 领域，Agent 特指基于大语言模型（LLM）构建的、能够独立完成任务的智能系统。

**核心特征：**
- **自主性（Autonomy）**：能够在没有人类干预的情况下独立行动
- **感知能力（Perception）**：能够接收和理解来自环境的信息
- **决策能力（Decision Making）**：能够基于目标和当前状态做出决策
- **行动能力（Action）**：能够执行操作来改变环境状态
- **目标导向（Goal-Oriented）**：行为始终围绕实现特定目标

### 1.2 发展历程

| 时期 | 阶段 | 代表工作 |
|------|------|----------|
| 1950s-1980s | 经典 AI Agent | SHRDLU、SOAR 架构 |
| 1990s-2010s | 强化学习 Agent | AlphaGo、DQN |
| 2020s | LLM-based Agent | ReAct、AutoGPT、LangChain |
| 2023-至今 | 多模态 Agent | GPT-4V、多模态规划 |

**关键里程碑：**
- **2022 年**：ChatGPT 发布，展示 LLM 的对话和推理能力
- **2023 年**：ReAct 框架提出，统一推理与行动
- **2023 年**：AutoGPT 等自主 Agent 系统涌现
- **2024 年**：多 Agent 协作系统成为研究热点

## 2. Agent 核心组件

### 2.1 Planning（规划）

规划是 Agent 将复杂任务分解为可执行步骤的能力。

**规划策略：**
- **任务分解（Task Decomposition）**：将大任务拆分为小步骤
- **思维链（Chain of Thought）**：逐步推理，展示思考过程
- **自我反思（Self-Reflection）**：评估当前进展并调整策略
- **多路径规划（Multi-Path Planning）**：考虑多种解决方案

```python
# 规划示例
任务："帮我准备下周的会议"
规划步骤：
1. 查询日历确认会议时间
2. 准备会议材料
3. 发送会议邀请
4. 预订会议室
5. 准备演示文稿
```

### 2.2 Memory（记忆）

记忆系统使 Agent 能够存储和检索信息，支持长期学习和上下文理解。

**记忆类型：**

| 类型 | 特点 | 用途 |
|------|------|------|
| **短期记忆** | 临时存储，容量有限 | 当前对话上下文、临时变量 |
| **长期记忆** | 持久存储，可检索 | 用户偏好、历史经验、知识库 |
| **程序性记忆** | 技能和程序 | 工具使用方法、操作流程 |
| **情景记忆** | 具体事件记录 | 历史交互、任务执行记录 |

### 2.3 Tool Use（工具使用）

工具使用是 Agent 与外部世界交互的关键能力。

**工具类型：**
- **计算工具**：计算器、代码执行器
- **信息工具**：搜索引擎、数据库查询
- **操作工具**：文件操作、API 调用
- **创作工具**：文本生成、图像处理

**工具调用流程：**
1. 识别需要使用的工具
2. 准备工具参数
3. 执行工具调用
4. 解析工具返回结果
5. 将结果整合到决策中

### 2.4 Action（行动）

行动是 Agent 执行决策、改变环境状态的具体操作。

**行动类型：**
- **内部行动**：思考、规划、记忆更新
- **外部行动**：调用工具、发送消息、执行代码
- **复合行动**：多个原子行动的组合

## 3. LLM-based Agent 架构

### 3.1 基本架构

```
┌─────────────────────────────────────────────────────┐
│                    User Input                        │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              Perception Module                       │
│         (输入处理、意图理解)                          │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              Reasoning Engine                        │
│         (LLM 核心、推理决策)                           │
└─────────────────────┬───────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Planning │ │  Memory  │ │   Tool   │
    │  Module  │ │  System  │ │  Usage   │
    └──────────┘ └──────────┘ └──────────┘
          │           │           │
          └───────────┼───────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              Action Executor                         │
│         (行动执行、结果返回)                          │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                   Output                             │
│         (响应生成、行动反馈)                          │
└─────────────────────────────────────────────────────┘
```

### 3.2 核心模块详解

**感知模块（Perception）：**
- 输入预处理和标准化
- 意图识别和分类
- 上下文提取

**推理引擎（Reasoning Engine）：**
- LLM 作为核心推理单元
- Prompt 工程优化
- 推理策略选择

**规划模块（Planning）：**
- 任务分解器
- 步骤排序
- 依赖关系管理

**记忆系统（Memory System）：**
- 短期记忆管理（对话历史）
- 长期记忆存储（向量数据库）
- 记忆检索和更新

**工具使用（Tool Usage）：**
- 工具注册和发现
- 参数验证
- 结果解析

## 4. 任务分解与规划

### 4.1 任务分解策略

**自上而下分解：**
```
主任务 → 子任务 1 → 步骤 1.1, 1.2, 1.3
       → 子任务 2 → 步骤 2.1, 2.2
       → 子任务 3 → 步骤 3.1, 3.2, 3.3
```

**依赖关系分析：**
- **串行依赖**：步骤 B 必须在步骤 A 完成后执行
- **并行依赖**：步骤 A 和 B 可以同时进行
- **条件依赖**：步骤 C 仅在条件 X 满足时执行

### 4.2 规划算法

**1. 线性规划（Linear Planning）**
```python
def linear_plan(task):
    steps = decompose(task)
    for step in steps:
        execute(step)
    return result
```

**2. 树形规划（Tree Planning）**
```python
def tree_plan(task):
    if is_atomic(task):
        return execute(task)
    subtasks = decompose(task)
    results = [tree_plan(sub) for sub in subtasks]
    return combine(results)
```

**3. 图规划（Graph Planning）**
- 构建任务依赖图
- 拓扑排序确定执行顺序
- 支持并行执行

### 4.3 规划优化

- **启发式搜索**：使用启发函数指导规划
- **动态重规划**：根据执行反馈调整计划
- **经验学习**：从历史成功规划中学习

## 5. 记忆机制

### 5.1 短期记忆（Short-term Memory）

**特点：**
- 存储当前会话的上下文
- 容量有限（受 LLM 上下文窗口限制）
- 快速访问

**实现方式：**
```python
class ConversationBuffer:
    def __init__(self, max_length=10):
        self.messages = []
        self.max_length = max_length
    
    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_length:
            self.messages.pop(0)
    
    def get_context(self):
        return self.messages
```

### 5.2 长期记忆（Long-term Memory）

**特点：**
- 持久化存储
- 容量大
- 需要检索机制

**实现方式（向量存储）：**
```python
class VectorStore:
    def __init__(self, embedding_model):
        self.embeddings = embedding_model
        self.store = {}
    
    def add(self, key, content):
        vector = self.embeddings.encode(content)
        self.store[key] = {"vector": vector, "content": content}
    
    def search(self, query, top_k=3):
        query_vector = self.embeddings.encode(query)
        # 计算相似度并返回最相关的 top_k 结果
        return self._similarity_search(query_vector, top_k)
```

### 5.3 记忆管理策略

**1. 记忆压缩**
- 摘要长对话
- 提取关键信息
- 删除冗余内容

**2. 记忆检索**
- 基于相似度检索
- 基于关键词检索
- 混合检索策略

**3. 记忆更新**
- 增量更新
- 版本控制
- 冲突解决

## 6. 工具使用基础

### 6.1 工具定义

```python
from abc import ABC, abstractmethod

class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> dict:
        """参数定义"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> any:
        """执行工具"""
        pass
```

### 6.2 工具注册与发现

```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}
    
    def register(self, tool: Tool):
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Tool:
        return self.tools.get(name)
    
    def list_tools(self) -> list:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools.values()
        ]
```

### 6.3 工具调用流程

```
1. Agent 分析任务需求
2. 选择合适的工具
3. 构造工具调用参数
4. 执行工具调用
5. 解析返回结果
6. 将结果用于后续决策
```

### 6.4 常见工具示例

**计算器工具：**
```python
class CalculatorTool(Tool):
    @property
    def name(self):
        return "calculator"
    
    @property
    def description(self):
        return "执行数学计算"
    
    @property
    def parameters(self):
        return {
            "expression": {"type": "string", "description": "数学表达式"}
        }
    
    def execute(self, expression: str):
        return eval(expression)  # 实际使用需安全检查
```

**搜索工具：**
```python
class SearchTool(Tool):
    @property
    def name(self):
        return "search"
    
    @property
    def description(self):
        return "搜索互联网信息"
    
    @property
    def parameters(self):
        return {
            "query": {"type": "string", "description": "搜索关键词"}
        }
    
    def execute(self, query: str):
        # 调用搜索 API
        return search_results
```

## 7. 代码实现示例

完整 Agent 系统实现请参考 `code/agent.py` 文件。

**核心类：**
- `BaseAgent`：基础 Agent 类
- `Memory`：记忆模块
- `Planner`：规划器
- `Tool`：工具接口

**使用示例：**
```python
# 创建 Agent
agent = BaseAgent()

# 添加记忆
agent.memory.add("user_preference", "喜欢简洁的回答")

# 注册工具
agent.register_tool(CalculatorTool())
agent.register_tool(SearchTool())

# 执行任务
result = agent.run("帮我计算 1+2+3 的结果")
print(result)
```

## 8. 参考文献

1. **ReAct: Synergizing Reasoning and Acting in Language Models**
   - Yao, S., et al. (2023)
   - ICLR 2023
   - [论文链接](https://arxiv.org/abs/2210.03629)

2. **Chain of Thought Prompting Elicits Reasoning in Large Language Models**
   - Wei, J., et al. (2022)
   - NeurIPS 2022
   - [论文链接](https://arxiv.org/abs/2201.11903)

3. **LangChain: Building Applications with LLMs**
   - Harrison Chase
   - [GitHub](https://github.com/langchain-ai/langchain)

4. **AutoGPT: An Experimental Open-Source Application**
   - Significant Gravitas
   - [GitHub](https://github.com/Significant-Gravitas/Auto-GPT)

5. **AgentBench: Evaluating LLMs as Agents**
   - Liu, X., et al. (2023)
   - [论文链接](https://arxiv.org/abs/2308.03688)

6. **A Survey on Large Language Model based Autonomous Agents**
   - Wang, L., et al. (2023)
   - [论文链接](https://arxiv.org/abs/2308.11432)

---

**下一章：第 20 章 ReAct 框架**
