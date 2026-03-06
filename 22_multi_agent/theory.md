# 第 22 章 多 Agent 协作

## 22.1 多 Agent 系统架构

### 22.1.1 什么是多 Agent 系统

多 Agent 系统（Multi-Agent System, MAS）是由多个智能 Agent 组成的系统，这些 Agent 能够：

- **自主决策**：每个 Agent 有自己的目标和决策能力
- **相互通信**：通过消息传递交换信息
- **协作或竞争**：共同完成任务或相互博弈
- **环境感知**：感知共享环境并做出响应

### 22.1.2 系统架构模式

#### 1. 去中心化架构（Decentralized）

```
Agent A ←→ Agent B
   ↑         ↑
   ↓         ↓
Agent C ←→ Agent D
```

- **特点**：无中心节点，Agent 平等通信
- **优点**：鲁棒性强，无单点故障
- **缺点**：协调困难，可能出现混乱

#### 2. 中心化架构（Centralized）

```
      Controller
     /    |    \
    ↓     ↓     ↓
Agent A  Agent B  Agent C
```

- **特点**：有中央控制器协调
- **优点**：易于管理，协调简单
- **缺点**：单点故障风险

#### 3. 分层架构（Hierarchical）

```
        Manager
       /       \
   Leader A   Leader B
    /   \       /   \
  A1   A2     B1    B2
```

- **特点**：多层级管理结构
- **优点**：职责清晰，扩展性好
- **缺点**：层级过多影响效率

### 22.1.3 核心组件

1. **Agent 个体**：具有感知、决策、行动能力
2. **通信机制**：消息传递、共享内存、黑板系统
3. **协调机制**：任务分配、冲突解决、共识达成
4. **环境**：共享状态、资源、约束条件

## 22.2 Agent 间通信机制

### 22.2.1 消息传递

最基础的通信方式，Agent 通过发送和接收消息进行交互。

```python
class Message:
    sender: str      # 发送者
    receiver: str    # 接收者
    content: Any     # 消息内容
    type: str        # 消息类型（请求、响应、通知等）
    timestamp: float # 时间戳
```

### 22.2.2 通信协议

#### FIPA-ACL（Foundation for Intelligent Physical Agents）

标准化的 Agent 通信语言：

```
(request
  :sender agent-1
  :receiver agent-2
  :content (action get-price item-123)
  :language sl
  :ontology e-commerce
)
```

#### 自定义协议

```python
@dataclass
class AgentMessage:
    msg_type: str  # "request", "response", "inform", "query"
    sender_id: str
    receiver_id: str
    payload: Dict
    conversation_id: str  # 会话 ID，用于追踪对话
    in_reply_to: str = None  # 回复的消息 ID
```

### 22.2.3 通信模式

1. **点对点（Point-to-Point）**：直接发送给特定 Agent
2. **广播（Broadcast）**：发送给所有 Agent
3. **发布 - 订阅（Pub-Sub）**：基于主题的消息分发
4. **黑板（Blackboard）**：共享空间，Agent 读写信息

## 22.3 协作模式

### 22.3.1 串行协作（Sequential）

Agent 按顺序执行任务，后一个 Agent 依赖前一个的输出。

```
用户请求 → Agent A → Agent B → Agent C → 最终结果
         (分析)    (搜索)    (总结)
```

**示例场景**：
- 内容创作：规划 → 写作 → 编辑 → 审核
- 数据分析：清洗 → 分析 → 可视化 → 报告

```python
def sequential_collaboration(task):
    result = task
    for agent in [analyst, researcher, writer, editor]:
        result = agent.process(result)
    return result
```

### 22.3.2 并行协作（Parallel）

多个 Agent 同时处理任务的不同部分，最后整合结果。

```
                → Agent A (部分 1) →
用户请求 → 分发器                   → 整合器 → 结果
                → Agent B (部分 2) →
                → Agent C (部分 3) →
```

**示例场景**：
- 文档翻译：不同章节并行翻译
- 代码审查：多个文件同时审查
- 市场调研：不同地区同时调研

```python
def parallel_collaboration(task, agents):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(agent.process, task) for agent in agents]
        results = [f.result() for f in futures]
    return integrate(results)
```

### 22.3.3 分层协作（Hierarchical）

Manager Agent 分配任务给 Worker Agent，形成层级结构。

```python
class ManagerAgent:
    def delegate(self, task):
        subtasks = self.decompose(task)
        for subtask, worker in zip(subtasks, self.workers):
            worker.assign(subtask)
        return self.aggregate([w.get_result() for w in self.workers])
```

**示例场景**：
- 项目管理：项目经理 → 团队领导 → 执行成员
- 软件开发：架构师 → 模块负责人 → 开发者

### 22.3.4 协作模式对比

| 模式 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| 串行 | 依赖关系强的任务 | 逻辑清晰，易于调试 | 速度慢，瓶颈明显 |
| 并行 | 可独立处理的子任务 | 速度快，效率高 | 需要整合，可能冲突 |
| 分层 | 复杂大型任务 | 职责清晰，易扩展 | 管理开销大 |

## 22.4 竞争与博弈

### 22.4.1 竞争场景

当资源有限或目标冲突时，Agent 之间会产生竞争。

**典型场景**：
- 资源分配：多个 Agent 争夺有限资源
- 任务竞标：多个 Agent 竞争同一任务
- 市场博弈：买卖 Agent 价格谈判

### 22.4.2 博弈论基础

#### 囚徒困境

```
              Agent B
            合作    背叛
Agent A 合作  (3,3)  (0,5)
        背叛  (5,0)  (1,1)
```

#### 纳什均衡

每个 Agent 在给定其他 Agent 策略的情况下，选择最优策略。

### 22.4.3 拍卖机制

```python
class Auction:
    def __init__(self, item):
        self.item = item
        self.bids = {}
    
    def place_bid(self, agent_id, amount):
        self.bids[agent_id] = amount
    
    def get_winner(self):
        if not self.bids:
            return None
        return max(self.bids.items(), key=lambda x: x[1])
```

### 22.4.4 协商与谈判

```python
def negotiate(agent_a, agent_b, issue):
    offer_a = agent_a.propose(issue)
    while not agent_b.accept(offer_a):
        counter = agent_b.counter_offer(offer_a)
        if agent_a.accept(counter):
            return counter
        offer_a = agent_a.adjust(counter)
    return offer_a
```

## 22.5 共识与协调

### 22.5.1 共识问题

多个 Agent 就某个决策达成一致。

**挑战**：
- 信息不对称
- 目标冲突
- 通信延迟
- 恶意 Agent

### 22.5.2 投票机制

```python
def majority_vote(agents, proposal):
    votes = [agent.vote(proposal) for agent in agents]
    yes_count = sum(1 for v in votes if v == True)
    return yes_count > len(agents) / 2
```

### 22.5.3 分布式共识算法

#### Paxos / Raft

用于分布式系统的一致性协议。

#### 基于信誉的共识

```python
class ReputationSystem:
    def __init__(self):
        self.reputation = {}  # agent_id → reputation_score
    
    def update_reputation(self, agent_id, performance):
        # 根据表现更新信誉分
        current = self.reputation.get(agent_id, 0.5)
        self.reputation[agent_id] = current * 0.9 + performance * 0.1
    
    def weighted_vote(self, agents, proposal):
        weighted_votes = sum(
            self.reputation.get(a.id, 0.5) * a.vote(proposal)
            for a in agents
        )
        return weighted_votes > sum(self.reputation.get(a.id, 0.5) for a in agents) / 2
```

### 22.5.4 冲突解决

```python
def resolve_conflict(conflicting_agents, issue):
    # 1. 识别冲突
    positions = {a: a.position(issue) for a in conflicting_agents}
    
    # 2. 寻找共同点
    common_ground = find_common_ground(positions)
    
    # 3. 协商妥协
    compromise = negotiate_compromise(positions, common_ground)
    
    # 4. 达成协议
    return compromise
```

## 22.6 应用场景

### 22.6.1 辩论系统

```
正方 Agent ←→ 裁判 Agent ←→ 反方 Agent
    ↑                           ↑
    └──────→ 观众 Agent ←───────┘
```

**角色设计**：
- **正方/反方**：提出论点、反驳对方
- **裁判**：评估论点质量、判定胜负
- **观众**：提问、投票

**实现要点**：
- 论点生成与评估
- 逻辑一致性检查
- 事实核查

### 22.6.2 协作编程

```
用户需求 → 架构师 Agent → 代码结构
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
模块 A     模块 B     模块 C
    ↓         ↓         ↓
    └─────────┼─────────┘
              ↓
        整合者 Agent → 完整代码
              ↓
        测试 Agent → 测试报告
```

### 22.6.3 客户服务

- **接待 Agent**：初步接待，问题分类
- **专业 Agent**：处理特定领域问题
- **升级 Agent**：处理复杂/投诉问题
- **质检 Agent**：监控服务质量

### 22.6.4 科学研究

- **文献 Agent**：检索相关论文
- **实验 Agent**：设计实验方案
- **分析 Agent**：处理实验数据
- **写作 Agent**：撰写研究报告

## 22.7 主流框架

### 22.7.1 AutoGen（Microsoft）

**特点**：
-  Conversational Agent 为核心
- 支持代码执行
- 灵活的 Agent 编排

```python
from autogen import ConversableAgent, UserProxyAgent

assistant = ConversableAgent(
    name="assistant",
    llm_config={"model": "gpt-4"}
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE"
)

user_proxy.initiate_chat(assistant, message="帮我写一个排序算法")
```

### 22.7.2 CrewAI

**特点**：
- 角色驱动（Role-based）
- 任务导向（Task-oriented）
- 流程编排（Process orchestration）

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role='研究员',
    goal='深入调研主题',
    backstory='你是资深研究专家'
)

writer = Agent(
    role='作家',
    goal='撰写高质量文章',
    backstory='你是专业作家'
)

task = Task(
    description='调研 AI 发展趋势并撰写报告',
    agent=researcher
)

crew = Crew(agents=[researcher, writer], tasks=[task])
result = crew.kickoff()
```

### 22.7.3 LangGraph

**特点**：
- 基于图的 Agent 编排
- 状态管理
- 循环与分支支持

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(State)
workflow.add_node("agent_a", node_a)
workflow.add_node("agent_b", node_b)
workflow.add_edge("agent_a", "agent_b")
workflow.add_edge("agent_b", END)
```

### 22.7.4 框架对比

| 框架 | 核心概念 | 易用性 | 灵活性 | 生态 |
|------|----------|--------|--------|------|
| AutoGen | 对话 Agent | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| CrewAI | 角色 + 任务 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| LangGraph | 状态图 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Camel | 角色对话 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

## 22.8 代码实现示例

### 22.8.1 核心架构

详见 `code/multi_agent.py`，包含：

1. **MultiAgentSystem 类**：系统核心
2. **Agent 基类**：Agent 抽象
3. **Message 类**：通信消息
4. **协作示例**：串行、并行、辩论

### 22.8.2 设计原则

1. **模块化**：每个 Agent 独立可测试
2. **可扩展**：易于添加新 Agent 类型
3. **可观测**：日志、监控、调试支持
4. **容错性**：单个 Agent 失败不影响整体

## 22.9 最佳实践

### 22.9.1 Agent 设计

1. **单一职责**：每个 Agent 专注一个领域
2. **明确接口**：定义清晰的输入输出
3. **状态管理**：谨慎处理内部状态
4. **错误处理**：优雅处理异常情况

### 22.9.2 通信设计

1. **消息格式**：统一、可扩展
2. **超时控制**：避免无限等待
3. **重试机制**：处理临时失败
4. **日志记录**：追踪消息流转

### 22.9.3 协调策略

1. **任务分解**：合理划分子任务
2. **负载均衡**：避免某些 Agent 过载
3. **冲突检测**：及时发现并解决冲突
4. **结果整合**：有效合并各 Agent 输出

### 22.9.4 性能优化

1. **并行执行**：充分利用并发
2. **缓存结果**：避免重复计算
3. **异步通信**：非阻塞消息传递
4. **资源管理**：控制 Agent 数量

## 22.10 挑战与展望

### 22.10.1 当前挑战

1. **协调复杂度**：Agent 数量增加时协调困难
2. **一致性保证**：分布式共识难以达成
3. **安全性**：恶意 Agent 或消息注入
4. **可解释性**：多 Agent 决策过程难以理解

### 22.10.2 未来方向

1. **自组织系统**：Agent 自主形成协作结构
2. **人机协作**：人类与 Agent 混合团队
3. **跨模态协作**：文本、代码、图像多模态 Agent
4. **持续学习**：Agent 从协作中学习改进

## 22.11 参考文献

1. "Multi-Agent Systems: An Introduction" - Weiss, G. (2012)
2. AutoGen Documentation: https://microsoft.github.io/autogen/
3. CrewAI Documentation: https://docs.crewai.com/
4. LangGraph Documentation: https://python.langchain.com/docs/langgraph
5. "Distributed Artificial Intelligence" - Huhns, M.N. (1987)
6. FIPA Specifications: http://www.fipa.org/
7. "Game Theory" - Osborne, M.J. (2004)
