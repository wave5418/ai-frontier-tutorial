# -*- coding: utf-8 -*-
"""
第 22 章 多 Agent 协作 - 代码实现

本模块实现了多 Agent 系统的核心组件：
- MultiAgentSystem 类：多 Agent 系统核心
- Agent 基类：Agent 抽象
- Message 类：通信消息
- 协作示例：串行、并行、辩论系统
- 完整可运行示例

作者：AI Frontier Tutorial
日期：2026
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


# ============================================================================
# 1. 消息类 - Agent 间通信的基本单位
# ============================================================================

class MessageType(Enum):
    """消息类型枚举"""
    REQUEST = "request"       # 请求
    RESPONSE = "response"     # 响应
    INFORM = "inform"         # 通知
    QUERY = "query"           # 查询
    COMMAND = "command"       # 命令
    ACK = "ack"               # 确认
    ERROR = "error"           # 错误


@dataclass
class Message:
    """
    Agent 间通信的消息类
    
    Attributes:
        msg_type: 消息类型
        sender_id: 发送者 ID
        receiver_id: 接收者 ID（None 表示广播）
        content: 消息内容
        conversation_id: 会话 ID，用于追踪相关消息
        in_reply_to: 回复的消息 ID
        timestamp: 时间戳
        metadata: 额外元数据
    """
    msg_type: MessageType
    sender_id: str
    content: Any
    receiver_id: str = None
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    in_reply_to: str = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "msg_type": self.msg_type.value,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "content": self.content,
            "conversation_id": self.conversation_id,
            "in_reply_to": self.in_reply_to,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        """从字典创建消息"""
        return cls(
            msg_type=MessageType(data["msg_type"]),
            sender_id=data["sender_id"],
            receiver_id=data.get("receiver_id"),
            content=data["content"],
            conversation_id=data.get("conversation_id", str(uuid.uuid4())),
            in_reply_to=data.get("in_reply_to"),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {})
        )
    
    def create_reply(self, content: Any, msg_type: MessageType = MessageType.RESPONSE) -> 'Message':
        """创建回复消息"""
        return Message(
            msg_type=msg_type,
            sender_id=self.receiver_id or "system",
            receiver_id=self.sender_id,
            content=content,
            conversation_id=self.conversation_id,
            in_reply_to=self.id
        )
    
    @property
    def id(self) -> str:
        """消息唯一 ID"""
        return f"{self.conversation_id}_{self.timestamp}"
    
    def __str__(self) -> str:
        return f"Message({self.msg_type.value}: {self.sender_id} → {self.receiver_id})"


# ============================================================================
# 2. Agent 基类 - 所有 Agent 的抽象
# ============================================================================

class AgentState(Enum):
    """Agent 状态"""
    IDLE = "idle"           # 空闲
    BUSY = "busy"           # 忙碌
    WAITING = "waiting"     # 等待
    ERROR = "error"         # 错误


class Agent(ABC):
    """
    Agent 抽象基类
    
    所有具体的 Agent 都应继承此类并实现相应方法
    
    Attributes:
        agent_id: Agent 唯一标识
        name: Agent 名称
        description: Agent 描述
        state: 当前状态
        message_queue: 待处理消息队列
    """
    
    def __init__(self, agent_id: str, name: str, description: str = ""):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.state = AgentState.IDLE
        self.message_queue: List[Message] = []
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._knowledge: Dict = {}  # Agent 内部知识
        self._memory: List[Message] = []  # 消息历史
    
    @abstractmethod
    def process_message(self, message: Message) -> Optional[Message]:
        """
        处理接收到的消息
        
        Args:
            message: 接收到的消息
            
        Returns:
            回复消息（如果有）
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        获取 Agent 能力列表
        
        Returns:
            能力描述列表
        """
        pass
    
    def receive_message(self, message: Message):
        """接收消息并加入队列"""
        self.message_queue.append(message)
        self._memory.append(message)
        if self.state == AgentState.IDLE:
            self.state = AgentState.BUSY
    
    def send_message(self, message: Message):
        """发送消息（由系统处理实际发送）"""
        # 实际发送由 MultiAgentSystem 处理
        pass
    
    def process_queue(self) -> List[Message]:
        """
        处理消息队列
        
        Returns:
            回复消息列表
        """
        replies = []
        while self.message_queue:
            message = self.message_queue.pop(0)
            try:
                reply = self.process_message(message)
                if reply:
                    replies.append(reply)
            except Exception as e:
                error_msg = Message(
                    msg_type=MessageType.ERROR,
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    content=f"处理错误：{str(e)}",
                    conversation_id=message.conversation_id,
                    in_reply_to=message.id
                )
                replies.append(error_msg)
        
        if not self.message_queue:
            self.state = AgentState.IDLE
        
        return replies
    
    def store_knowledge(self, key: str, value: Any):
        """存储知识"""
        self._knowledge[key] = value
    
    def get_knowledge(self, key: str, default: Any = None) -> Any:
        """获取知识"""
        return self._knowledge.get(key, default)
    
    def __str__(self) -> str:
        return f"Agent({self.name}, {self.state.value})"


# ============================================================================
# 3. 示例 Agent 实现
# ============================================================================

class CalculatorAgent(Agent):
    """计算器 Agent - 执行数学计算"""
    
    def __init__(self):
        super().__init__("calc_001", "计算器 Agent", "执行数学计算")
    
    def process_message(self, message: Message) -> Optional[Message]:
        if message.msg_type == MessageType.QUERY:
            expression = message.content
            try:
                # 简单实现，实际应更安全
                result = eval(str(expression), {"__builtins__": {}}, {})
                return message.create_reply(f"计算结果：{result}")
            except Exception as e:
                return message.create_reply(f"计算错误：{str(e)}")
        return None
    
    def get_capabilities(self) -> List[str]:
        return ["数学计算", "表达式求值"]


class SearchAgent(Agent):
    """搜索 Agent - 模拟网络搜索"""
    
    def __init__(self):
        super().__init__("search_001", "搜索 Agent", "搜索信息")
        self._mock_results = {
            "AI": ["AI 最新进展", "机器学习教程", "深度学习框架"],
            "Python": ["Python 教程", "Python 最佳实践", "Python 库推荐"],
            "天气": ["北京天气", "上海天气", "广州天气"]
        }
    
    def process_message(self, message: Message) -> Optional[Message]:
        if message.msg_type == MessageType.QUERY:
            query = str(message.content)
            # 模拟搜索结果
            results = self._mock_results.get(query[:2], [f"关于'{query}'的结果 1", f"结果 2", f"结果 3"])
            return message.create_reply("\n".join(results))
        return None
    
    def get_capabilities(self) -> List[str]:
        return ["信息搜索", "知识检索"]


class WriterAgent(Agent):
    """写作 Agent - 生成文本内容"""
    
    def __init__(self):
        super().__init__("writer_001", "写作 Agent", "生成文本内容")
        self._style = "正式"
    
    def process_message(self, message: Message) -> Optional[Message]:
        if message.msg_type == MessageType.COMMAND:
            topic = message.content
            # 模拟写作
            content = f"""
# {topic}

## 引言
{topic}是当今重要的话题...

## 主要内容
1. 第一点：详细说明...
2. 第二点：深入分析...
3. 第三点：实践应用...

## 结论
综上所述，{topic}具有重要意义。
"""
            return message.create_reply(content.strip())
        return None
    
    def get_capabilities(self) -> List[str]:
        return ["文章写作", "内容生成", "文档创作"]


class ReviewerAgent(Agent):
    """审核 Agent - 检查和改进内容"""
    
    def __init__(self):
        super().__init__("reviewer_001", "审核 Agent", "内容审核与改进")
    
    def process_message(self, message: Message) -> Optional[Message]:
        if message.msg_type == MessageType.REQUEST:
            content = str(message.content)
            # 模拟审核
            issues = []
            if len(content) < 100:
                issues.append("内容过短")
            if "..." in content:
                issues.append("存在未完成的内容")
            
            if issues:
                return message.create_reply(f"审核意见：{', '.join(issues)}")
            else:
                return message.create_reply("审核通过：内容质量良好")
        return None
    
    def get_capabilities(self) -> List[str]:
        return ["内容审核", "质量检查", "改进建议"]


class DebateAgent(Agent):
    """辩论 Agent - 参与辩论"""
    
    def __init__(self, side: str = "affirmative"):
        """
        Args:
            side: "affirmative"（正方）或 "negative"（反方）
        """
        agent_id = f"debate_{side[:3]}_{uuid.uuid4().hex[:4]}"
        name = f"{'正方' if side == 'affirmative' else '反方'}辩手"
        super().__init__(agent_id, name, f"辩论{side}方")
        self.side = side
        self._arguments = []
    
    def process_message(self, message: Message) -> Optional[Message]:
        if message.msg_type == MessageType.QUERY:
            topic = message.content
            return self._make_argument(topic)
        elif message.msg_type == MessageType.REQUEST:
            # 反驳对方论点
            opponent_arg = message.content
            return self._rebuttal(opponent_arg)
        return None
    
    def _make_argument(self, topic: str) -> Message:
        """构建论点"""
        if self.side == "affirmative":
            arg = f"我方认为{topic}。理由：1) 积极影响显著 2) 符合发展趋势 3) 实践验证有效"
        else:
            arg = f"我方反对{topic}。理由：1) 存在潜在风险 2) 实施难度大 3) 有更好的替代方案"
        
        self._arguments.append(arg)
        return Message(
            msg_type=MessageType.INFORM,
            sender_id=self.agent_id,
            content=arg
        )
    
    def _rebuttal(self, opponent_arg: str) -> Message:
        """反驳对方论点"""
        rebuttals = [
            "对方观点存在逻辑漏洞",
            "该论点缺乏实证支持",
            "对方忽视了关键因素",
            "这种说法过于绝对化"
        ]
        import random
        return Message(
            msg_type=MessageType.RESPONSE,
            sender_id=self.agent_id,
            content=f"反驳：{random.choice(rebuttals)}。我方坚持原有立场。"
        )
    
    def get_capabilities(self) -> List[str]:
        return ["论点构建", "反驳论证", "逻辑分析"]


class JudgeAgent(Agent):
    """裁判 Agent - 评估辩论"""
    
    def __init__(self):
        super().__init__("judge_001", "裁判 Agent", "辩论裁判")
        self._scores = {}
    
    def process_message(self, message: Message) -> Optional[Message]:
        if message.msg_type == MessageType.REQUEST:
            # 评估辩论表现
            debate_record = message.content
            evaluation = self._evaluate(debate_record)
            return message.create_reply(evaluation)
        return None
    
    def _evaluate(self, record: str) -> str:
        """评估辩论记录"""
        # 简单模拟评估
        return """
辩论评估报告：
- 正方得分：85/100
- 反方得分：82/100
- 获胜方：正方
- 点评：双方表现优秀，正方论据更充分
"""
    
    def get_capabilities(self) -> List[str]:
        return ["辩论评估", "表现打分", "胜负判定"]


class ManagerAgent(Agent):
    """管理 Agent - 协调其他 Agent"""
    
    def __init__(self):
        super().__init__("manager_001", "管理 Agent", "任务协调与管理")
        self._workers: List[str] = []
        self._tasks: Dict = {}
    
    def process_message(self, message: Message) -> Optional[Message]:
        if message.msg_type == MessageType.COMMAND:
            task = message.content
            # 分解任务并分配
            return message.create_reply(f"任务已接收，正在分配：{task}")
        return None
    
    def register_worker(self, worker_id: str):
        """注册工作 Agent"""
        self._workers.append(worker_id)
    
    def get_capabilities(self) -> List[str]:
        return ["任务分解", "资源分配", "进度跟踪"]


# ============================================================================
# 4. MultiAgentSystem 类 - 多 Agent 系统核心
# ============================================================================

class CommunicationMode(Enum):
    """通信模式"""
    DIRECT = "direct"           # 直接通信
    BROADCAST = "broadcast"     # 广播
    PUBSUB = "pubsub"           # 发布订阅
    BLACKBOARD = "blackboard"   # 黑板


class MultiAgentSystem:
    """
    多 Agent 系统核心类
    
    功能：
    - Agent 注册与管理
    - 消息路由与分发
    - 协作模式支持
    - 系统监控
    
    Attributes:
        agents: 注册的 Agent 字典
        message_history: 消息历史
        communication_mode: 通信模式
    """
    
    def __init__(self, name: str = "MAS"):
        self.name = name
        self.agents: Dict[str, Agent] = {}
        self.message_history: List[Message] = []
        self.communication_mode = CommunicationMode.DIRECT
        self._subscriptions: Dict[str, Set[str]] = {}  # topic → agent_ids
        self._blackboard: Dict = {}  # 黑板数据
        self._running = False
    
    def register_agent(self, agent: Agent):
        """注册 Agent"""
        self.agents[agent.agent_id] = agent
        print(f"[{self.name}] 注册 Agent: {agent.name} ({agent.agent_id})")
    
    def unregister_agent(self, agent_id: str):
        """注销 Agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            print(f"[{self.name}] 注销 Agent: {agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """获取 Agent"""
        return self.agents.get(agent_id)
    
    def send_message(self, message: Message):
        """
        发送消息
        
        根据通信模式路由消息
        """
        self.message_history.append(message)
        
        if self.communication_mode == CommunicationMode.DIRECT:
            # 直接发送给指定接收者
            if message.receiver_id and message.receiver_id in self.agents:
                self.agents[message.receiver_id].receive_message(message)
        
        elif self.communication_mode == CommunicationMode.BROADCAST:
            # 广播给所有 Agent
            for agent in self.agents.values():
                agent.receive_message(message)
        
        elif self.communication_mode == CommunicationMode.PUBSUB:
            # 发布 - 订阅模式
            topic = message.metadata.get("topic")
            if topic and topic in self._subscriptions:
                for subscriber_id in self._subscriptions[topic]:
                    if subscriber_id in self.agents:
                        self.agents[subscriber_id].receive_message(message)
        
        elif self.communication_mode == CommunicationMode.BLACKBOARD:
            # 黑板模式：写入共享空间
            key = message.metadata.get("key", str(time.time()))
            self._blackboard[key] = message
    
    def subscribe(self, agent_id: str, topic: str):
        """订阅主题"""
        if topic not in self._subscriptions:
            self._subscriptions[topic] = set()
        self._subscriptions[topic].add(agent_id)
    
    def write_blackboard(self, key: str, value: Any):
        """写入黑板"""
        self._blackboard[key] = value
    
    def read_blackboard(self, key: str, default: Any = None) -> Any:
        """读取黑板"""
        return self._blackboard.get(key, default)
    
    def process_all(self) -> List[Message]:
        """
        处理所有 Agent 的消息队列
        
        Returns:
            产生的新消息列表
        """
        new_messages = []
        
        for agent in self.agents.values():
            replies = agent.process_queue()
            for reply in replies:
                self.send_message(reply)
                new_messages.append(reply)
        
        return new_messages
    
    def run_until_idle(self, max_iterations: int = 10) -> List[Message]:
        """
        运行系统直到所有 Agent 空闲
        
        Args:
            max_iterations: 最大迭代次数
            
        Returns:
            所有产生的消息
        """
        all_messages = []
        
        for i in range(max_iterations):
            # 检查是否所有 Agent 都空闲
            if all(agent.state == AgentState.IDLE for agent in self.agents.values()):
                break
            
            # 处理消息
            new_messages = self.process_all()
            all_messages.extend(new_messages)
            
            if not new_messages:
                break
        
        return all_messages
    
    def get_status(self) -> Dict:
        """获取系统状态"""
        return {
            "name": self.name,
            "agent_count": len(self.agents),
            "agents": {
                agent_id: {
                    "name": agent.name,
                    "state": agent.state.value,
                    "queue_size": len(agent.message_queue)
                }
                for agent_id, agent in self.agents.items()
            },
            "message_count": len(self.message_history),
            "communication_mode": self.communication_mode.value
        }
    
    def __str__(self) -> str:
        return f"MultiAgentSystem({self.name}, {len(self.agents)} agents)"


# ============================================================================
# 5. 协作模式实现
# ============================================================================

class SequentialCollaboration:
    """
    串行协作模式
    
    Agent 按顺序处理任务，后一个依赖前一个的输出
    """
    
    def __init__(self, system: MultiAgentSystem, agent_order: List[str]):
        self.system = system
        self.agent_order = agent_order
    
    def execute(self, initial_input: Any) -> Any:
        """
        执行串行协作
        
        Args:
            initial_input: 初始输入
            
        Returns:
            最终输出
        """
        current_data = initial_input
        
        for agent_id in self.agent_order:
            agent = self.system.get_agent(agent_id)
            if not agent:
                continue
            
            # 发送处理请求
            msg = Message(
                msg_type=MessageType.COMMAND,
                sender_id="system",
                receiver_id=agent_id,
                content=current_data
            )
            self.system.send_message(msg)
            
            # 处理并获取结果
            self.system.run_until_idle(max_iterations=2)
            
            # 从 Agent 获取结果（简化处理）
            if agent._memory:
                last_msg = agent._memory[-1]
                if last_msg.content:
                    current_data = last_msg.content
        
        return current_data


class ParallelCollaboration:
    """
    并行协作模式
    
    多个 Agent 同时处理任务的不同部分
    """
    
    def __init__(self, system: MultiAgentSystem, agent_ids: List[str]):
        self.system = system
        self.agent_ids = agent_ids
    
    def execute(self, task: Any, split_func: Callable = None, 
                merge_func: Callable = None) -> Any:
        """
        执行并行协作
        
        Args:
            task: 任务
            split_func: 任务拆分函数
            merge_func: 结果合并函数
            
        Returns:
            合并后的结果
        """
        # 拆分任务（默认均分）
        if split_func:
            subtasks = split_func(task)
        else:
            subtasks = [task] * len(self.agent_ids)
        
        # 发送给各个 Agent
        for agent_id, subtask in zip(self.agent_ids, subtasks):
            msg = Message(
                msg_type=MessageType.COMMAND,
                sender_id="system",
                receiver_id=agent_id,
                content=subtask
            )
            self.system.send_message(msg)
        
        # 等待所有 Agent 完成
        self.system.run_until_idle()
        
        # 收集结果
        results = []
        for agent_id in self.agent_ids:
            agent = self.system.get_agent(agent_id)
            if agent and agent._memory:
                results.append(agent._memory[-1].content)
        
        # 合并结果
        if merge_func:
            return merge_func(results)
        return results


# ============================================================================
# 6. 辩论系统示例
# ============================================================================

class DebateSystem:
    """
    辩论系统
    
    包含正方、反方和裁判，进行结构化辩论
    """
    
    def __init__(self, topic: str):
        self.topic = topic
        self.system = MultiAgentSystem(f"Debate-{topic[:10]}")
        self.debate_record: List[str] = []
        
        # 创建辩论 Agent
        self.affirmative = DebateAgent("affirmative")
        self.negative = DebateAgent("negative")
        self.judge = JudgeAgent()
        
        # 注册到系统
        self.system.register_agent(self.affirmative)
        self.system.register_agent(self.negative)
        self.system.register_agent(self.judge)
    
    def run_debate(self, rounds: int = 3) -> str:
        """
        运行辩论
        
        Args:
            rounds: 辩论轮数
            
        Returns:
            辩论结果
        """
        print(f"\n{'='*60}")
        print(f"辩论主题：{self.topic}")
        print(f"{'='*60}")
        
        # 立论阶段
        print("\n【立论阶段】")
        self._round("立论")
        
        # 攻辩阶段
        for i in range(rounds):
            print(f"\n【攻辩第{i+1}轮】")
            self._round(f"攻辩{i+1}")
        
        # 总结阶段
        print("\n【总结阶段】")
        self._round("总结")
        
        # 裁判评估
        print("\n【裁判评估】")
        record = "\n".join(self.debate_record)
        eval_msg = Message(
            msg_type=MessageType.REQUEST,
            sender_id="system",
            receiver_id=self.judge.agent_id,
            content=record
        )
        self.system.send_message(eval_msg)
        self.system.run_until_idle()
        
        result = "辩论结束，详见记录。"
        if self.judge._memory:
            result = self.judge._memory[-1].content
        
        return result
    
    def _round(self, stage: str):
        """进行一轮辩论"""
        # 正方发言
        arg_msg = Message(
            msg_type=MessageType.QUERY,
            sender_id="system",
            receiver_id=self.affirmative.agent_id,
            content=f"{stage}: {self.topic}"
        )
        self.system.send_message(arg_msg)
        self.system.run_until_idle(max_iterations=2)
        
        if self.affirmative._memory:
            aff_arg = self.affirmative._memory[-1].content
            self.debate_record.append(f"正方({stage}): {aff_arg}")
            print(f"正方：{aff_arg}")
        
        # 反方发言
        arg_msg = Message(
            msg_type=MessageType.QUERY,
            sender_id="system",
            receiver_id=self.negative.agent_id,
            content=f"{stage}: {self.topic}"
        )
        self.system.send_message(arg_msg)
        self.system.run_until_idle(max_iterations=2)
        
        if self.negative._memory:
            neg_arg = self.negative._memory[-1].content
            self.debate_record.append(f"反方({stage}): {neg_arg}")
            print(f"反方：{neg_arg}")


# ============================================================================
# 7. 完整示例
# ============================================================================

def demo_basic_mas():
    """演示基本多 Agent 系统"""
    print("=" * 60)
    print("基本多 Agent 系统演示")
    print("=" * 60)
    
    # 创建系统
    system = MultiAgentSystem("DemoSystem")
    
    # 创建并注册 Agent
    calc_agent = CalculatorAgent()
    search_agent = SearchAgent()
    writer_agent = WriterAgent()
    
    system.register_agent(calc_agent)
    system.register_agent(search_agent)
    system.register_agent(writer_agent)
    
    # 发送消息
    print("\n发送计算请求...")
    msg = Message(
        msg_type=MessageType.QUERY,
        sender_id="user",
        receiver_id="calc_001",
        content="100 + 200 * 3"
    )
    system.send_message(msg)
    system.run_until_idle()
    
    print("\n发送搜索请求...")
    msg = Message(
        msg_type=MessageType.QUERY,
        sender_id="user",
        receiver_id="search_001",
        content="AI"
    )
    system.send_message(msg)
    system.run_until_idle()
    
    # 显示状态
    print("\n系统状态:")
    status = system.get_status()
    print(f"  Agent 数量：{status['agent_count']}")
    print(f"  消息数量：{status['message_count']}")


def demo_sequential_collaboration():
    """演示串行协作"""
    print("\n" + "=" * 60)
    print("串行协作演示：内容创作流程")
    print("=" * 60)
    
    system = MultiAgentSystem("ContentCreation")
    
    # 注册 Agent
    writer = WriterAgent()
    reviewer = ReviewerAgent()
    
    system.register_agent(writer)
    system.register_agent(reviewer)
    
    # 执行串行协作
    collaboration = SequentialCollaboration(
        system,
        [writer.agent_id, reviewer.agent_id]
    )
    
    result = collaboration.execute("人工智能的发展")
    print(f"\n最终结果:\n{result}")


def demo_parallel_collaboration():
    """演示并行协作"""
    print("\n" + "=" * 60)
    print("并行协作演示：多 Agent 同时搜索")
    print("=" * 60)
    
    system = MultiAgentSystem("ParallelSearch")
    
    # 创建多个搜索 Agent
    for i in range(3):
        agent = SearchAgent()
        agent.agent_id = f"search_{i}"
        system.register_agent(agent)
    
    # 执行并行协作
    collaboration = ParallelCollaboration(
        system,
        ["search_0", "search_1", "search_2"]
    )
    
    results = collaboration.execute("Python")
    print(f"\n并行搜索结果:\n{results}")


def demo_debate_system():
    """演示辩论系统"""
    print("\n" + "=" * 60)
    print("辩论系统演示")
    print("=" * 60)
    
    debate = DebateSystem("人工智能是否应该取代人类工作")
    result = debate.run_debate(rounds=2)
    
    print(f"\n{result}")


def demo_custom_agent():
    """演示自定义 Agent"""
    print("\n" + "=" * 60)
    print("自定义 Agent 演示")
    print("=" * 60)
    
    # 创建自定义 Agent
    class WeatherAgent(Agent):
        def __init__(self):
            super().__init__("weather_001", "天气 Agent", "查询天气")
        
        def process_message(self, message: Message) -> Optional[Message]:
            if message.msg_type == MessageType.QUERY:
                city = message.content
                return message.create_reply(f"{city}今天晴朗，气温 25°C")
            
            return None
        
        def get_capabilities(self) -> List[str]:
            return ["天气查询", "气象预报"]
    
    # 使用自定义 Agent
    system = MultiAgentSystem("WeatherSystem")
    weather_agent = WeatherAgent()
    system.register_agent(weather_agent)
    
    # 发送查询
    msg = Message(
        msg_type=MessageType.QUERY,
        sender_id="user",
        receiver_id="weather_001",
        content="北京"
    )
    system.send_message(msg)
    system.run_until_idle()
    
    print(f"\n天气查询完成，查看 Agent 记忆:")
    if weather_agent._memory:
        for m in weather_agent._memory:
            print(f"  {m}")


if __name__ == "__main__":
    # 运行所有演示
    demo_basic_mas()
    demo_sequential_collaboration()
    demo_parallel_collaboration()
    demo_debate_system()
    demo_custom_agent()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
