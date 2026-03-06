# -*- coding: utf-8 -*-
"""
第 19 章 Agent 基础 - 代码实现

本模块实现了 Agent 系统的核心组件：
- BaseAgent: 基础 Agent 类
- Memory: 记忆模块（ConversationBuffer、VectorStore）
- Planner: 规划器
- Tool: 工具接口与示例工具

作者：AI 前沿技术教程
版本：1.0
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


# ============================================================================
# 记忆模块（Memory Module）
# ============================================================================

@dataclass
class MemoryItem:
    """记忆项数据结构"""
    key: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 1.0  # 重要性评分
    access_count: int = 0  # 访问次数


class ConversationBuffer:
    """
    对话缓冲区 - 短期记忆
    
    存储最近的对话历史，用于维护上下文连贯性。
    使用 FIFO 策略，当超过最大长度时删除最早的记录。
    """
    
    def __init__(self, max_length: int = 10):
        """
        初始化对话缓冲区
        
        Args:
            max_length: 最大存储消息数
        """
        self.messages: List[Dict[str, str]] = []
        self.max_length = max_length
    
    def add(self, role: str, content: str) -> None:
        """
        添加消息到缓冲区
        
        Args:
            role: 角色（user/assistant/system）
            content: 消息内容
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # 如果超出最大长度，移除最早的消息
        while len(self.messages) > self.max_length:
            self.messages.pop(0)
    
    def get_context(self) -> List[Dict[str, str]]:
        """获取完整的对话上下文"""
        return self.messages
    
    def get_recent(self, n: int) -> List[Dict[str, str]]:
        """获取最近 n 条消息"""
        return self.messages[-n:] if n < len(self.messages) else self.messages
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.messages = []
    
    def __len__(self) -> int:
        return len(self.messages)


class VectorStore:
    """
    向量存储 - 长期记忆
    
    使用向量嵌入存储和检索记忆内容。
    支持基于语义相似度的检索。
    """
    
    def __init__(self, embedding_fn: Optional[callable] = None):
        """
        初始化向量存储
        
        Args:
            embedding_fn: 嵌入函数，将文本转换为向量
                         如果为 None，使用简单的哈希嵌入
        """
        self.store: Dict[str, MemoryItem] = {}
        self.vectors: Dict[str, List[float]] = {}
        self.embedding_fn = embedding_fn or self._default_embedding
    
    def _default_embedding(self, text: str) -> List[float]:
        """
        默认嵌入函数（简单哈希嵌入）
        
        实际应用中应使用真正的嵌入模型（如 sentence-transformers）
        """
        # 简单实现：基于字符的哈希
        vector = [0.0] * 128
        for i, char in enumerate(text):
            vector[i % 128] += ord(char) / 1000.0
        return vector
    
    def add(self, key: str, content: str, importance: float = 1.0) -> None:
        """
        添加记忆到存储
        
        Args:
            key: 记忆键
            content: 记忆内容
            importance: 重要性评分
        """
        item = MemoryItem(key=key, content=content, importance=importance)
        self.store[key] = item
        self.vectors[key] = self.embedding_fn(content)
    
    def get(self, key: str) -> Optional[MemoryItem]:
        """根据键获取记忆"""
        item = self.store.get(key)
        if item:
            item.access_count += 1
        return item
    
    def search(self, query: str, top_k: int = 3) -> List[MemoryItem]:
        """
        基于相似度搜索记忆
        
        Args:
            query: 查询文本
            top_k: 返回最相关的 k 个结果
            
        Returns:
            最相关的记忆项列表
        """
        query_vector = self.embedding_fn(query)
        
        # 计算余弦相似度
        similarities = []
        for key, vector in self.vectors.items():
            sim = self._cosine_similarity(query_vector, vector)
            similarities.append((key, sim))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回 top_k 结果
        results = []
        for key, _ in similarities[:top_k]:
            item = self.store[key]
            item.access_count += 1
            results.append(item)
        
        return results
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def remove(self, key: str) -> bool:
        """移除记忆"""
        if key in self.store:
            del self.store[key]
            del self.vectors[key]
            return True
        return False
    
    def list_all(self) -> List[MemoryItem]:
        """列出所有记忆"""
        return list(self.store.values())


class Memory:
    """
    记忆系统 - 统一管理短期和长期记忆
    """
    
    def __init__(self, short_term_capacity: int = 10):
        """
        初始化记忆系统
        
        Args:
            short_term_capacity: 短期记忆容量
        """
        self.short_term = ConversationBuffer(max_length=short_term_capacity)
        self.long_term = VectorStore()
    
    def add_short_term(self, role: str, content: str) -> None:
        """添加到短期记忆"""
        self.short_term.add(role, content)
    
    def add_long_term(self, key: str, content: str, importance: float = 1.0) -> None:
        """添加到长期记忆"""
        self.long_term.add(key, content, importance)
    
    def get_conversation_context(self) -> List[Dict[str, str]]:
        """获取对话上下文"""
        return self.short_term.get_context()
    
    def search_memories(self, query: str, top_k: int = 3) -> List[MemoryItem]:
        """搜索长期记忆"""
        return self.long_term.search(query, top_k)
    
    def get_memory(self, key: str) -> Optional[MemoryItem]:
        """获取特定记忆"""
        return self.long_term.get(key)


# ============================================================================
# 工具模块（Tool Module）
# ============================================================================

class Tool(ABC):
    """
    工具抽象基类
    
    所有工具必须继承此类并实现抽象方法。
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，用于 Agent 理解工具用途"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """参数定义，描述工具接受的参数"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """执行工具"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """将工具信息转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class CalculatorTool(Tool):
    """
    计算器工具
    
    执行基本数学计算。
    """
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "执行数学计算，支持加减乘除、幂运算等"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '2 + 3 * 4'"
            }
        }
    
    def execute(self, expression: str) -> float:
        """
        执行数学计算
        
        Args:
            expression: 数学表达式
            
        Returns:
            计算结果
        """
        # 安全检查：只允许数学字符
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("表达式包含非法字符")
        
        try:
            result = eval(expression)
            return float(result)
        except Exception as e:
            return f"计算错误：{str(e)}"


class SearchTool(Tool):
    """
    搜索工具
    
    模拟搜索引擎查询。
    """
    
    @property
    def name(self) -> str:
        return "search"
    
    @property
    def description(self) -> str:
        return "搜索信息，返回相关知识"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "搜索关键词"
            }
        }
    
    def execute(self, query: str) -> str:
        """
        执行搜索
        
        Args:
            query: 搜索关键词
            
        Returns:
            搜索结果
        """
        # 模拟搜索结果
        mock_results = {
            "python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
            "ai": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行智能任务的系统。",
            "weather": "天气信息需要调用实际的气象 API 获取。"
        }
        
        query_lower = query.lower()
        for key, value in mock_results.items():
            if key in query_lower:
                return value
        
        return f"未找到关于'{query}'的具体信息，请尝试更精确的关键词。"


class WeatherTool(Tool):
    """
    天气查询工具
    
    获取指定城市的天气信息。
    """
    
    @property
    def name(self) -> str:
        return "weather"
    
    @property
    def description(self) -> str:
        return "查询指定城市的天气情况"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "city": {
                "type": "string",
                "description": "城市名称"
            }
        }
    
    def execute(self, city: str) -> str:
        """
        查询天气
        
        Args:
            city: 城市名称
            
        Returns:
            天气信息
        """
        # 模拟天气数据
        mock_weather = {
            "北京": "晴，温度 25°C，湿度 40%",
            "上海": "多云，温度 28°C，湿度 65%",
            "广州": "小雨，温度 30°C，湿度 80%",
            "深圳": "晴，温度 32°C，湿度 70%"
        }
        
        return mock_weather.get(city, f"暂无{city}的天气数据")


# ============================================================================
# 规划器模块（Planner Module）
# ============================================================================

@dataclass
class TaskStep:
    """任务步骤"""
    id: int
    description: str
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    dependencies: List[int] = field(default_factory=list)


class Planner:
    """
    规划器
    
    负责任务分解和执行顺序规划。
    """
    
    def __init__(self):
        """初始化规划器"""
        self.steps: List[TaskStep] = []
        self.current_step_id: int = 0
    
    def decompose(self, task: str) -> List[TaskStep]:
        """
        将任务分解为步骤
        
        Args:
            task: 原始任务描述
            
        Returns:
            任务步骤列表
        """
        # 简单实现：根据关键词识别任务类型
        task_lower = task.lower()
        
        if "计算" in task or "算" in task:
            self.steps = [
                TaskStep(id=1, description="解析数学表达式"),
                TaskStep(id=2, description="执行计算"),
                TaskStep(id=3, description="返回结果")
            ]
        elif "天气" in task:
            self.steps = [
                TaskStep(id=1, description="识别城市名称"),
                TaskStep(id=2, description="查询天气数据"),
                TaskStep(id=3, description="格式化输出")
            ]
        elif "搜索" in task or "查询" in task:
            self.steps = [
                TaskStep(id=1, description="提取搜索关键词"),
                TaskStep(id=2, description="执行搜索"),
                TaskStep(id=3, description="整理搜索结果")
            ]
        else:
            # 通用任务分解
            self.steps = [
                TaskStep(id=1, description="理解任务需求"),
                TaskStep(id=2, description="制定执行计划"),
                TaskStep(id=3, description="执行计划"),
                TaskStep(id=4, description="返回结果")
            ]
        
        return self.steps
    
    def get_next_step(self) -> Optional[TaskStep]:
        """获取下一个待执行的步骤"""
        for step in self.steps:
            if step.status == "pending":
                # 检查依赖是否都已完成
                deps_completed = all(
                    self.steps[dep_id - 1].status == "completed"
                    for dep_id in step.dependencies
                )
                if deps_completed:
                    return step
        return None
    
    def mark_step(self, step_id: int, status: str, result: Any = None) -> None:
        """标记步骤状态"""
        for step in self.steps:
            if step.id == step_id:
                step.status = status
                step.result = result
                break
    
    def get_progress(self) -> Dict[str, Any]:
        """获取执行进度"""
        total = len(self.steps)
        completed = sum(1 for s in self.steps if s.status == "completed")
        failed = sum(1 for s in self.steps if s.status == "failed")
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "progress": f"{completed}/{total}",
            "percentage": (completed / total * 100) if total > 0 else 0
        }
    
    def reset(self) -> None:
        """重置规划器"""
        self.steps = []
        self.current_step_id = 0


# ============================================================================
# 基础 Agent 类（Base Agent）
# ============================================================================

class BaseAgent:
    """
    基础 Agent 类
    
    整合记忆、规划、工具使用等核心组件，
    实现完整的 Agent 系统。
    """
    
    def __init__(self, name: str = "Assistant"):
        """
        初始化 Agent
        
        Args:
            name: Agent 名称
        """
        self.name = name
        self.memory = Memory(short_term_capacity=10)
        self.planner = Planner()
        self.tools: Dict[str, Tool] = {}
        self.is_running = False
        
        # 注册默认工具
        self.register_tool(CalculatorTool())
        self.register_tool(SearchTool())
        self.register_tool(WeatherTool())
    
    def register_tool(self, tool: Tool) -> None:
        """
        注册工具
        
        Args:
            tool: 工具实例
        """
        self.tools[tool.name] = tool
        print(f"[{self.name}] 已注册工具：{tool.name}")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具列表"""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def select_tool(self, task: str) -> Optional[Tool]:
        """
        根据任务选择合适的工具
        
        Args:
            task: 任务描述
            
        Returns:
            选中的工具
        """
        task_lower = task.lower()
        
        # 简单匹配策略
        if any(k in task_lower for k in ["计算", "算", "加", "减", "乘", "除"]):
            return self.tools.get("calculator")
        elif "天气" in task_lower:
            return self.tools.get("weather")
        elif any(k in task_lower for k in ["搜索", "查询", "找"]):
            return self.tools.get("search")
        
        return None
    
    def execute_tool(self, tool: Tool, **kwargs) -> Any:
        """
        执行工具
        
        Args:
            tool: 工具实例
            **kwargs: 工具参数
            
        Returns:
            执行结果
        """
        try:
            print(f"[{self.name}] 执行工具：{tool.name}")
            result = tool.execute(**kwargs)
            print(f"[{self.name}] 工具执行完成")
            return result
        except Exception as e:
            print(f"[{self.name}] 工具执行失败：{str(e)}")
            return f"错误：{str(e)}"
    
    def run(self, task: str) -> str:
        """
        运行 Agent 处理任务
        
        Args:
            task: 用户任务
            
        Returns:
            执行结果
        """
        print(f"\n{'='*60}")
        print(f"[{self.name}] 接收任务：{task}")
        print(f"{'='*60}\n")
        
        # 记录到短期记忆
        self.memory.add_short_term("user", task)
        
        # 任务分解
        steps = self.planner.decompose(task)
        print(f"[{self.name}] 任务已分解为 {len(steps)} 个步骤")
        
        # 选择并执行工具
        tool = self.select_tool(task)
        
        if tool:
            # 提取参数（简单实现）
            params = self._extract_params(task, tool)
            result = self.execute_tool(tool, **params)
            
            # 记录结果
            self.memory.add_short_term("assistant", str(result))
            
            # 重要结果存入长期记忆
            if isinstance(result, (str, int, float)) and len(str(result)) < 500:
                self.memory.add_long_term(
                    key=f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    content=f"任务：{task}\n结果：{result}",
                    importance=0.8
                )
            
            return str(result)
        else:
            # 没有合适工具，返回通用响应
            response = f"我理解您的任务：'{task}'，但目前没有合适的工具来处理。"
            self.memory.add_short_term("assistant", response)
            return response
    
    def _extract_params(self, task: str, tool: Tool) -> Dict[str, Any]:
        """
        从任务中提取工具参数
        
        Args:
            task: 任务描述
            tool: 目标工具
            
        Returns:
            参数字典
        """
        params = {}
        
        if tool.name == "calculator":
            # 提取数学表达式
            import re
            match = re.search(r'[\d+\-*/().\s]+', task)
            if match:
                params["expression"] = match.group().strip()
            else:
                params["expression"] = "0"
        
        elif tool.name == "weather":
            # 提取城市名（简单实现）
            cities = ["北京", "上海", "广州", "深圳", "杭州", "成都"]
            for city in cities:
                if city in task:
                    params["city"] = city
                    break
            if "city" not in params:
                params["city"] = "北京"  # 默认城市
        
        elif tool.name == "search":
            # 提取搜索关键词
            params["query"] = task.replace("搜索", "").replace("查询", "").strip()
        
        return params
    
    def chat(self, message: str) -> str:
        """
        对话模式
        
        Args:
            message: 用户消息
            
        Returns:
            回复
        """
        # 记录对话
        self.memory.add_short_term("user", message)
        
        # 简单响应逻辑
        if "?" in message or "吗" in message:
            response = f"关于'{message}'，让我思考一下..."
        else:
            response = f"我收到了：'{message}'"
        
        self.memory.add_short_term("assistant", response)
        return response
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆摘要"""
        return {
            "short_term_count": len(self.memory.short_term),
            "long_term_count": len(self.memory.long_term.store),
            "recent_conversation": self.memory.short_term.get_recent(3)
        }
    
    def reset(self) -> None:
        """重置 Agent 状态"""
        self.memory = Memory(short_term_capacity=10)
        self.planner.reset()
        print(f"[{self.name}] 已重置")


# ============================================================================
# 示例与测试
# ============================================================================

def demo():
    """演示 Agent 功能"""
    print("\n" + "="*60)
    print("Agent 系统演示")
    print("="*60 + "\n")
    
    # 创建 Agent
    agent = BaseAgent(name="小助手")
    
    # 演示 1: 计算任务
    print("\n【演示 1】数学计算")
    result = agent.run("请帮我计算 2 + 3 * 4")
    print(f"结果：{result}")
    
    # 演示 2: 天气查询
    print("\n【演示 2】天气查询")
    result = agent.run("北京今天天气怎么样")
    print(f"结果：{result}")
    
    # 演示 3: 信息搜索
    print("\n【演示 3】信息搜索")
    result = agent.run("搜索一下 Python 是什么")
    print(f"结果：{result}")
    
    # 演示 4: 查看记忆
    print("\n【演示 4】记忆状态")
    summary = agent.get_memory_summary()
    print(f"短期记忆条数：{summary['short_term_count']}")
    print(f"长期记忆条数：{summary['long_term_count']}")
    
    # 演示 5: 对话模式
    print("\n【演示 5】对话模式")
    response = agent.chat("你好，介绍一下你自己")
    print(f"Agent: {response}")
    
    print("\n" + "="*60)
    print("演示结束")
    print("="*60 + "\n")


if __name__ == "__main__":
    demo()
