# -*- coding: utf-8 -*-
"""
第 21 章 工具使用与 Function Calling - 代码实现

本模块实现了 Function Calling 的核心组件：
- FunctionCall 类：表示一次函数调用
- ToolRegistry 类：工具注册表
- 示例工具：计算器、搜索、代码执行等
- Agent 类：完整的 Agent + Tools 示例

作者：AI Frontier Tutorial
日期：2026
"""

import json
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


# ============================================================================
# 1. FunctionCall 类 - 表示一次函数调用
# ============================================================================

@dataclass
class FunctionCall:
    """
    表示一次函数调用请求
    
    Attributes:
        name: 函数名称
        arguments: 函数参数字典
        call_id: 调用 ID（用于追踪）
    """
    name: str
    arguments: Dict[str, Any]
    call_id: str = field(default_factory=lambda: f"call_{int(time.time() * 1000)}")
    
    def to_dict(self) -> Dict:
        """转换为字典格式（兼容 OpenAI API 格式）"""
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False)
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FunctionCall':
        """从字典创建 FunctionCall 实例"""
        func_data = data.get("function", {})
        return cls(
            name=func_data.get("name", ""),
            arguments=json.loads(func_data.get("arguments", "{}")),
            call_id=data.get("id", "")
        )
    
    def __str__(self) -> str:
        return f"FunctionCall({self.name}, {self.arguments})"


# ============================================================================
# 2. Tool Schema 定义 - 工具的 JSON Schema 描述
# ============================================================================

@dataclass
class ToolSchema:
    """
    工具的 Schema 定义，用于描述工具的输入参数
    
    Attributes:
        name: 工具名称
        description: 工具描述
        parameters: 参数定义（JSON Schema 格式）
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """转换为 OpenAI API 兼容格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    @staticmethod
    def create_param_schema(
        param_type: str = "string",
        description: str = "",
        enum: Optional[List] = None,
        default: Any = None
    ) -> Dict:
        """
        创建参数 Schema
        
        Args:
            param_type: 参数类型 (string, number, boolean, array, object)
            description: 参数描述
            enum: 可选值列表
            default: 默认值
            
        Returns:
            参数字典
        """
        schema = {
            "type": param_type,
            "description": description
        }
        if enum:
            schema["enum"] = enum
        return schema


# ============================================================================
# 3. ToolRegistry 类 - 工具注册表
# ============================================================================

class ToolRegistry:
    """
    工具注册表 - 管理所有可用工具
    
    功能：
    - 注册工具函数
    - 获取工具 Schema
    - 执行工具调用
    - 批量管理工具
    """
    
    def __init__(self):
        self._tools: Dict[str, Dict] = {}
    
    def register(
        self,
        name: str = None,
        description: str = None,
        parameters: Dict = None
    ):
        """
        装饰器：注册工具函数
        
        Args:
            name: 工具名称（默认为函数名）
            description: 工具描述
            parameters: 参数 Schema
            
        Example:
            @registry.register(
                name="get_weather",
                description="获取天气信息",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "城市名"}
                    },
                    "required": ["location"]
                }
            )
            def get_weather(location: str) -> str:
                return f"{location}的天气是晴天"
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            
            # 如果没有提供描述，尝试从 docstring 获取
            tool_description = description or func.__doc__ or ""
            
            # 如果没有提供参数，尝试从函数签名推断
            tool_params = parameters or self._infer_params(func)
            
            self._tools[tool_name] = {
                "function": func,
                "schema": ToolSchema(
                    name=tool_name,
                    description=tool_description.strip(),
                    parameters=tool_params
                )
            }
            return func
        return decorator
    
    def _infer_params(self, func: Callable) -> Dict:
        """
        从函数签名推断参数 Schema（简化版）
        
        实际使用中建议手动定义参数 Schema 以获得更好的控制
        """
        import inspect
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for name, param in sig.parameters.items():
            if name in ('self', 'cls'):
                continue
            
            # 推断类型
            param_type = "string"
            if param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"
            elif param.annotation == list:
                param_type = "array"
            
            properties[name] = {
                "type": param_type,
                "description": f"参数 {name}"
            }
            
            # 如果没有默认值，则是必需参数
            if param.default == inspect.Parameter.empty:
                required.append(name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def get_schema(self, name: str) -> Optional[Dict]:
        """获取工具的 Schema（OpenAI API 格式）"""
        tool = self._tools.get(name)
        return tool["schema"].to_dict() if tool else None
    
    def get_all_schemas(self) -> List[Dict]:
        """获取所有工具的 Schema 列表"""
        return [tool["schema"].to_dict() for tool in self._tools.values()]
    
    def execute(self, tool_name: str, **kwargs) -> Any:
        """
        执行工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
        """
        if tool_name not in self._tools:
            raise ValueError(f"未知工具：{tool_name}")
        
        func = self._tools[tool_name]["function"]
        return func(**kwargs)
    
    def has_tool(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self._tools
    
    def list_tools(self) -> List[str]:
        """列出所有注册的工具名称"""
        return list(self._tools.keys())


# ============================================================================
# 4. 示例工具实现
# ============================================================================

# 创建全局注册表实例
registry = ToolRegistry()


@registry.register(
    name="calculator",
    description="执行数学计算，支持加减乘除、幂运算等",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'"
            }
        },
        "required": ["expression"]
    }
)
def calculator(expression: str) -> str:
    """
    计算器工具 - 安全地执行数学表达式
    
    Args:
        expression: 数学表达式
        
    Returns:
        计算结果
    """
    try:
        # 只允许安全的数学运算
        import math
        
        # 定义安全的命名空间
        safe_dict = {
            "abs": abs, "round": round,
            "pow": pow, "sum": sum, "min": min, "max": max,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, 
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e
        }
        
        # 验证表达式只包含安全字符
        if not re.match(r'^[\d+\-*/().\s\w]+$', expression):
            return "错误：表达式包含非法字符"
        
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


@registry.register(
    name="web_search",
    description="搜索网络信息，获取最新资讯",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词"
            },
            "num_results": {
                "type": "integer",
                "description": "返回结果数量",
                "default": 5
            }
        },
        "required": ["query"]
    }
)
def web_search(query: str, num_results: int = 5) -> str:
    """
    网络搜索工具（模拟实现）
    
    实际使用时可以集成真实的搜索 API（如 Bing Search API、Google Custom Search 等）
    
    Args:
        query: 搜索关键词
        num_results: 返回结果数量
        
    Returns:
        搜索结果摘要
    """
    # 模拟搜索结果
    results = [
        f"结果 1: 关于'{query}'的相关信息 - 这是模拟的搜索结果",
        f"结果 2: {query}的最新动态 - 请访问相关网站获取详细信息",
        f"结果 3: {query}的详细介绍 - 包含历史、特点等内容",
        f"结果 4: {query}的常见问题解答",
        f"结果 5: {query}的相关资源链接"
    ]
    
    return "\n".join(results[:num_results])


@registry.register(
    name="code_executor",
    description="执行 Python 代码并返回结果",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的 Python 代码"
            },
            "timeout": {
                "type": "integer",
                "description": "执行超时时间（秒）",
                "default": 30
            }
        },
        "required": ["code"]
    }
)
def code_executor(code: str, timeout: int = 30) -> str:
    """
    代码执行工具 - 在沙箱环境中执行 Python 代码
    
    ⚠️ 注意：生产环境需要更严格的安全措施
    
    Args:
        code: Python 代码
        timeout: 超时时间
        
    Returns:
        执行结果或错误信息
    """
    try:
        # 创建安全的执行环境
        safe_globals = {
            "__builtins__": {
                'abs': abs, 'all': all, 'any': any, 'bool': bool,
                'dict': dict, 'enumerate': enumerate, 'float': float,
                'int': int, 'len': len, 'list': list, 'map': map,
                'max': max, 'min': min, 'range': range, 'str': str,
                'sum': sum, 'zip': zip, 'reversed': reversed,
                'sorted': sorted, 'type': type
            }
        }
        safe_locals = {}
        
        # 执行代码
        exec(code, safe_globals, safe_locals)
        
        # 返回结果
        output = []
        for name, value in safe_locals.items():
            if not name.startswith('_'):
                output.append(f"{name} = {value}")
        
        if output:
            return "执行成功:\n" + "\n".join(output)
        else:
            return "代码执行成功（无输出）"
            
    except Exception as e:
        return f"代码执行错误：{type(e).__name__}: {str(e)}"


@registry.register(
    name="get_weather",
    description="获取指定城市的天气信息",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "城市名称，如'北京'、'上海'"
            },
            "date": {
                "type": "string",
                "description": "日期（可选），格式'YYYY-MM-DD'，默认为今天"
            }
        },
        "required": ["location"]
    }
)
def get_weather(location: str, date: str = None) -> str:
    """
    天气查询工具（模拟实现）
    
    实际使用时可以集成 wttr.in 或 Open-Meteo API
    
    Args:
        location: 城市名称
        date: 日期（可选）
        
    Returns:
        天气信息
    """
    # 模拟天气数据
    import random
    conditions = ["晴天", "多云", "小雨", "大雨", "阴天"]
    temp = random.randint(15, 35)
    
    date_str = date or "今天"
    return f"{location}{date_str}的天气：{random.choice(conditions)}，气温{temp}°C"


@registry.register(
    name="current_time",
    description="获取当前日期和时间",
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "时区，如'Asia/Shanghai'、'UTC'",
                "default": "Asia/Shanghai"
            },
            "format": {
                "type": "string",
                "description": "时间格式，如'%Y-%m-%d %H:%M:%S'",
                "default": "%Y-%m-%d %H:%M:%S"
            }
        }
    }
)
def current_time(timezone: str = "Asia/Shanghai", format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    获取当前时间
    
    Args:
        timezone: 时区
        format: 时间格式
        
    Returns:
        格式化后的时间字符串
    """
    from datetime import datetime
    
    # 简单实现（实际应考虑时区转换）
    now = datetime.now()
    return now.strftime(format)


# ============================================================================
# 5. Agent 类 - 完整的 Agent + Tools 实现
# ============================================================================

class Message:
    """消息类，表示对话中的一条消息"""
    
    def __init__(self, role: str, content: str = None, tool_calls: List[FunctionCall] = None, 
                 tool_call_id: str = None):
        self.role = role  # "user", "assistant", "tool"
        self.content = content
        self.tool_calls = tool_calls or []
        self.tool_call_id = tool_call_id
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        result = {"role": self.role}
        if self.content:
            result["content"] = self.content
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


class Agent:
    """
    基于 Function Calling 的智能 Agent
    
    功能：
    - 理解用户意图
    - 选择合适的工具
    - 执行工具调用
    - 整合结果并回复
    
    Attributes:
        registry: 工具注册表
        messages: 对话历史
        max_iterations: 最大迭代次数（防止无限循环）
    """
    
    def __init__(self, registry: ToolRegistry = None, max_iterations: int = 5):
        self.registry = registry or registry
        self.messages: List[Message] = []
        self.max_iterations = max_iterations
    
    def add_message(self, role: str, content: str = None, 
                    tool_calls: List[FunctionCall] = None, tool_call_id: str = None):
        """添加消息到对话历史"""
        self.messages.append(Message(role, content, tool_calls, tool_call_id))
    
    def clear_history(self):
        """清空对话历史"""
        self.messages = []
    
    def _simulate_llm_call(self) -> Message:
        """
        模拟 LLM 调用
        
        实际使用时应替换为真实的 LLM API 调用（如 OpenAI、Claude 等）
        这里使用简单的规则匹配来演示 Function Calling 流程
        """
        last_user_msg = ""
        for msg in reversed(self.messages):
            if msg.role == "user":
                last_user_msg = msg.content
                break
        
        # 简单的意图识别（实际应使用 LLM）
        tool_calls = []
        
        # 检测是否需要调用计算器
        if any(kw in last_user_msg for kw in ['计算', '算一下', '等于', '+', '-', '*', '/']):
            # 提取表达式
            match = re.search(r'([0-9+\-*/().\s]+)', last_user_msg)
            if match:
                expr = match.group(1).strip()
                tool_calls.append(FunctionCall(
                    name="calculator",
                    arguments={"expression": expr}
                ))
        
        # 检测是否需要搜索
        elif any(kw in last_user_msg for kw in ['搜索', '查一下', '找一下', '百度', 'google']):
            match = re.search(r'[搜索查找](?:一)?下 (.+?)(?:？|$)', last_user_msg)
            if match:
                query = match.group(1).strip()
                tool_calls.append(FunctionCall(
                    name="web_search",
                    arguments={"query": query}
                ))
        
        # 检测是否需要查天气
        elif any(kw in last_user_msg for kw in ['天气', '气温', '多少度']):
            match = re.search(r'([北京上海广州深圳]+).*?天气', last_user_msg)
            if match:
                location = match.group(1)
                tool_calls.append(FunctionCall(
                    name="get_weather",
                    arguments={"location": location}
                ))
        
        # 检测是否需要执行代码
        elif 'python' in last_user_msg.lower() or ('代码' in last_user_msg and '执行' in last_user_msg):
            match = re.search(r'```python\s*(.+?)\s*```', last_user_msg, re.DOTALL)
            if match:
                code = match.group(1).strip()
                tool_calls.append(FunctionCall(
                    name="code_executor",
                    arguments={"code": code}
                ))
        
        # 检测是否需要时间
        elif any(kw in last_user_msg for kw in ['时间', '几点', '日期', '今天']):
            tool_calls.append(FunctionCall(
                name="current_time",
                arguments={}
            ))
        
        if tool_calls:
            return Message(role="assistant", tool_calls=tool_calls)
        else:
            # 没有工具调用，直接回复
            return Message(role="assistant", content=f"我收到了你的消息：{last_user_msg}")
    
    def _execute_tool_calls(self, tool_calls: List[FunctionCall]) -> List[Message]:
        """
        执行工具调用
        
        Args:
            tool_calls: 工具调用列表
            
        Returns:
            工具结果消息列表
        """
        results = []
        for tc in tool_calls:
            try:
                result = self.registry.execute(tool_name=tc.name, **tc.arguments)
                results.append(Message(
                    role="tool",
                    content=str(result),
                    tool_call_id=tc.call_id
                ))
            except Exception as e:
                results.append(Message(
                    role="tool",
                    content=f"工具执行错误：{str(e)}",
                    tool_call_id=tc.call_id
                ))
        return results
    
    def chat(self, user_message: str) -> str:
        """
        与用户对话
        
        Args:
            user_message: 用户消息
            
        Returns:
            Agent 回复
        """
        # 添加用户消息
        self.add_message("user", user_message)
        
        # 迭代处理
        for _ in range(self.max_iterations):
            # 模拟 LLM 调用
            response = self._simulate_llm_call()
            
            # 如果没有工具调用，返回最终回复
            if not response.tool_calls:
                self.add_message("assistant", response.content)
                return response.content
            
            # 添加 Assistant 消息（包含工具调用）
            self.add_message("assistant", tool_calls=response.tool_calls)
            
            # 执行工具
            tool_results = self._execute_tool_calls(response.tool_calls)
            
            # 添加工具结果
            for result_msg in tool_results:
                self.add_message(
                    result_msg.role,
                    content=result_msg.content,
                    tool_call_id=result_msg.tool_call_id
                )
        
        # 达到最大迭代次数，返回当前结果
        return "已达到最大迭代次数，这是目前的处理结果。"


# ============================================================================
# 6. 使用示例
# ============================================================================

def demo_basic_usage():
    """演示基本用法"""
    print("=" * 60)
    print("Function Calling 基础演示")
    print("=" * 60)
    
    # 1. 查看注册的工具
    print("\n已注册的工具:")
    for tool_name in registry.list_tools():
        print(f"  - {tool_name}")
    
    # 2. 直接调用工具
    print("\n直接调用计算器:")
    result = registry.execute(tool_name="calculator", expression="2 + 3 * 4")
    print(f"  结果：{result}")
    
    print("\n直接查询天气:")
    result = registry.execute(tool_name="get_weather", location="北京")
    print(f"  结果：{result}")


def demo_agent_chat():
    """演示 Agent 对话"""
    print("\n" + "=" * 60)
    print("Agent 对话演示")
    print("=" * 60)
    
    agent = Agent()
    
    # 测试场景 1: 计算
    print("\n用户：请计算 123 + 456 * 2")
    response = agent.chat("请计算 123 + 456 * 2")
    print(f"Agent: {response}")
    
    # 测试场景 2: 搜索
    print("\n用户：搜索一下人工智能的最新进展")
    response = agent.chat("搜索一下人工智能的最新进展")
    print(f"Agent: {response}")
    
    # 测试场景 3: 天气
    print("\n用户：北京天气怎么样？")
    response = agent.chat("北京天气怎么样？")
    print(f"Agent: {response}")
    
    # 测试场景 4: 时间
    print("\n用户：现在几点了？")
    response = agent.chat("现在几点了？")
    print(f"Agent: {response}")


def demo_custom_tool():
    """演示自定义工具"""
    print("\n" + "=" * 60)
    print("自定义工具演示")
    print("=" * 60)
    
    # 创建新的注册表
    custom_registry = ToolRegistry()
    
    # 注册自定义工具
    @custom_registry.register(
        name="greet",
        description="向用户打招呼",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "用户姓名"
                },
                "time_of_day": {
                    "type": "string",
                    "description": "时间段",
                    "enum": ["早上", "下午", "晚上"]
                }
            },
            "required": ["name"]
        }
    )
    def greet(name: str, time_of_day: str = "早上") -> str:
        return f"你好{name}，{time_of_day}好！"
    
    # 测试自定义工具
    result = custom_registry.execute(tool_name="greet", name="老板", time_of_day="下午")
    print(f"自定义工具结果：{result}")


if __name__ == "__main__":
    # 运行演示
    demo_basic_usage()
    demo_agent_chat()
    demo_custom_tool()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
