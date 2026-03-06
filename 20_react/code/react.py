# -*- coding: utf-8 -*-
"""
第 20 章 ReAct 框架 - 代码实现

本模块实现了 ReAct（Reasoning + Acting）框架的核心组件：
- ReActAgent: ReAct Agent 主类
- Thought-Action-Observation 循环
- 工具集成与调用
- 示例任务演示

作者：AI 前沿技术教程
版本：1.0
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re


# ============================================================================
# 数据结构定义
# ============================================================================

class ActionType(Enum):
    """行动类型枚举"""
    SEARCH = "search"
    CALCULATOR = "calculator"
    WEATHER = "weather"
    FINISH = "finish"


@dataclass
class Thought:
    """思考记录"""
    content: str
    step: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Action:
    """行动记录"""
    action_type: ActionType
    input: str
    step: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Observation:
    """观察记录"""
    content: str
    step: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReActStep:
    """ReAct 循环单步记录"""
    thought: Thought
    action: Action
    observation: Observation


# ============================================================================
# 工具模块（Tool Module）
# ============================================================================

class Tool(ABC):
    """工具抽象基类"""
    
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
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """执行工具"""
        pass


class CalculatorTool(Tool):
    """
    计算器工具
    执行数学计算
    """
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "执行数学计算，支持加减乘除、幂运算等"
    
    def execute(self, expression: str) -> float:
        """
        执行数学计算
        
        Args:
            expression: 数学表达式
            
        Returns:
            计算结果
        """
        # 安全检查
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含非法字符"
        
        try:
            result = eval(expression)
            return float(result)
        except Exception as e:
            return f"计算错误：{str(e)}"


class SearchTool(Tool):
    """
    搜索工具
    模拟搜索引擎查询
    """
    
    @property
    def name(self) -> str:
        return "search"
    
    @property
    def description(self) -> str:
        return "搜索互联网信息，获取相关知识"
    
    def execute(self, query: str) -> str:
        """
        执行搜索
        
        Args:
            query: 搜索关键词
            
        Returns:
            搜索结果
        """
        # 模拟搜索结果数据库
        knowledge_base = {
            "python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建，以简洁易读著称。",
            "ai": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行智能任务的系统。",
            "奥运会": "2024 年夏季奥运会在法国巴黎举办，这是巴黎第三次举办奥运会。",
            "诺贝尔奖": "诺贝尔奖是根据阿尔弗雷德·诺贝尔的遗嘱设立的年度奖项，包括物理、化学、医学、文学、和平和经济学。",
            "马斯克": "埃隆·马斯克是特斯拉和 SpaceX 的 CEO，2024 年净资产约 2500 亿美元。",
            "北京": "北京是中国的首都，人口约 2100 万，是政治、文化和教育中心。",
            "上海": "上海是中国的经济中心，人口约 2400 万，是中国最大的城市。"
        }
        
        query_lower = query.lower()
        
        # 查找匹配的知识
        for key, value in knowledge_base.items():
            if key in query_lower or key.lower() in query_lower:
                return value
        
        return f"未找到关于'{query}'的具体信息。请尝试更精确的关键词。"


class WeatherTool(Tool):
    """
    天气查询工具
    获取指定城市的天气信息
    """
    
    @property
    def name(self) -> str:
        return "weather"
    
    @property
    def description(self) -> str:
        return "查询指定城市的天气情况"
    
    def execute(self, city: str) -> str:
        """
        查询天气
        
        Args:
            city: 城市名称
            
        Returns:
            天气信息
        """
        # 模拟天气数据
        weather_data = {
            "北京": "晴，温度 25°C，湿度 40%，空气质量良",
            "上海": "多云，温度 28°C，湿度 65%，有轻微雾霾",
            "广州": "小雨，温度 30°C，湿度 80%，注意带伞",
            "深圳": "晴，温度 32°C，湿度 70%，适宜户外活动",
            "杭州": "阴，温度 26°C，湿度 60%",
            "成都": "多云，温度 24°C，湿度 55%"
        }
        
        return weather_data.get(city, f"暂无{city}的天气数据")


# ============================================================================
# ReAct Agent 核心类
# ============================================================================

class ReActAgent:
    """
    ReAct Agent 主类
    
    实现 Thought-Action-Observation 循环，
    整合推理和行动能力。
    """
    
    def __init__(self, name: str = "ReAct Agent", max_iterations: int = 10):
        """
        初始化 ReAct Agent
        
        Args:
            name: Agent 名称
            max_iterations: 最大迭代次数，防止无限循环
        """
        self.name = name
        self.max_iterations = max_iterations
        self.tools: Dict[str, Tool] = {}
        self.history: List[ReActStep] = []
        self.current_step = 0
        
        # 注册默认工具
        self.register_tool(CalculatorTool())
        self.register_tool(SearchTool())
        self.register_tool(WeatherTool())
        
        print(f"[{self.name}] 初始化完成，最大迭代次数：{max_iterations}")
    
    def register_tool(self, tool: Tool) -> None:
        """
        注册工具
        
        Args:
            tool: 工具实例
        """
        self.tools[tool.name] = tool
        print(f"[{self.name}] 已注册工具：{tool.name}")
    
    def get_tool_descriptions(self) -> str:
        """获取工具描述文本，用于 Prompt"""
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)
    
    def get_tool_names(self) -> List[str]:
        """获取工具名称列表"""
        return list(self.tools.keys())
    
    def _parse_action(self, thought_text: str) -> Optional[Tuple[str, str]]:
        """
        从思考文本中解析行动
        
        Args:
            thought_text: 思考文本
            
        Returns:
            (action_type, action_input) 或 None
        """
        # 尝试匹配行动模式
        patterns = [
            r"Action:\s*(\w+)\s*\n?\s*Action Input:\s*(.+)",
            r"行动：\s*(\w+)\s*\n?\s*输入：\s*(.+)",
            r"调用：\s*(\w+)\s*\n?\s*参数：\s*(.+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, thought_text, re.IGNORECASE)
            if match:
                return match.group(1).lower(), match.group(2).strip()
        
        return None
    
    def _execute_action(self, action_type: str, action_input: str) -> str:
        """
        执行行动
        
        Args:
            action_type: 行动类型
            action_input: 行动输入
            
        Returns:
            观察结果
        """
        # 检查是否是结束行动
        if action_type in ["finish", "结束", "answer", "答案"]:
            return f"FINAL_ANSWER: {action_input}"
        
        # 查找并执行工具
        tool = self.tools.get(action_type)
        if not tool:
            return f"错误：未知工具 '{action_type}'，可用工具：{list(self.tools.keys())}"
        
        try:
            # 解析参数
            params = self._parse_action_input(action_input, tool)
            result = tool.execute(**params)
            return str(result)
        except Exception as e:
            return f"执行错误：{str(e)}"
    
    def _parse_action_input(self, action_input: str, tool: Tool) -> Dict[str, Any]:
        """
        解析行动输入参数
        
        Args:
            action_input: 行动输入字符串
            tool: 目标工具
            
        Returns:
            参数字典
        """
        params = {}
        
        if tool.name == "calculator":
            # 提取数学表达式
            params["expression"] = action_input.strip('"\'')
        
        elif tool.name == "search":
            params["query"] = action_input.strip('"\'')
        
        elif tool.name == "weather":
            # 提取城市名
            cities = ["北京", "上海", "广州", "深圳", "杭州", "成都"]
            city_found = False
            for city in cities:
                if city in action_input:
                    params["city"] = city
                    city_found = True
                    break
            if not city_found:
                params["city"] = "北京"  # 默认城市
        
        return params
    
    def _generate_thought(self, task: str, observations: List[str]) -> str:
        """
        生成思考（模拟 LLM 推理）
        
        在实际应用中，这里应该调用 LLM 生成思考。
        这里使用规则-based 方法模拟。
        
        Args:
            task: 原始任务
            observations: 历史观察列表
            
        Returns:
            思考文本
        """
        task_lower = task.lower()
        
        # 根据任务类型生成思考
        if self.current_step == 0:
            # 第一步：分析任务
            if any(k in task_lower for k in ["计算", "算", "加", "减", "乘", "除"]):
                return f"""Thought: 这是一个计算任务，我需要使用计算器工具
Action: calculator
Action Input: {self._extract_expression(task)}"""
            
            elif "天气" in task_lower:
                city = self._extract_city(task)
                return f"""Thought: 用户想查询天气信息，我需要使用天气工具
Action: weather
Action Input: {city}"""
            
            elif any(k in task_lower for k in ["搜索", "查询", "找", "什么", "是谁"]):
                return f"""Thought: 用户需要查询信息，我需要使用搜索工具
Action: search
Action Input: {task.replace('搜索', '').replace('查询', '').strip()}"""
            
            else:
                return f"""Thought: 我需要分析这个任务并选择合适的工具
Action: search
Action Input: {task}"""
        
        else:
            # 后续步骤：基于观察继续推理
            if observations:
                last_obs = observations[-1]
                
                if "错误" in last_obs:
                    return """Thought: 上一步执行出错，让我尝试其他方法
Action: search
Action Input: 重试查询"""
                
                elif "未找到" in last_obs:
                    return """Thought: 没有找到相关信息，让我换个关键词搜索
Action: search
Action Input: 简化关键词"""
                
                else:
                    # 已经有结果，可以结束
                    return f"""Thought: 我已经获得了足够的信息，可以给出最终答案了
Action: finish
Action Input: {task} 的答案是：{last_obs}"""
            
            return """Thought: 继续处理任务
Action: finish
Action Input: 任务完成"""
    
    def _extract_expression(self, task: str) -> str:
        """从任务中提取数学表达式"""
        match = re.search(r'[\d+\-*/().\s]+', task)
        return match.group().strip() if match else "0"
    
    def _extract_city(self, task: str) -> str:
        """从任务中提取城市名"""
        cities = ["北京", "上海", "广州", "深圳", "杭州", "成都"]
        for city in cities:
            if city in task:
                return city
        return "北京"
    
    def run(self, task: str, verbose: bool = True) -> str:
        """
        运行 ReAct 循环处理任务
        
        Args:
            task: 用户任务
            verbose: 是否输出详细日志
            
        Returns:
            最终答案
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"[{self.name}] 任务：{task}")
            print(f"{'='*60}\n")
        
        observations = []
        self.current_step = 0
        self.history = []
        
        while self.current_step < self.max_iterations:
            self.current_step += 1
            
            if verbose:
                print(f"--- 第 {self.current_step} 步 ---")
            
            # 1. Thought（思考）
            thought_text = self._generate_thought(task, observations)
            thought = Thought(content=thought_text, step=self.current_step)
            
            if verbose:
                print(f"\n[Thought {self.current_step}]")
                # 只显示第一行思考
                first_line = thought_text.split('\n')[0]
                print(first_line)
            
            # 2. Action（行动）
            action_parsed = self._parse_action(thought_text)
            
            if not action_parsed:
                if verbose:
                    print("[Action] 无法解析行动，尝试直接回答")
                return f"我无法解析这个任务：{task}"
            
            action_type, action_input = action_parsed
            action = Action(action_type=ActionType(action_type) if action_type in [e.value for e in ActionType] else ActionType.SEARCH,
                          input=action_input, step=self.current_step)
            
            if verbose:
                print(f"\n[Action {self.current_step}]")
                print(f"工具：{action_type}")
                print(f"输入：{action_input}")
            
            # 3. Observation（观察）
            observation_content = self._execute_action(action_type, action_input)
            observation = Observation(content=observation_content, step=self.current_step)
            
            if verbose:
                print(f"\n[Observation {self.current_step}]")
                print(observation_content)
            
            # 检查是否是最终答案
            if "FINAL_ANSWER:" in observation_content:
                final_answer = observation_content.replace("FINAL_ANSWER:", "").strip()
                
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"[{self.name}] 最终答案：{final_answer}")
                    print(f"{'='*60}\n")
                
                # 记录完整历史
                step_record = ReActStep(thought=thought, action=action, observation=observation)
                self.history.append(step_record)
                
                return final_answer
            
            # 记录这一步
            step_record = ReActStep(thought=thought, action=action, observation=observation)
            self.history.append(step_record)
            observations.append(observation_content)
            
            if verbose:
                print()  # 空行
        
        # 达到最大迭代次数
        return f"达到最大迭代次数 ({self.max_iterations})，未能完成任务。最后观察：{observations[-1] if observations else '无'}"
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return [
            {
                "step": step.thought.step,
                "thought": step.thought.content,
                "action": {
                    "type": step.action.action_type.value,
                    "input": step.action.input
                },
                "observation": step.observation.content
            }
            for step in self.history
        ]
    
    def reset(self) -> None:
        """重置 Agent 状态"""
        self.history = []
        self.current_step = 0
        print(f"[{self.name}] 已重置")


# ============================================================================
# 示例任务演示
# ============================================================================

def demo_math():
    """演示数学计算任务"""
    print("\n" + "="*60)
    print("【示例 1】数学计算任务")
    print("="*60)
    
    agent = ReActAgent(name="数学助手")
    result = agent.run("请计算 123 + 456 的结果")
    
    print(f"\n最终结果：{result}")
    return result


def demo_search():
    """演示信息搜索任务"""
    print("\n" + "="*60)
    print("【示例 2】信息搜索任务")
    print("="*60)
    
    agent = ReActAgent(name="搜索助手")
    result = agent.run("Python 是什么编程语言")
    
    print(f"\n最终结果：{result}")
    return result


def demo_weather():
    """演示天气查询任务"""
    print("\n" + "="*60)
    print("【示例 3】天气查询任务")
    print("="*60)
    
    agent = ReActAgent(name="天气助手")
    result = agent.run("北京今天天气怎么样")
    
    print(f"\n最终结果：{result}")
    return result


def demo_complex():
    """演示复杂任务（多步骤）"""
    print("\n" + "="*60)
    print("【示例 4】复杂任务（多步骤）")
    print("="*60)
    
    agent = ReActAgent(name="全能助手", max_iterations=15)
    
    # 演示多轮交互
    tasks = [
        "计算 100 除以 4",
        "搜索人工智能的定义",
        "上海天气如何"
    ]
    
    results = []
    for task in tasks:
        result = agent.run(task, verbose=True)
        results.append({"task": task, "result": result})
        agent.reset()  # 重置 Agent 状态
    
    print("\n" + "="*60)
    print("【任务汇总】")
    print("="*60)
    for item in results:
        print(f"\n任务：{item['task']}")
        print(f"结果：{item['result']}")
    
    return results


def demo_history():
    """演示历史记录查看"""
    print("\n" + "="*60)
    print("【示例 5】查看执行历史")
    print("="*60)
    
    agent = ReActAgent(name="记录助手")
    agent.run("计算 25 * 4", verbose=False)
    
    history = agent.get_history()
    
    print("\n【执行历史】")
    for step in history:
        print(f"\n步骤 {step['step']}:")
        print(f"  思考：{step['thought'][:50]}...")
        print(f"  行动：{step['action']['type']}({step['action']['input']})")
        print(f"  观察：{step['observation'][:50]}...")
    
    return history


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序入口"""
    print("\n" + "="*60)
    print("ReAct 框架演示程序")
    print("="*60)
    
    # 运行所有演示
    demo_math()
    demo_search()
    demo_weather()
    demo_complex()
    demo_history()
    
    print("\n" + "="*60)
    print("所有演示完成！")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
