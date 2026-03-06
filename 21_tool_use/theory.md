# 第 21 章 工具使用与 Function Calling

## 21.1 Function Calling 原理

### 21.1.1 什么是 Function Calling

Function Calling（函数调用）是大型语言模型（LLM）与外部工具、API 和系统交互的核心机制。它允许 LLM：

- **理解用户意图**：识别用户需要调用哪个工具
- **提取参数**：从自然语言中提取工具调用所需的参数
- **执行动作**：通过调用外部函数完成实际任务
- **返回结果**：将工具执行结果整合到对话中

### 21.1.2 为什么需要 Function Calling

LLM 本身存在以下限制：

1. **知识截止**：训练数据有截止时间，无法获取最新信息
2. **计算能力有限**：不擅长精确数学计算
3. **无法执行动作**：不能直接操作外部系统
4. **上下文限制**：无法访问私有数据或特定领域知识

Function Calling 通过"外挂"工具扩展了 LLM 的能力边界。

### 21.1.3 工作流程

```
用户请求 → LLM 分析 → 选择工具 → 提取参数 → 执行工具 → 返回结果 → LLM 整合 → 最终回复
```

## 21.2 OpenAI Function Calling API

### 21.2.1 API 结构

OpenAI 的 Function Calling 通过 `chat.completions` API 实现：

```python
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，如'北京'"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京天气怎么样？"}],
    tools=tools
)
```

### 21.2.2 响应解析

当 LLM 决定调用工具时，返回消息包含 `tool_calls`：

```python
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    # 执行实际函数
    result = execute_function(function_name, arguments)
    
    # 将结果返回给 LLM
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result
    })
```

### 21.2.3 并行调用

LLM 可以一次性决定调用多个工具：

```python
# 响应可能包含多个 tool_calls
for tool_call in response.choices[0].message.tool_calls:
    # 并行或串行执行各个工具
    pass
```

## 21.3 工具定义与注册

### 21.3.1 工具 Schema 设计

良好的工具定义应包含：

1. **清晰的名称**：使用动词 + 名词格式，如 `get_weather`
2. **详细的描述**：说明工具用途、适用场景
3. **完整的参数定义**：
   - 参数类型（string, number, boolean, array, object）
   - 参数描述
   - 是否必需
   - 枚举值（如果适用）

### 21.3.2 工具注册表

```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}
    
    def register(self, func):
        """装饰器：注册工具函数"""
        name = func.__name__
        self.tools[name] = {
            "function": func,
            "schema": self._infer_schema(func)
        }
        return func
    
    def get_tool_schema(self, name):
        """获取工具的 JSON Schema"""
        return self.tools.get(name, {}).get("schema")
    
    def execute(self, name, **kwargs):
        """执行工具"""
        if name not in self.tools:
            raise ValueError(f"未知工具：{name}")
        return self.tools[name]["function"](**kwargs)
```

## 21.4 工具选择与调用

### 21.4.1 工具选择策略

LLM 基于以下因素选择工具：

1. **用户意图匹配**：分析用户请求的语义
2. **工具描述相关性**：匹配工具描述与用户需求
3. **参数可用性**：检查是否有所需参数
4. **上下文信息**：考虑对话历史

### 21.4.2 调用流程

```python
def agent_loop(user_message):
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        # 1. 调用 LLM
        response = llm.chat(messages, tools=registry.get_all_schemas())
        message = response.choices[0].message
        
        # 2. 检查是否需要调用工具
        if not message.tool_calls:
            return message.content
        
        # 3. 执行工具
        messages.append(message)
        for tool_call in message.tool_calls:
            result = registry.execute(
                tool_call.function.name,
                **json.loads(tool_call.function.arguments)
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
```

### 21.4.3 错误处理

```python
def safe_execute(name, **kwargs):
    try:
        return registry.execute(name, **kwargs)
    except Exception as e:
        return f"工具执行错误：{str(e)}"
```

## 21.5 复杂任务分解

### 21.5.1 任务分解模式

复杂任务通常需要多次工具调用：

**示例：旅行规划**

```
用户：帮我规划一个去日本的 5 天旅行

分解：
1. search_flights(origin="北京", destination="东京")
2. search_hotels(location="东京", days=5)
3. get_weather(location="东京", dates=[...])
4. search_attractions(location="东京")
5. calculate_budget(flights, hotels, activities)
```

### 21.5.2 链式调用

```python
def plan_trip(destination, days):
    # 第一步：查机票
    flights = search_flights(destination)
    
    # 第二步：根据机票时间查酒店
    check_in = flights.arrival_date
    hotels = search_hotels(destination, check_in, days)
    
    # 第三步：查天气
    weather = get_weather(destination, check_in)
    
    return {
        "flights": flights,
        "hotels": hotels,
        "weather": weather
    }
```

### 21.5.3 条件分支

```python
def answer_question(question):
    # 判断问题类型
    if is_math_question(question):
        return calculate(question)
    elif is_search_question(question):
        return search_web(question)
    elif is_code_question(question):
        return execute_code(question)
    else:
        return llm_answer(question)
```

## 21.6 代码解释器与插件

### 21.6.1 代码解释器

代码解释器允许 LLM 生成并执行代码：

```python
def code_interpreter(code, language="python"):
    """安全地执行代码"""
    # 限制：超时、内存、禁止系统调用
    result = subprocess.run(
        [language, "-c", code],
        capture_output=True,
        text=True,
        timeout=30
    )
    return result.stdout or result.stderr
```

### 21.6.2 使用场景

- **数据可视化**：生成图表
- **数学计算**：复杂公式求解
- **数据处理**：文件解析、转换
- **算法实现**：排序、搜索等

### 21.6.3 安全考虑

```python
SAFE_BUILTINS = {
    'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'float',
    'int', 'len', 'list', 'map', 'max', 'min', 'range', 'str', 'sum'
}

def safe_eval(code):
    # 限制可用的内置函数
    safe_globals = {"__builtins__": {k: __builtins__[k] for k in SAFE_BUILTINS}}
    return eval(code, safe_globals, {})
```

## 21.7 主流方案对比

### 21.7.1 OpenAI Function Calling

**优点**：
- 集成度高，使用简单
- 模型理解能力强
- 支持并行调用

**缺点**：
- 仅限 OpenAI 模型
- 成本较高

### 21.7.2 LangChain Tools

**优点**：
- 工具生态丰富
- 支持多种 LLM
- 链式编排灵活

**缺点**：
- 学习曲线较陡
- 依赖较多

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

tools = [
    Tool(
        name="Search",
        func=search_function,
        description="搜索网络信息"
    )
]

agent = initialize_agent(tools, OpenAI(), agent="zero-shot-react-description")
```

### 21.7.3 LlamaIndex Tools

**优点**：
- 专注于数据索引与检索
- 与向量数据库集成好

**缺点**：
- 通用工具支持较少

### 21.7.4 AutoGen Tools

**优点**：
- 多 Agent 协作支持
- 代码执行能力强

**缺点**：
- 配置复杂

### 21.7.5 对比表格

| 特性 | OpenAI | LangChain | LlamaIndex | AutoGen |
|------|--------|-----------|------------|---------|
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| 工具生态 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 多模型支持 | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 多 Agent | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 代码执行 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## 21.8 代码实现示例

### 21.8.1 完整示例架构

```
tool_use/
├── function_call.py    # FunctionCall 类
├── tool_registry.py    # ToolRegistry 注册表
├── tools/              # 工具实现
│   ├── calculator.py
│   ├── search.py
│   └── code_exec.py
└── agent.py            # Agent 主逻辑
```

### 21.8.2 核心类设计

详见 `code/tool_use.py`

## 21.9 最佳实践

### 21.9.1 工具设计原则

1. **单一职责**：每个工具只做一件事
2. **幂等性**：重复调用产生相同结果
3. **错误处理**：返回清晰的错误信息
4. **文档完善**：描述清晰，参数明确

### 21.9.2 性能优化

1. **缓存结果**：避免重复调用
2. **并行执行**：独立工具并发调用
3. **超时控制**：防止长时间阻塞
4. **限流保护**：避免 API 超限

### 21.9.3 安全考虑

1. **输入验证**：检查参数合法性
2. **权限控制**：限制敏感操作
3. **审计日志**：记录工具调用
4. **沙箱执行**：隔离代码执行

## 21.10 参考文献

1. OpenAI Function Calling Documentation: https://platform.openai.com/docs/guides/function-calling
2. LangChain Tools: https://python.langchain.com/docs/modules/agents/tools/
3. AutoGen: https://microsoft.github.io/autogen/
4. LlamaIndex Tools: https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/
5. "Function Calling for Large Language Models" - arXiv:2304.xxxxx
