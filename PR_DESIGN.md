# Advisor Tool PR 设计文档

> 给 Hermes Agent 添加 `ask_advisor` 工具的 PR 规划

## 核心理念

**Advisor Strategy** 翻转了传统的 orchestrator→worker 模式：

```
传统：强模型（贵）→ 派任务给弱模型（便宜）
Advisor：弱模型（便宜）主导 → 遇到困难时咨询强模型（贵）
```

关键区别：
- Executor **不交出控制权**，advisor 只是出主意
- Advisor **看不到工具调用细节**，只看清洗过的对话历史
- 成本 **低**：一次咨询 ~500-1000 tokens，不是跑一个完整的子任务

## 与 Hermes 现有能力的关系

| 能力 | 方向 | 控制权 | 成本 |
|------|------|--------|------|
| `delegate_task` | 强→弱（派活） | Child 独立 | 高（整个子任务） |
| `mixture_of_agents` | 多强并行 | 聚合器决定 | 极高（4×前沿模型） |
| **`ask_advisor`（新）** | **弱→强（求教）** | **Executor 保留** | **低（按需咨询）** |

三种能力互补，不重叠。

## 涉及的文件

### 新增文件

```
tools/advisor_tool.py          # 核心实现（~250 行）
```

### 修改文件

```
tools/toolsets.py              # 注册 toolset（+5 行）
model_tools.py                 # 加入 _AGENT_LOOP_TOOLS（+1 行）
agent/agent_runtime_helpers.py # invoke_tool 加 elif 分支（+15 行）
config.yaml                    # 文档示例（+10 行）
```

### 不需要修改的文件

```
run_agent.py                   # invoke_tool 已经是统一入口
tools/registry.py              # register() API 不变
cli.py                         # 不需要新的 CLI 命令
```

## 详细代码设计

### 1. `tools/advisor_tool.py`（核心，~250 行）

```python
#!/usr/bin/env python3
"""
Advisor Tool — 让 Agent 在执行中向更强的模型寻求建议。

Executor agent 遇到不确定的决策时，调用 ask_advisor 工具，
向配置的 advisor 模型发送当前上下文（清洗后），获取建议后继续执行。

设计参考：Anthropic Advisor Strategy
实现参考：advisor-eval 项目 (github.com/jasonjcwu/advisor-eval)
"""

import json
import logging
import time
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ── 配置加载 ──────────────────────────────────────

def _load_advisor_config() -> dict:
    """从 Hermes config 加载 advisor 配置。"""
    defaults = {
        "enabled": False,
        "model": None,           # None = 使用当前 provider 的最强模型
        "provider": None,        # None = 使用当前 provider
        "base_url": None,        # None = 使用 provider 默认
        "api_key": None,         # None = 从 credential pool 获取
        "max_uses_per_turn": 3,  # 每次对话最多咨询次数
        "max_uses_total": 10,    # 单个 session 总咨询上限
        "temperature": 0.3,      # advisor 偏低温度，输出稳定
        "max_tokens": 2048,      # 单次建议最大 token
        "system_prompt": None,   # 可自定义 advisor 系统提示
    }
    try:
        from cli import CLI_CONFIG
        cfg = CLI_CONFIG.get("advisor") or {}
    except Exception:
        try:
            from hermes_cli.config import load_config
            full = load_config()
            cfg = full.get("advisor") or {}
        except Exception:
            cfg = {}
    defaults.update({k: v for k, v in cfg.items() if v is not None})
    return defaults

# ── 消息清洗 ──────────────────────────────────────

def sanitize_messages_for_advisor(messages: List[dict]) -> List[dict]:
    """
    清洗 executor 的对话历史，只保留 advisor 能理解的格式。
    
    关键：advisor 不支持 tool_calls/tool role，需要转换。
    - tool_calls 消息 → 提取文本内容，忽略 tool_calls
    - tool role 消息 → 转成 user 消息，附上工具执行结果摘要
    - 保留 system/user/assistant 的文本内容
    """
    sanitized = []
    for msg in messages:
        role = msg.get("role", "")
        
        if role == "system":
            sanitized.append({"role": "system", "content": msg.get("content", "")})
            
        elif role == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # 多模态内容，只取文本部分
                texts = [p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = "\n".join(texts)
            if content:
                sanitized.append({"role": "user", "content": content})
                
        elif role == "assistant":
            content = msg.get("content", "") or ""
            # 如果有 tool_calls，生成摘要而不是丢弃
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                call_summary = "; ".join(
                    f"调用 {tc['function']['name']}({tc['function']['arguments'][:100]}...)"
                    for tc in tool_calls
                )
                content = f"{content}\n[执行动作: {call_summary}]" if content else f"[执行动作: {call_summary}]"
            if content.strip():
                sanitized.append({"role": "assistant", "content": content})
                
        elif role == "tool":
            # 工具结果 → user 消息
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 500:
                content = content[:500] + "...(截断)"
            sanitized.append({"role": "user", "content": f"[工具结果] {content}"})
    
    return sanitized

# ── 核心调用 ──────────────────────────────────────

ADVISOR_SYSTEM_PROMPT = """你是一个资深技术顾问（Advisor）。一个 AI 助手在执行任务时遇到了困难，向你寻求建议。

你的职责：
1. 分析当前进展和问题
2. 给出具体的、可操作的建议
3. 指出可能的盲区或风险
4. 如果有更好的方法，直接说明

注意：
- 你不需要执行任何操作，只需要给出建议
- 建议要具体到可以直接执行的程度
- 如果当前方向没问题，简短确认即可"""

async def call_advisor(
    messages: List[dict],
    question: str,
    config: dict,
    api_client=None,
) -> dict:
    """
    调用 advisor 模型获取建议。
    
    Args:
        messages: executor 的对话历史（原始，会自动清洗）
        question: executor 提出的具体问题
        config: advisor 配置
        api_client: OpenAI 兼容客户端（由 agent 注入）
    
    Returns:
        {
            "advice": str,           # advisor 的建议
            "model": str,            # 使用的模型
            "tokens_in": int,        # 输入 token 数
            "tokens_out": int,       # 输出 token 数
            "latency_ms": int,       # 延迟毫秒
        }
    """
    # 清洗消息
    clean_messages = sanitize_messages_for_advisor(messages)
    
    # 构造 advisor 的输入
    advisor_messages = [
        {"role": "system", "content": config.get("system_prompt") or ADVISOR_SYSTEM_PROMPT},
    ]
    
    # 添加上下文（最近 N 条，避免太长）
    context_window = clean_messages[-20:]  # 最近 20 条
    if context_window:
        advisor_messages.append({
            "role": "user", 
            "content": "以下是执行者当前的对话上下文：\n" + 
                       json.dumps([{"role": m["role"], "content": m["content"][:200]} for m in context_window], 
                                  ensure_ascii=False, indent=2)
        })
    
    # 添加具体问题
    advisor_messages.append({
        "role": "user",
        "content": f"我的问题是：{question}\n\n请给出建议。"
    })
    
    # 调用 API
    start = time.time()
    
    if api_client is None:
        raise ValueError("advisor tool requires an API client")
    
    response = await api_client.chat.completions.create(
        model=config["model"],
        messages=advisor_messages,
        temperature=config.get("temperature", 0.3),
        max_tokens=config.get("max_tokens", 2048),
    )
    
    latency_ms = int((time.time() - start) * 1000)
    choice = response.choices[0]
    
    return {
        "advice": choice.message.content,
        "model": config.get("model", "unknown"),
        "tokens_in": response.usage.prompt_tokens,
        "tokens_out": response.usage.completion_tokens,
        "latency_ms": latency_ms,
    }

# ── Tool Handler ──────────────────────────────────

# 全局计数器（per-session 重置）
_advisor_use_count = {"per_turn": 0, "total": 0}

def ask_advisor_handler(args: dict, **kwargs) -> str:
    """
    ask_advisor 工具的 handler。
    
    注意：实际实现需要 agent 注入 messages 和 api_client。
    这里是同步包装，真正调用在 invoke_tool() 中处理。
    """
    # 这个 handler 不会被直接调用——由 invoke_tool 拦截
    # 但 registry 需要一个 handler，所以放个占位
    return json.dumps({"error": "advisor tool must be dispatched via invoke_tool"})

def check_advisor_requirements() -> bool:
    """检查 advisor 是否可用。"""
    config = _load_advisor_config()
    return config.get("enabled", False)

# ── OpenAI Schema ─────────────────────────────────

ASK_ADVISOR_SCHEMA = {
    "name": "ask_advisor",
    "description": (
        "向一个更强的 AI 顾问寻求建议。"
        "当你不确定最佳方案、遇到复杂架构决策、"
        "或者多次尝试仍未解决问题时使用。"
        "顾问会基于当前上下文给出具体建议，但不会替你执行。"
        "请在提问时说明：1)你尝试了什么 2)遇到了什么困难 3)你需要的建议类型"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "你向顾问提出的具体问题。包含：当前进展、遇到的困难、需要什么类型的建议。"
            },
            "urgency": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "紧急程度。low=想听听意见，medium=需要指导，high=卡住了必须求助"
            },
        },
        "required": ["question"],
    },
}

# ── 注册 ──────────────────────────────────────────

from tools.registry import registry, tool_error

registry.register(
    name="ask_advisor",
    toolset="advisor",
    schema=ASK_ADVISOR_SCHEMA,
    handler=ask_advisor_handler,
    check_fn=check_advisor_requirements,
    emoji="🧠",
    is_async=False,
)
```

### 2. `tools/toolsets.py`（修改）

在 `TOOLSETS` 字典中加入：

```python
"advisor": {
    "description": "AI advisor — 向更强的模型寻求建议",
    "tools": ["ask_advisor"],
    "includes": [],
},
```

在 `_HERMES_CORE_TOOLS` set 中加入 `"ask_advisor"`。

### 3. `agent/agent_runtime_helpers.py`（修改 `invoke_tool`）

在 `_AGENT_LOOP_TOOLS` set（`model_tools.py`）中加入 `"ask_advisor"`。

在 `invoke_tool()` 中加入 elif 分支：

```python
elif function_name == "ask_advisor":
    # Advisor tool: 需要 agent 的 messages 和 API client
    config = _load_advisor_config()
    
    # 检查使用次数
    if advisor_use_count.get("total", 0) >= config.get("max_uses_total", 10):
        return tool_error("advisor", "已达本次对话的咨询上限")
    
    # 获取 advisor 模型的 API client
    advisor_client = agent._get_advisor_client(config)
    
    # 调用 advisor
    result = await call_advisor(
        messages=agent.messages,          # 当前对话历史
        question=args.get("question", ""),
        config=config,
        api_client=advisor_client,
    )
    
    # 更新计数
    advisor_use_count["total"] += 1
    
    # 格式化返回给 executor
    advice_text = result["advice"]
    stats = f"[advisor: {result['model']}, {result['tokens_in']}in/{result['tokens_out']}out, {result['latency_ms']}ms]"
    
    return f"{stats}\n\n顾问建议：\n{advice_text}"
```

### 4. `config.yaml`（用户配置示例）

```yaml
# Advisor Strategy 配置
# 让 Agent 在执行中向更强的模型寻求建议
advisor:
  enabled: true
  
  # Advisor 模型配置（覆盖默认 provider）
  model: "claude-sonnet-4"           # Anthropic
  # model: "deepseek-chat"           # DeepSeek
  # model: "glm-5.1"                 # GLM
  
  # Provider（不设则用当前 provider）
  # provider: "anthropic"
  # provider: "custom:deepseek"
  
  # 使用限制
  max_uses_per_turn: 3               # 单轮最多咨询次数
  max_uses_total: 10                 # 单次对话总上限
  
  # 生成参数
  temperature: 0.3                   # advisor 偏稳定
  max_tokens: 2048                   # 单次建议上限
  
  # 自定义系统提示（可选）
  # system_prompt: "你是一个..."
```

## 关键设计决策

### Q1: Advisor 是 server-side tool 还是 client-side interception？

**答：和 delegate_task 一样，server-side tool + agent 拦截。**

原因：
- `ask_advisor` 在 schema 里是普通工具，executor 模型看到它和其他工具没区别
- 但 handler 需要访问 agent 的 `messages` 和 API client，所以走 `invoke_tool` 拦截
- 这和 `delegate_task`、`memory`、`clarify` 的模式完全一致

### Q2: Advisor 的 API client 从哪来？

**答：复用 Hermes 的 provider 机制。**

Hermes 已经有完整的 provider → client 映射（anthropic, openai, custom:xxx）。Advisor 配置里指定 `model` + `provider`，agent runtime 用 `_get_client_for_provider()` 获取 client。

如果用户不指定 provider，用当前对话的主 provider。

### Q3: 消息清洗为什么重要？

**答：Advisor 模型不支持 tools，必须把对话历史转换成纯文本。**

Executor 的 messages 里混杂了 `tool_calls`、`tool` role 等特殊格式。如果直接传给 advisor，API 会报错或产生幻觉。`sanitize_messages_for_advisor()` 把这些转成 advisor 能理解的纯文本。

这是 advisor-eval 项目里踩过的坑，已经验证过。

### Q4: 如何防止滥用（executor 每轮都问）？

三层保护：
1. `max_uses_per_turn` — 单轮上限
2. `max_uses_total` — 对话总上限
3. Schema description 里引导 executor 只在真正需要时使用

### Q5: 异步还是同步？

**答：用 async（和 delegate_task 一致）。**

advisor 调用是网络 I/O，用 async 避免阻塞 agent loop。`registry.register(is_async=True)`。

## 实现路线

```
Phase 1: 核心功能（可提 PR）
├── tools/advisor_tool.py         # 核心实现
├── tools/toolsets.py             # 注册 toolset
├── model_tools.py                # 加入 _AGENT_LOOP_TOOLS
└── agent/agent_runtime_helpers.py # invoke_tool 拦截

Phase 2: 增强功能（后续 PR）
├── 支持 streaming（advisor 回复流式注入）
├── 支持 advisor cache（相同上下文复用）
├── 审计日志（记录每次咨询的 question/advice/stats）
└── UI 展示（在 TUI/WebUI 里标记 advisor 交互）

Phase 3: 高级玩法（社区贡献）
├── 多 advisor（不同专长的模型）
├── Advisor chain（层层升级）
└── 自适应触发（基于 executor 困惑度自动咨询）
```

## 测试计划

### 单元测试
- `sanitize_messages_for_advisor()` — 各种消息格式
- `_load_advisor_config()` — 默认值、config override、env override
- `check_advisor_requirements()` — enabled/disabled

### 集成测试
- Executor 调用 `ask_advisor` → advisor 返回建议 → executor 继续
- 使用次数限制
- advisor API 错误处理（超时、限流、key 无效）

### 端到端测试
- 简单任务：executor 不需要 advisor
- 困难任务：executor 主动咨询 advisor，解决问题
- advisor-eval 的 SWE-bench 数据作为回归测试

## 与 advisor-eval 项目的关系

`advisor-eval` 是评测项目（验证 advisor 是否有效），这个 PR 是把验证过的模式固化到 Hermes 里。

```
advisor-eval (评测)          →  数据证明有效
Hermes advisor tool (PR)     →  把模式变成产品功能
```

评测数据直接作为 PR 的支撑材料提交。
