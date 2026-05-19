# Advisor Eval — 自建 Advisor Strategy 评测框架

> 复现 Anthropic 的 advisor tool 模式，用 DeepSeek + GLM 验证效果。

## 背景

Anthropic 发布了 [advisor strategy](https://claude.com/blog/the-advisor-strategy)：
让便宜的模型（executor）跑任务，遇到困难时咨询贵模型（advisor）。
结果：接近 advisor-solo 的质量，成本大幅降低。

本项目**自建** advisor 机制（不依赖 Anthropic 服务端工具），用中国模型的 API 验证同样思路是否成立。

## 架构

```
Executor (DeepSeek/GLM, 便宜) ──跑工具──→ file_edit / bash_run
      │
      │ 卡住了？需要架构决策？
      ▼
Advisor (更强模型, 贵) ──纯文本──→ 返回 plan/correction (不调工具)
      │
      ▼
Executor 继续执行 (带着 advisor 的指导)
```

核心区别于 Anthropic 原版：
- 原版是服务端 tool（`advisor_20260301`），只有 Claude 系列能用
- 我们是客户端实现，**任何 OpenAI 兼容 API 都能用**
- executor 把 `ask_advisor` 当普通 tool 调用，我们在客户端拦截并转发给 advisor 模型

## 快速开始

```bash
# 1. 安装依赖
pip install openai pyyaml datasets

# 2. 跑 demo（简单 bug-fix，验证链路）
python3 swe_runner.py --demo

# 3. 跑单个 SWE-bench 任务
python3 swe_runner.py --executor deepseek-chat --advisor glm-5.1

# 4. 跑 18 题子集
python3 swe_runner.py --matrix --subset 18
```

## 文件结构

| 文件 | 说明 |
|------|------|
| `agent_loop.py` | 核心：带 advisor 的 agent loop（~500行） |
| `config.py` | 模型配置 + API key 自动发现 |
| `swe_runner.py` | SWE-bench Multilingual 集成 |
| `results/` | 输出目录 |

## 评测矩阵

| Executor | Advisor | 说明 |
|----------|---------|------|
| deepseek-chat | solo | 基线 |
| deepseek-chat | glm-5.1 | DS+GLM advisor |
| deepseek-v4-flash | glm-5.1 | 便宜 executor+GLM |
| deepseek-v4-flash | deepseek-chat | flash+pro |
| glm-5.1 | solo | 基线 |
| glm-5.1 | deepseek-chat | GLM+DS advisor |

## 关键设计决策

### 1. Advisor 消息清洗
executor 的对话包含 `tool_calls` 和 `tool` 角色，advisor API 不支持。
`_sanitize_for_advisor()` 把它们转成纯文本：
- `tool` → `[Tool output]: ...`
- `assistant(tool_calls)` → `[Called file_edit(...)]`

### 2. Executor System Prompt
引导 executor 在合适时机调用 advisor：
- 开始前求架构指导
- 遇到不懂的错误
- 复杂逻辑任务

### 3. 指标追踪
每次运行记录：turns、advisor_calls、token 用量（分 executor/advisor）、成本、tool 调用分布。
