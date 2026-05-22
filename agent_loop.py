"""
Advisor Agent Loop — MVP implementation.

Architecture:
  Executor (cheap model) runs the task with tools (file_edit, bash, ask_advisor).
  When it calls ask_advisor, a separate API request goes to the Advisor model.
  The advisor sees the full conversation and returns guidance (no tools).
  The executor continues with the guidance injected as tool_result.

This mirrors Anthropic's advisor_20260301 tool pattern,
but works with any OpenAI-compatible API (DeepSeek, GLM, etc.).
"""

import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from config import ModelConfig, get_model


# ─── Tool Definitions ───────────────────────────────────────────────

EXECUTOR_SYSTEM_PROMPT_SOLO = """\
You are a coding agent. You solve software engineering tasks by reading code, 
editing files, and running commands.

You have these tools:
- file_read: Read a file's content (use for exploration)
- file_edit: Replace an exact string in a file with a new string
- bash_run: Run a bash command and get output

Work methodically: explore the codebase first, identify the root cause, make
minimal targeted edits, then verify your fix. Be concise.
"""

EXECUTOR_SYSTEM_PROMPT_ADVISOR = """\
You are a coding agent. You solve software engineering tasks by reading code, 
editing files, and running commands.

You have these tools:
- file_read: Read a file's content (use for exploration)
- file_edit: Replace an exact string in a file with a new string
- bash_run: Run a bash command and get output
- ask_advisor: Consult a more capable model for strategic guidance

**When to call ask_advisor (follow these rules precisely):**

1. First, do exploratory work — read files, search the codebase, understand the structure
2. BEFORE writing any code or deciding on your approach, call ask_advisor with your findings and proposed plan
3. If you encounter an error you don't understand, call ask_advisor
4. If your approach fails after 1+ attempts, call ask_advisor
5. Before declaring the task done, call ask_advisor for a final review
6. For complex tasks, call ask_advisor at least twice (before implementation + before finalizing)
7. For simple tasks, you don't need frequent advisor calls — the first call is the most valuable

**When NOT to use ask_advisor:**
- Reading files or running simple commands during initial exploration (this is information gathering, not substantive work)

**How to treat advisor advice:**
- Take the advisor's guidance seriously — it has broader context
- If you have concrete evidence that contradicts the advice, adapt
- If you find a conflict between the advice and your findings, call ask_advisor again to reconcile

Be concise. Focus on solving the task. Show your reasoning briefly.
"""

# Default alias for backward compatibility
EXECUTOR_SYSTEM_PROMPT = EXECUTOR_SYSTEM_PROMPT_ADVISOR

ADVISOR_SYSTEM_PROMPT = """\
You are an advisor to a coding agent. The agent is working on a software engineering task.

Review the conversation history and provide strategic guidance. Your advice should be:
- A clear plan or course correction (not code to copy-paste)
- Focused on the key decision or obstacle
- CONCISE (under 80 words)

Do NOT write code. Do NOT call tools. Just provide strategic guidance.
If the agent is on the right track, say so briefly and let it continue.
If the agent should stop (task is already solved), say "STOP: the task appears to be solved."
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read the content of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_edit",
            "description": "Replace an exact string in a file with a new string. The old_string must match exactly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "old_string": {"type": "string", "description": "Exact string to find"},
                    "new_string": {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash_run",
            "description": "Run a bash command and return its output",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command to run"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)", "default": 30},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_advisor",
            "description": "Consult a more capable model for strategic guidance. Use when stuck, unsure about approach, or need architectural advice.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What you need guidance on. Be specific about where you're stuck.",
                    }
                },
                "required": ["question"],
            },
        },
    },
]


# ─── Metrics ─────────────────────────────────────────────────────────

@dataclass
class TurnMetrics:
    role: str  # "executor" or "advisor"
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


@dataclass
class RunMetrics:
    executor_model: str
    advisor_model: Optional[str]
    task_id: str
    turns: list = field(default_factory=list)
    advisor_calls: int = 0
    tool_calls: dict = field(default_factory=dict)  # tool_name -> count
    final_answer: str = ""
    error: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total_input_tokens(self):
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self):
        return sum(t.output_tokens for t in self.turns)

    @property
    def executor_input_tokens(self):
        return sum(t.input_tokens for t in self.turns if t.role == "executor")

    @property
    def executor_output_tokens(self):
        return sum(t.output_tokens for t in self.turns if t.role == "executor")

    @property
    def advisor_input_tokens(self):
        return sum(t.input_tokens for t in self.turns if t.role == "advisor")

    @property
    def advisor_output_tokens(self):
        return sum(t.output_tokens for t in self.turns if t.role == "advisor")

    @property
    def total_seconds(self):
        return self.end_time - self.start_time if self.end_time else 0

    def cost_usd(self, executor_cfg: ModelConfig, advisor_cfg: ModelConfig = None):
        e_cost = (self.executor_input_tokens / 1e6 * executor_cfg.cost_per_million_input +
                  self.executor_output_tokens / 1e6 * executor_cfg.cost_per_million_output)
        a_cost = 0
        if advisor_cfg and self.advisor_input_tokens > 0:
            a_cost = (self.advisor_input_tokens / 1e6 * advisor_cfg.cost_per_million_input +
                      self.advisor_output_tokens / 1e6 * advisor_cfg.cost_per_million_output)
        return e_cost + a_cost


# ─── Agent Loop ──────────────────────────────────────────────────────

class AdvisorAgent:
    """Agent loop with advisor support — the core MVP."""

    def __init__(
        self,
        executor_model: str,
        advisor_model: str = None,
        max_turns: int = 15,
        workdir: str = "/tmp/advisor-workspace",
        verbose: bool = True,
    ):
        self.executor_cfg = get_model(executor_model)
        self.advisor_cfg = get_model(advisor_model) if advisor_model else None
        self.max_turns = max_turns
        self.workdir = workdir
        self.verbose = verbose
        self.include_advisor_tool = True

        # Create OpenAI clients
        self.executor_client = OpenAI(
            api_key=self.executor_cfg.api_key,
            base_url=self.executor_cfg.base_url,
        )
        if self.advisor_cfg:
            self.advisor_client = OpenAI(
                api_key=self.advisor_cfg.api_key,
                base_url=self.advisor_cfg.base_url,
            )
        else:
            self.advisor_client = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"  {msg}")

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool call and return the result."""
        import subprocess

        if tool_name == "file_read":
            path = args["path"]
            if not path.startswith("/"):
                path = f"{self.workdir}/{path}"
            try:
                with open(path, "r") as f:
                    content = f.read()
                # Truncate very long files
                if len(content) > 10000:
                    content = content[:10000] + f"\n... (truncated, {len(content)} chars total)"
                return content
            except FileNotFoundError:
                return f"Error: File not found: {path}"
            except Exception as e:
                return f"Error reading file: {e}"

        elif tool_name == "file_edit":
            path = args["path"]
            if not path.startswith("/"):
                path = f"{self.workdir}/{path}"
            try:
                with open(path, "r") as f:
                    content = f.read()
                old = args["old_string"]
                new = args["new_string"]
                if old not in content:
                    return f"Error: old_string not found in {path}"
                content = content.replace(old, new, 1)
                with open(path, "w") as f:
                    f.write(content)
                return f"Successfully edited {path}"
            except Exception as e:
                return f"Error editing file: {e}"

        elif tool_name == "bash_run":
            cmd = args["command"]
            timeout = args.get("timeout", 30)
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=timeout, cwd=self.workdir,
                )
                output = result.stdout
                if result.stderr:
                    output += f"\nSTDERR:\n{result.stderr}"
                if result.returncode != 0:
                    output += f"\nExit code: {result.returncode}"
                # Truncate
                if len(output) > 8000:
                    output = output[:8000] + "... (truncated)"
                return output
            except subprocess.TimeoutExpired:
                return f"Error: Command timed out after {timeout}s"
            except Exception as e:
                return f"Error running command: {e}"

        elif tool_name == "ask_advisor":
            # Handle both "question" and "message" parameter names
            question = args.get("question") or args.get("message") or str(args)
            return self._call_advisor(question)

        else:
            return f"Error: Unknown tool {tool_name}"

    def _sanitize_for_advisor(self, messages: list) -> list:
        """Clean messages for advisor: strip tool_calls/tool roles, convert to pure text."""
        clean = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "tool":
                # Convert tool results to user message
                clean.append({"role": "user", "content": f"[Tool output]: {content}"})
            elif role == "assistant" and msg.get("tool_calls"):
                # Extract text + tool calls as readable summary
                parts = []
                if content:
                    parts.append(content)
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    parts.append(f"[Called {fn.get('name','?')}({fn.get('arguments','')[:200]})]")
                clean.append({"role": "assistant", "content": "\n".join(parts)})
            elif content:
                clean.append({"role": role, "content": content})
        return clean

    def _call_advisor(self, question: str) -> str:
        """Call the advisor model with conversation context."""
        if not self.advisor_client:
            return "Error: No advisor model configured."

        # Build advisor messages: system + cleaned conversation + question
        advisor_messages = [{"role": "system", "content": ADVISOR_SYSTEM_PROMPT}]
        advisor_messages.extend(self._sanitize_for_advisor(self.messages))

        # Add the specific question with output trimming directive
        advisor_messages.append({
            "role": "user",
            "content": f"[Advisor request]: {question}\n\n(Advisor: please keep your guidance under 80 words — I need a focused starting point, not a comprehensive plan.)",
        })

        t0 = time.time()
        try:
            resp = self.advisor_client.chat.completions.create(
                model=self.advisor_cfg.name,
                messages=advisor_messages,
                max_tokens=1024,  # Advisor should be concise
                temperature=0.3,
            )
            latency = int((time.time() - t0) * 1000)

            advice = resp.choices[0].message.content

            # Track metrics
            usage = resp.usage
            self.metrics.turns.append(TurnMetrics(
                role="advisor",
                model=self.advisor_cfg.name,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency_ms=latency,
            ))
            self.metrics.advisor_calls += 1

            self._log(f"🧠 Advisor ({self.advisor_cfg.name}): {advice[:200]}")
            return advice

        except Exception as e:
            self._log(f"⚠️ Advisor error: {e}")
            return f"Advisor unavailable: {e}"

    def run(self, task: str, task_id: str = "default") -> RunMetrics:
        """
        Run the agent loop on a task.

        Args:
            task: The task description (e.g., a SWE-bench issue)
            task_id: Identifier for tracking

        Returns:
            RunMetrics with full statistics
        """
        self.metrics = RunMetrics(
            executor_model=self.executor_cfg.name,
            advisor_model=self.advisor_cfg.name if self.advisor_cfg else None,
            task_id=task_id,
            start_time=time.time(),
        )

        # Use advisor-aware or solo system prompt
        system_prompt = EXECUTOR_SYSTEM_PROMPT_ADVISOR if self.include_advisor_tool else EXECUTOR_SYSTEM_PROMPT_SOLO
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        self._log(f"🚀 Starting: executor={self.executor_cfg.name} "
                   f"advisor={self.advisor_cfg.name if self.advisor_cfg else 'none'}")

        for turn in range(self.max_turns):
            self._log(f"--- Turn {turn + 1}/{self.max_turns} ---")

            t0 = time.time()
            try:
                # Disable thinking mode for executors that support it (DeepSeek V4 Flash)
                extra_kwargs = {}
                if hasattr(self.executor_cfg, 'disable_thinking') and self.executor_cfg.disable_thinking:
                    extra_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                # Build tools list (optionally exclude ask_advisor)
                tools = TOOLS if self.executor_cfg.supports_tools else []
                if not self.include_advisor_tool:
                    tools = [t for t in TOOLS if t.get("function", {}).get("name") != "ask_advisor"]
                resp = self.executor_client.chat.completions.create(
                    model=self.executor_cfg.name,
                    messages=self.messages,
                    tools=tools,
                    max_tokens=self.executor_cfg.max_tokens,
                    temperature=0.2,
                    **extra_kwargs,
                )
            except Exception as e:
                self.metrics.error = f"Executor API error: {e}"
                self._log(f"❌ API error: {e}")
                break

            latency = int((time.time() - t0) * 1000)
            usage = resp.usage
            self.metrics.turns.append(TurnMetrics(
                role="executor",
                model=self.executor_cfg.name,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency_ms=latency,
            ))

            msg = resp.choices[0].message
            finish_reason = resp.choices[0].finish_reason

            # Handle GLM/DeepSeek thinking mode: content may be empty, actual text in reasoning_content
            content = msg.content or ""
            reasoning_content = getattr(msg, "reasoning_content", None)
            if not content.strip() and reasoning_content:
                content = reasoning_content

            # Add assistant response to history
            assistant_msg = {"role": "assistant", "content": content}
            if reasoning_content:
                assistant_msg["reasoning_content"] = reasoning_content
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            self.messages.append(assistant_msg)

            # Check if done
            if not msg.tool_calls or finish_reason == "stop":
                self.metrics.final_answer = content
                self._log(f"✅ Done (finish_reason={finish_reason})")
                break

            # Execute tool calls
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)

                # Track tool usage
                self.metrics.tool_calls[tool_name] = self.metrics.tool_calls.get(tool_name, 0) + 1
                self._log(f"🔧 {tool_name}({json.dumps(tool_args)[:100]})")

                result = self._execute_tool(tool_name, tool_args)

                # Add tool result to conversation
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

        else:
            self.metrics.error = "Max turns reached"

        self.metrics.end_time = time.time()
        self._log(f"⏱️ Total: {self.metrics.total_seconds:.1f}s, "
                   f"turns={len([t for t in self.metrics.turns if t.role=='executor'])}, "
                   f"advisor_calls={self.metrics.advisor_calls}")
        return self.metrics


# ─── Convenience ─────────────────────────────────────────────────────

def run_task(
    task: str,
    executor: str = "deepseek-chat",
    advisor: str = None,
    max_turns: int = 15,
    workdir: str = "/tmp/advisor-workspace",
) -> RunMetrics:
    """One-liner to run a task."""
    agent = AdvisorAgent(
        executor_model=executor,
        advisor_model=advisor,
        max_turns=max_turns,
        workdir=workdir,
    )
    return agent.run(task)
