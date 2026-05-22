"""
Anthropic API adapter — wraps PackyAPI Anthropic endpoint as OpenAI-compatible client.

PackyAPI uses the standard Anthropic Messages API format:
  POST /v1/messages
  x-api-key: <key>
  anthropic-version: 2023-06-01

This adapter translates OpenAI-format chat completion requests into
Anthropic-format messages requests and converts responses back.
"""

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class AnthropicUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AnthropicToolCall:
    id: str
    function: Any


@dataclass
class AnthropicFunction:
    name: str
    arguments: str


@dataclass
class AnthropicMessage:
    content: str
    tool_calls: list = None
    reasoning_content: str = None


@dataclass
class AnthropicChoice:
    message: AnthropicMessage
    finish_reason: str = "stop"


def _openai_tools_to_anthropic(tools: list) -> list:
    """Convert OpenAI tool definitions to Anthropic format."""
    anthropic_tools = []
    for t in tools:
        fn = t.get("function", t)
        anthropic_tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return anthropic_tools


def _openai_messages_to_anthropic(messages: list, system_content: str = None) -> tuple:
    """Convert OpenAI-format messages to Anthropic format.
    
    Returns (system_text, anthropic_messages).
    Anthropic separates system from messages.
    """
    system_parts = []
    anthropic_msgs = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            if content:
                system_parts.append(content)
        elif role == "user":
            if isinstance(content, str) and content.strip():
                anthropic_msgs.append({"role": "user", "content": content})
        elif role == "assistant":
            text = content if isinstance(content, str) else ""
            tool_calls = msg.get("tool_calls", [])
            anthropic_content = []
            
            if text.strip():
                anthropic_content.append({"type": "text", "text": text})
            
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                anthropic_content.append({
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                    "name": fn.get("name", ""),
                    "input": args,
                })
            
            if anthropic_content:
                anthropic_msgs.append({"role": "assistant", "content": anthropic_content})
            elif text.strip():
                anthropic_msgs.append({"role": "assistant", "content": text})
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            tool_content = content if isinstance(content, str) else str(content)
            anthropic_msgs.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": tool_content[:8000],  # Truncate
                }],
            })
    
    system_text = "\n".join(system_parts) if system_parts else None
    return system_text, anthropic_msgs


def _anthropic_response_to_openai(resp: dict, messages: list = None) -> tuple:
    """Convert Anthropic response to OpenAI-compatible format.
    
    Returns (choices, usage).
    """
    content_blocks = resp.get("content", [])
    text_parts = []
    tool_calls = []
    reasoning_content = None
    
    for block in content_blocks:
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            tool_calls.append({
                "id": block.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                },
            })
        elif block_type == "thinking":
            reasoning_content = block.get("thinking", "")
    
    # If there's a redacted_thinking block, extract from that too
    for block in content_blocks:
        if block.get("type") == "redacted_thinking":
            data = block.get("data", "")
            if data and not reasoning_content:
                reasoning_content = f"[redacted thinking: {len(data)} chars]"
    
    stop_reason = resp.get("stop_reason", "end_turn")
    finish_reason = "tool_calls" if tool_calls else ("stop" if stop_reason == "end_turn" else "length")
    
    msg = AnthropicMessage(
        content="\n".join(text_parts),
        tool_calls=tool_calls if tool_calls else None,
        reasoning_content=reasoning_content,
    )
    
    usage = resp.get("usage", {})
    usage_obj = AnthropicUsage(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
    )
    
    choice = AnthropicChoice(message=msg, finish_reason=finish_reason)
    
    # Handle thinking/reasoning content — check for thinking blocks
    for block in content_blocks:
        if block.get("type") == "thinking":
            msg.reasoning_content = block.get("thinking", "")
            break
    
    return [choice], usage_obj


class AnthropicClient:
    """Minimal Anthropic API client compatible with OpenAI client interface."""
    
    def __init__(self, api_key: str, base_url: str, timeout: int = 300):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.chat = self  # For openai-style client.chat.completions.create()
        self.completions = self
    
    def create(self, *, model: str, messages: list, tools: list = None,
               max_tokens: int = 4096, temperature: float = 0.2,
               extra_body: dict = None, **kwargs):
        """OpenAI-compatible chat.completions.create().
        
        Translates to Anthropic /v1/messages and converts response back.
        """
        import urllib.request
        import urllib.error
        
        # Build system prompt
        system_text, anthropic_msgs = _openai_messages_to_anthropic(messages)
        
        # Build request body
        body = {
            "model": model,
            "messages": anthropic_msgs,
            "max_tokens": max_tokens,
        }
        
        if system_text:
            body["system"] = system_text
        
        # Add tools if present
        if tools:
            body["tools"] = _openai_tools_to_anthropic(tools)
        
        # Handle thinking mode toggle
        if extra_body and extra_body.get("thinking", {}).get("type") == "disabled":
            body["thinking"] = {"type": "disabled"}
        elif extra_body and extra_body.get("thinking", {}).get("type") == "enabled":
            body["thinking"] = {"type": "enabled", "budget_tokens": 4000}
        
        # Temperature — Anthropic doesn't support temp with thinking
        if temperature and "thinking" not in body:
            body["temperature"] = temperature
        
        # Make request
        url = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            raise Exception(f"API error {e.code}: {error_body}")
        
        choices, usage = _anthropic_response_to_openai(response_data)
        
        # Build a response object that looks like openai's
        class FakeResponse:
            def __init__(self, choices, usage):
                self.choices = choices
                self.usage = usage
        
        return FakeResponse(choices, usage)
