"""
Advisor Eval — Model configurations and API keys.

Supports: DeepSeek (deepseek-chat, deepseek-v4-flash/pro), GLM (glm-5.1)
All use OpenAI-compatible API.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    name: str
    base_url: str
    api_key: str
    max_tokens: int = 4096
    supports_tools: bool = True
    cost_per_million_input: float = 0.0
    cost_per_million_output: float = 0.0


def _load_keys_from_hermes():
    """Extract API keys from Hermes auth.json."""
    import json
    auth_path = os.path.expanduser("~/.hermes/auth.json")
    if not os.path.exists(auth_path):
        return {}
    
    with open(auth_path) as f:
        auth = json.load(f)
    
    pool = auth.get("credential_pool", {})
    keys = {}
    
    # GLM
    glm_creds = pool.get("custom:glmcode", [])
    if glm_creds:
        keys["glm"] = {
            "api_key": glm_creds[0]["access_token"],
            "base_url": glm_creds[0]["base_url"],
        }
    
    # DeepSeek (try both sources)
    ds_creds = pool.get("deepseek", []) or pool.get("custom:deepseek", [])
    if ds_creds:
        keys["deepseek"] = {
            "api_key": ds_creds[0]["access_token"],
            "base_url": ds_creds[0].get("base_url", "https://api.deepseek.com/v1"),
        }
    
    return keys


# Auto-discover keys
_KEYS = _load_keys_from_hermes()

MODELS = {
    "deepseek-chat": ModelConfig(
        name="deepseek-chat",
        base_url=_KEYS.get("deepseek", {}).get("base_url", "https://api.deepseek.com/v1"),
        api_key=_KEYS.get("deepseek", {}).get("api_key", os.environ.get("DEEPSEEK_API_KEY", "")),
        max_tokens=8192,
        supports_tools=True,
        cost_per_million_input=0.27,
        cost_per_million_output=1.10,
    ),
    "deepseek-v4-flash": ModelConfig(
        name="deepseek-v4-flash",
        base_url=_KEYS.get("deepseek", {}).get("base_url", "https://api.deepseek.com/v1"),
        api_key=_KEYS.get("deepseek", {}).get("api_key", os.environ.get("DEEPSEEK_API_KEY", "")),
        max_tokens=8192,
        supports_tools=True,
        cost_per_million_input=0.07,
        cost_per_million_output=0.30,
    ),
    "glm-5.1": ModelConfig(
        name="glm-5.1",
        base_url=_KEYS.get("glm", {}).get("base_url", "https://open.bigmodel.cn/api/coding/paas/v4"),
        api_key=_KEYS.get("glm", {}).get("api_key", os.environ.get("GLMCODE_API_KEY", "")),
        max_tokens=4096,
        supports_tools=True,
        cost_per_million_input=0.50,
        cost_per_million_output=2.00,
    ),
}


def get_model(name: str) -> ModelConfig:
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    cfg = MODELS[name]
    if not cfg.api_key:
        raise ValueError(f"No API key for {name}. Set env var or check hermes auth.json")
    return cfg


# Pre-defined executor/advisor pairs for evaluation
EVAL_PAIRS = [
    # (executor, advisor) — None means no advisor
    ("deepseek-chat", None),
    ("deepseek-chat", "glm-5.1"),
    ("deepseek-v4-flash", "glm-5.1"),
    ("deepseek-v4-flash", "deepseek-chat"),
    ("glm-5.1", None),
    ("glm-5.1", "deepseek-chat"),
]
