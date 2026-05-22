#!/usr/bin/env python3
"""
Advisor Loop Evaluation Framework

Core idea: executor works in a loop. Each round it decides:
- Am I confident? → submit solution
- Am I stuck? → ask advisor, then continue

Goal: prove advisor loop beats executor-only on:
  1. Better completion (more tests pass)
  2. FEWER total tokens (advisor guidance prevents wasted effort)

Usage:
  python3 eval_loop.py --config solo          # executor-only baseline
  python3 eval_loop.py --config advisor        # executor + advisor loop
  python3 eval_loop.py --config advisor-v2     # different advisor prompt
  python3 eval_loop.py --all                   # run all configs
  python3 eval_loop.py --compare               # compare results
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Eval set
# ---------------------------------------------------------------------------

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def load_eval_set() -> List[Dict]:
    with open(EVAL_SET_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _load_key(env_var: str) -> str:
    key = os.environ.get(env_var)
    if key:
        return key
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{env_var}=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def call_llm(model: str, messages: List[Dict], api_key: str,
             base_url: str, temperature: float = 0.2,
             max_tokens: int = 2048, timeout: int = 60) -> Dict:
    """Call OpenAI-compatible LLM. Returns {content, tokens_in, tokens_out, latency_s}."""
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    start = time.time()
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens, timeout=timeout,
    )
    latency = time.time() - start
    content = resp.choices[0].message.content or ""
    # reasoning_content fallback
    if not content.strip():
        rc = getattr(resp.choices[0].message, "reasoning_content", None)
        if rc:
            content = rc
    usage = resp.usage
    return {
        "content": content,
        "tokens_in": usage.prompt_tokens if usage else 0,
        "tokens_out": usage.completion_tokens if usage else 0,
        "latency_s": round(latency, 1),
    }


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

CONFIGS = {
    # Solo baselines
    "ds-flash-solo": {
        "executor_model": "deepseek-chat",
        "executor_key": "DEEPSEEK_API_KEY",
        "executor_url": "https://api.deepseek.com",
        "advisor": None,
        "label": "DeepSeek Flash (solo)",
    },
    "glm4-flash-solo": {
        "executor_model": "glm-4-flash",
        "executor_key": "GLMCODE_API_KEY",
        "executor_url": "https://open.bigmodel.cn/api/paas/v4",
        "advisor": None,
        "label": "GLM-4-Flash (solo)",
    },
    # Advisor pairs
    "ds-flash-ds-pro": {
        "executor_model": "deepseek-chat",
        "executor_key": "DEEPSEEK_API_KEY",
        "executor_url": "https://api.deepseek.com",
        "advisor": {
            "model": "deepseek-reasoner",
            "key": "DEEPSEEK_API_KEY",
            "url": "https://api.deepseek.com",
        },
        "label": "DS Flash + DS Pro (advisor)",
    },
    "glm4-flash-glm51": {
        "executor_model": "glm-4-flash",
        "executor_key": "GLMCODE_API_KEY",
        "executor_url": "https://open.bigmodel.cn/api/paas/v4",
        "advisor": {
            "model": "glm-5.1",
            "key": "GLMCODE_API_KEY",
            "url": "https://open.bigmodel.cn/api/paas/v4",
        },
        "label": "GLM-4-Flash + GLM-5.1 (advisor)",
    },
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXECUTOR_SYSTEM_SOLO = """You are an expert Python programmer. Solve the given problem.

Rules:
1. Output ONLY the function/class implementation — no imports, no test code, no explanation
2. Use correct Python syntax
3. Handle edge cases

After writing code, review it once. If you find bugs, fix them before outputting."""

EXECUTOR_SYSTEM_ADVISOR = """You are an expert Python programmer. Solve the given problem.

You have access to a senior engineer advisor you can consult if you get stuck.
Only ask the advisor if you are genuinely uncertain — easy problems don't need help.

Rules:
1. Output ONLY the function/class implementation — no imports, no test code, no explanation
2. Use correct Python syntax
3. Handle edge cases
4. If you received advisor guidance, incorporate it into your solution

After writing code, review it once. If you find bugs, fix them before outputting."""

EXECUTOR_SELF_CHECK = """You just wrote code for a problem. Review your solution:

PROBLEM: {problem}

YOUR CODE:
```python
{code}
```

Think about:
1. Does it handle all edge cases from the problem?
2. Are there any off-by-one errors?
3. Does it match the expected function signature exactly?

If you find issues, output ONLY the corrected code (no explanation).
If the code looks correct, output ONLY the code unchanged."""

# Executor decides: ask advisor or proceed directly
EXECUTOR_JUDGE_PROMPT = """You are solving a programming problem. Rate your confidence:

PROBLEM: {problem}

Rate your confidence from 1-5:
1 = No idea how to approach
2 = Have a vague approach but unsure about edge cases
3 = Know the approach but some implementation details unclear
4 = Confident but want to verify my approach
5 = Very confident, straightforward problem

Output ONLY a single number (1-5)."""

ADVISOR_PROMPT = """You are a senior staff engineer. The executor is about to write code for this problem:

PROBLEM: {problem}

{attempt_context}

Provide concise guidance in under 60 words:
1. Key edge cases to handle
2. Common pitfalls for this type of problem
3. Suggested approach (algorithm/data structure)

Do NOT write the full solution — just guide the executor."""

# ---------------------------------------------------------------------------
# Code extraction & testing
# ---------------------------------------------------------------------------

def extract_code(text: str) -> str:
    """Extract Python code from model output."""
    # Try code block
    for pattern in [r"```python\n(.*?)```", r"```\n(.*?)```"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    # If no code block, check if it looks like code
    lines = text.strip().split("\n")
    code_lines = [l for l in lines if not l.startswith("#") or "=" in l]
    if any("def " in l or "class " in l for l in code_lines):
        return "\n".join(code_lines).strip()
    return text.strip()


def run_tests(code: str, test_code: str) -> Dict:
    """Run test code against generated code. Returns {passed, failed, error}."""
    full_code = code + "\n\n" + test_code
    try:
        exec(full_code, {})
        return {"passed": True, "tests_passed": "all", "error": None}
    except AssertionError as e:
        return {"passed": False, "tests_passed": "partial", "error": str(e)}
    except Exception as e:
        return {"passed": False, "tests_passed": 0, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Solo mode: executor writes, self-reviews, done
# ---------------------------------------------------------------------------

def run_solo(problem: Dict, config: Dict) -> Dict:
    """Run executor-only baseline. Multiple rounds of self-review."""
    api_key = _load_key(config["executor_key"])
    base_url = config["executor_url"]
    model = config["executor_model"]

    total_tokens_in = 0
    total_tokens_out = 0
    total_latency = 0
    rounds = 0

    # Round 1: initial solution
    messages = [
        {"role": "system", "content": EXECUTOR_SYSTEM_SOLO},
        {"role": "user", "content": problem["prompt"]},
    ]
    r = call_llm(model, messages, api_key, base_url)
    total_tokens_in += r["tokens_in"]
    total_tokens_out += r["tokens_out"]
    total_latency += r["latency_s"]
    rounds += 1

    code = extract_code(r["content"])

    # Round 2: self-review (executor checks own code)
    review_msg = EXECUTOR_SELF_CHECK.format(problem=problem["prompt"], code=code)
    messages.append({"role": "assistant", "content": r["content"]})
    messages.append({"role": "user", "content": review_msg})
    r2 = call_llm(model, messages, api_key, base_url)
    total_tokens_in += r2["tokens_in"]
    total_tokens_out += r2["tokens_out"]
    total_latency += r2["latency_s"]
    rounds += 1

    reviewed_code = extract_code(r2["content"])
    # Use reviewed code if it's valid, else keep original
    if reviewed_code and ("def " in reviewed_code or "class " in reviewed_code):
        code = reviewed_code

    # Test
    test_result = run_tests(code, problem["test_code"])

    return {
        "problem_id": problem["id"],
        "problem_name": problem["name"],
        "config": config["label"],
        "passed": test_result["passed"],
        "error": test_result["error"],
        "code": code,
        "rounds": rounds,
        "advisor_calls": 0,
        "tokens_in": total_tokens_in,
        "tokens_out": total_tokens_out,
        "total_tokens": total_tokens_in + total_tokens_out,
        "latency_s": round(total_latency, 1),
    }


# ---------------------------------------------------------------------------
# Advisor loop mode: executor → check → advisor → executor → ...
# ---------------------------------------------------------------------------

def run_advisor_loop(problem: Dict, config: Dict, max_rounds: int = 4) -> Dict:
    """Run executor + advisor loop. Executor judges confidence first, only calls advisor if stuck."""
    api_key = _load_key(config["executor_key"])
    base_url = config["executor_url"]
    model = config["executor_model"]
    adv = config["advisor"]
    adv_key = _load_key(adv["key"])
    adv_url = adv["url"]
    adv_model = adv["model"]

    total_tokens_in = 0
    total_tokens_out = 0
    advisor_tokens_in = 0
    advisor_tokens_out = 0
    total_latency = 0
    advisor_calls = 0
    rounds = 0
    code = ""
    advisor_advice = ""
    asked_advisor = False

    # Step 1: Executor judges confidence (1 token output)
    judge_msg = [{"role": "user", "content": EXECUTOR_JUDGE_PROMPT.format(problem=problem["prompt"])}]
    jr = call_llm(model, judge_msg, api_key, base_url, temperature=0.0, max_tokens=5)
    total_tokens_in += jr["tokens_in"]
    total_tokens_out += jr["tokens_out"]
    total_latency += jr["latency_s"]

    # Parse confidence score
    confidence = 5  # default: confident
    try:
        confidence = int(re.search(r'[1-5]', jr["content"]).group())
    except (AttributeError, ValueError):
        pass

    # Step 2: Only ask advisor if confidence <= 3 (stuck/uncertain)
    if confidence <= 3:
        adv_messages = [
            {"role": "system", "content": ADVISOR_PROMPT.format(
                problem=problem["prompt"],
                attempt_context="This is the first attempt. No previous code exists.")},
            {"role": "user", "content": f"Problem: {problem['prompt']}\n\nProvide guidance (under 60 words)."},
        ]
        adv_r = call_llm(adv_model, adv_messages, adv_key, adv_url, temperature=0.3, max_tokens=256)
        advisor_tokens_in += adv_r["tokens_in"]
        advisor_tokens_out += adv_r["tokens_out"]
        total_latency += adv_r["latency_s"]
        advisor_calls += 1
        advisor_advice = adv_r["content"]
        asked_advisor = True

    # Step 3: Executor writes solution (with or without advisor)
    if asked_advisor:
        messages = [
            {"role": "system", "content": EXECUTOR_SYSTEM_ADVISOR},
            {"role": "user", "content": f"{problem['prompt']}\n\nSenior Engineer's Guidance:\n{advisor_advice}"},
        ]
    else:
        messages = [
            {"role": "system", "content": EXECUTOR_SYSTEM_SOLO},
            {"role": "user", "content": problem["prompt"]},
        ]

    r = call_llm(model, messages, api_key, base_url)
    total_tokens_in += r["tokens_in"]
    total_tokens_out += r["tokens_out"]
    total_latency += r["latency_s"]
    rounds += 1
    code = extract_code(r["content"])

    # Step 4: Test + retry loop with advisor for failures
    for attempt in range(max_rounds - 1):
        test_result = run_tests(code, problem["test_code"])
        if test_result["passed"]:
            break

        # Failed — always ask advisor on retry (we know we're stuck now)
        error_info = test_result.get("error", "unknown")
        adv_messages = [
            {"role": "system", "content": ADVISOR_PROMPT.format(
                problem=problem["prompt"],
                attempt_context=f"Previous attempt FAILED.\n\nCode:\n```python\n{code}\n```\n\nError: {error_info}")},
            {"role": "user", "content": "The code above has bugs. What's wrong and how to fix? (under 60 words)"},
        ]
        adv_r = call_llm(adv_model, adv_messages, adv_key, adv_url, temperature=0.3, max_tokens=256)
        advisor_tokens_in += adv_r["tokens_in"]
        advisor_tokens_out += adv_r["tokens_out"]
        total_latency += adv_r["latency_s"]
        advisor_calls += 1
        fix_advice = adv_r["content"]

        messages = [
            {"role": "system", "content": EXECUTOR_SYSTEM_ADVISOR},
            {"role": "user", "content": f"{problem['prompt']}\n\nPrevious code had a bug:\n```python\n{code}\n```\nError: {error_info}\n\nSenior Engineer's Fix Guidance:\n{fix_advice}\n\nWrite the corrected code."},
        ]
        r = call_llm(model, messages, api_key, base_url)
        total_tokens_in += r["tokens_in"]
        total_tokens_out += r["tokens_out"]
        total_latency += r["latency_s"]
        rounds += 1
        new_code = extract_code(r["content"])
        if new_code and ("def " in new_code or "class " in new_code):
            code = new_code

    # Final test
    test_result = run_tests(code, problem["test_code"])

    return {
        "problem_id": problem["id"],
        "problem_name": problem["name"],
        "config": config["label"],
        "passed": test_result["passed"],
        "error": test_result["error"],
        "code": code,
        "rounds": rounds,
        "advisor_calls": advisor_calls,
        "confidence": confidence,
        "asked_advisor_initially": asked_advisor,
        "tokens_in": total_tokens_in,
        "tokens_out": total_tokens_out,
        "advisor_tokens_in": advisor_tokens_in,
        "advisor_tokens_out": advisor_tokens_out,
        "total_tokens": total_tokens_in + total_tokens_out + advisor_tokens_in + advisor_tokens_out,
        "latency_s": round(total_latency, 1),
        "advisor_advice_first": advisor_advice[:200] if advisor_advice else "",
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("/root/advisor-eval/loop_results")


def run_config(config_name: str, n: int = None):
    """Run a single config against the eval set."""
    config = CONFIGS.get(config_name)
    if not config:
        print(f"Unknown config: {config_name}")
        return

    eval_set = load_eval_set()
    if n:
        eval_set = eval_set[:n]

    is_advisor = config["advisor"] is not None
    results = []

    print(f"\n{'='*60}")
    print(f"Config: {config['label']} ({'advisor loop' if is_advisor else 'solo'})")
    print(f"Problems: {len(eval_set)}")
    print(f"{'='*60}")

    for problem in eval_set:
        print(f"\n--- {problem['id']}: {problem['name']} ({problem['difficulty']}) ---")

        try:
            if is_advisor:
                result = run_advisor_loop(problem, config)
            else:
                result = run_solo(problem, config)
            results.append(result)

            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"  {status} | rounds={result['rounds']} "
                  f"tokens={result['total_tokens']} "
                  f"advisor_calls={result['advisor_calls']} "
                  f"latency={result['latency_s']}s")
            if not result["passed"]:
                print(f"  error: {result['error'][:100]}")

        except Exception as e:
            print(f"  💥 ERROR: {e}")
            results.append({
                "problem_id": problem["id"],
                "problem_name": problem["name"],
                "config": config["label"],
                "passed": False,
                "error": str(e),
                "rounds": 0, "advisor_calls": 0,
                "tokens_in": 0, "tokens_out": 0,
                "total_tokens": 0, "latency_s": 0,
            })

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{config_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    passed = sum(1 for r in results if r["passed"])
    total_tokens = sum(r["total_tokens"] for r in results)
    avg_rounds = sum(r["rounds"] for r in results) / len(results)
    avg_advisor = sum(r["advisor_calls"] for r in results) / len(results)
    total_latency = sum(r["latency_s"] for r in results)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {config['label']}")
    print(f"  Completion: {passed}/{len(results)} ({100*passed/len(results):.0f}%)")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg rounds: {avg_rounds:.1f}")
    print(f"  Avg advisor calls: {avg_advisor:.1f}")
    print(f"  Total latency: {total_latency:.1f}s")
    print(f"  Saved to: {out_path}")
    print(f"{'='*60}")

    return results


def compare_results():
    """Compare all result files."""
    results_files = sorted(RESULTS_DIR.glob("*.json"))
    if not results_files:
        print("No results found. Run --all first.")
        return

    all_results = {}
    for f in results_files:
        with open(f) as fh:
            all_results[f.stem] = json.load(fh)

    print(f"\n{'='*70}")
    print(f"COMPARISON — {len(all_results)} configs")
    print(f"{'='*70}")

    # Header
    configs = list(all_results.keys())
    print(f"{'Problem':<25}", end="")
    for c in configs:
        print(f" {c:<20}", end="")
    print()

    # Per-problem comparison
    problems = sorted(set(
        r["problem_id"] for results in all_results.values() for r in results
    ))

    for pid in problems:
        print(f"{pid:<25}", end="")
        for config in configs:
            results = all_results[config]
            r = next((x for x in results if x["problem_id"] == pid), None)
            if r:
                status = "✅" if r["passed"] else "❌"
                tokens = r["total_tokens"]
                print(f" {status}{tokens:>5}tok{'':>6}", end="")
            else:
                print(f" {'N/A':>20}", end="")
        print()

    # Summary row
    print(f"{'─'*70}")
    print(f"{'COMPLETION':<25}", end="")
    for config in configs:
        results = all_results[config]
        passed = sum(1 for r in results if r["passed"])
        print(f" {passed}/{len(results)}{'':>14}", end="")
    print()

    print(f"{'TOTAL TOKENS':<25}", end="")
    for config in configs:
        results = all_results[config]
        total = sum(r["total_tokens"] for r in results)
        print(f" {total:>7,}{'':>10}", end="")
    print()

    print(f"{'AVG ROUNDS':<25}", end="")
    for config in configs:
        results = all_results[config]
        avg = sum(r["rounds"] for r in results) / len(results) if results else 0
        print(f" {avg:>7.1f}{'':>10}", end="")
    print()

    print(f"{'AVG ADVISOR CALLS':<25}", end="")
    for config in configs:
        results = all_results[config]
        avg = sum(r["advisor_calls"] for r in results) / len(results) if results else 0
        print(f" {avg:>7.1f}{'':>10}", end="")
    print()

    print(f"{'TOTAL LATENCY (s)':<25}", end="")
    for config in configs:
        results = all_results[config]
        total = sum(r["latency_s"] for r in results)
        print(f" {total:>7.1f}{'':>10}", end="")
    print()

    # Key comparison: advisor vs solo
    print(f"\n{'='*70}")
    print("KEY: Can advisor loop beat solo on FEWER tokens + BETTER completion?")
    print(f"{'='*70}")

    pairs = [
        ("ds-flash-solo", "ds-flash-ds-pro"),
        ("glm4-flash-solo", "glm4-flash-glm51"),
    ]
    for solo_name, adv_name in pairs:
        if solo_name in all_results and adv_name in all_results:
            solo = all_results[solo_name]
            adv = all_results[adv_name]
            solo_passed = sum(1 for r in solo if r["passed"])
            adv_passed = sum(1 for r in adv if r["passed"])
            solo_tokens = sum(r["total_tokens"] for r in solo)
            adv_tokens = sum(r["total_tokens"] for r in adv)

            completion_delta = adv_passed - solo_passed
            token_delta = solo_tokens - adv_tokens  # positive = advisor uses fewer

            comp_str = f"+{completion_delta}" if completion_delta > 0 else str(completion_delta)
            tok_str = f"↓{token_delta:,}" if token_delta > 0 else f"↑{abs(token_delta):,}"

            print(f"\n  {solo_name} vs {adv_name}:")
            print(f"    Completion: {solo_passed} → {adv_passed} ({comp_str})")
            print(f"    Tokens:     {solo_tokens:,} → {adv_tokens:,} ({tok_str})")
            if completion_delta > 0 and token_delta > 0:
                print(f"    ✅ ADVISOR WINS: better + cheaper")
            elif completion_delta > 0:
                print(f"    ⚠️ Advisor better but costs more tokens")
            elif completion_delta == 0 and token_delta > 0:
                print(f"    ⚠️ Same quality but advisor cheaper")
            else:
                print(f"    ❌ Advisor does not help")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Advisor Loop Evaluation")
    parser.add_argument("--config", choices=list(CONFIGS.keys()), help="Run single config")
    parser.add_argument("--all", action="store_true", help="Run all configs")
    parser.add_argument("--compare", action="store_true", help="Compare results")
    parser.add_argument("-n", type=int, default=None, help="Number of problems (default: all 10)")
    args = parser.parse_args()

    if args.config:
        run_config(args.config, args.n)
    elif args.all:
        # Run solo baselines first, then advisor pairs
        for name in CONFIGS:
            run_config(name, args.n)
        compare_results()
    elif args.compare:
        compare_results()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
