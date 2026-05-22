#!/usr/bin/env python3
"""SWE-bench evaluation using Claude Code CLI as the agent.

Runs Claude Code in print mode (-p) against SWE-bench instances.
Compares solo (haiku only) vs advisor (haiku + @advisor opus) modes.

Usage:
    python3 run_claude_code_eval.py --solo --limit 6
    python3 run_claude_code_eval.py --advisor --limit 6
    python3 run_claude_code_eval.py --both --limit 6
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
PACKYAPI_KEY = "sk-Y8H1iV4oTJ0sppszOAjDopIb4ht7LhSCdBHMHTxZZACrYrzY"
PACKYAPI_BASE = "https://www.packyapi.com"
EXECUTOR_MODEL = "claude-haiku-4-5-20251001"
ADVISOR_MODEL = "claude-opus-4-7"

BASE_DIR = Path("/root/advisor-eval")
EVAL_DIR = BASE_DIR / "claude-code-swe"
RESULTS_DIR = BASE_DIR / "swe_bench_results" / "claude-code"
WORKSPACES_DIR = Path("/tmp/swe-cc")

MAX_TURNS = 15
TIMEOUT_PER_INSTANCE = 600  # 10 min per instance


def setup_workspace(inst: dict) -> str:
    """Clone or reset the repo at the correct base commit."""
    repo = inst.get("repo", "")
    base_commit = inst.get("base_commit", "")
    instance_id = inst["instance_id"]
    ws = str(WORKSPACES_DIR / instance_id.replace("/", "__"))

    if os.path.exists(ws):
        # Reset to clean state at base commit
        try:
            subprocess.run(
                ["git", "checkout", "--quiet", base_commit],
                check=True, timeout=30, capture_output=True, cwd=ws,
            )
            subprocess.run(
                ["git", "clean", "-fdq"],
                check=True, timeout=30, capture_output=True, cwd=ws,
            )
            subprocess.run(
                ["git", "reset", "--hard", base_commit],
                check=True, timeout=30, capture_output=True, cwd=ws,
            )
        except Exception:
            pass
        return ws

    os.makedirs(str(WORKSPACES_DIR), exist_ok=True)
    print(f"    Clone {repo}...")
    try:
        subprocess.run(
            ["git", "clone", "--quiet", f"https://github.com/{repo}.git", ws],
            check=True, timeout=180, capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", "--quiet", base_commit],
            check=True, timeout=30, capture_output=True, cwd=ws,
        )
    except Exception as e:
        print(f"    Clone failed: {e}")
        os.makedirs(ws, exist_ok=True)
    return ws


def build_prompt(inst: dict, mode: str) -> str:
    """Build the prompt for Claude Code."""
    repo = inst.get("repo", "")
    issue = inst.get("problem_statement", "")
    hints = inst.get("hints_text", "")
    fail_tests_raw = inst.get("FAIL_TO_PASS", "[]")
    pass_tests_raw = inst.get("PASS_TO_PASS", "[]")

    # Parse JSON strings if needed
    fail_tests = json.loads(fail_tests_raw) if isinstance(fail_tests_raw, str) else fail_tests_raw
    pass_tests = json.loads(pass_tests_raw) if isinstance(pass_tests_raw, str) else pass_tests_raw

    prompt = f"""## Bug Fix Task: {inst['instance_id']}

Repository: {repo}

### Problem
{issue}
"""
    if hints:
        prompt += f"\n### Hints\n{hints}\n"

    if fail_tests:
        prompt += f"\n### Tests that should PASS after your fix\n"
        for t in fail_tests[:10]:
            prompt += f"- {t}\n"

    prompt += """
### Instructions
1. First, explore the codebase to understand the structure
2. Identify the root cause of the bug
3. Make minimal, targeted edits to fix it
4. Verify the fix (run the failing tests if possible)

The codebase is in the current directory. Start by reading relevant files.
"""

    if mode == "advisor":
        prompt += """
**IMPORTANT: Before making ANY code edits, call @advisor to discuss your analysis and proposed approach with a senior engineer.** Then proceed with the fix.
"""

    return prompt


def get_diff(ws: str) -> str:
    """Get the git diff from the workspace."""
    try:
        if not os.path.exists(os.path.join(ws, ".git")):
            return ""
        r = subprocess.run(
            ["git", "diff"],
            capture_output=True, text=True, cwd=ws, timeout=30,
        )
        return r.stdout
    except Exception:
        return ""


def run_claude_code(prompt: str, workdir: str, mode: str, max_turns: int = MAX_TURNS) -> dict:
    """Run Claude Code CLI in print mode."""
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = PACKYAPI_KEY
    env["ANTHROPIC_BASE_URL"] = PACKYAPI_BASE

    # Build claude command
    # Note: no --dangerously-skip-permissions (root can't use it)
    # Permissions are configured via ~/.claude/settings.json
    cmd = [
        "claude",
        "-p", prompt,
        "--model", EXECUTOR_MODEL,
        "--max-turns", str(max_turns),
        "--output-format", "json",
        "--no-session-persistence",
    ]

    # Add CLAUDE.md as system prompt context
    claude_md = EVAL_DIR / "CLAUDE.md"
    if claude_md.exists():
        cmd.extend(["--append-system-prompt-file", str(claude_md)])

    if mode == "advisor":
        # Add the agents config
        cmd.extend([
            "--agents", json.dumps({
                "advisor": {
                    "description": "Strategic coding advisor using Claude Opus",
                    "prompt": "You are a senior staff engineer acting as a strategic advisor. Analyze the problem, identify root cause, propose a minimal fix approach, and warn about pitfalls. Be concise and actionable.",
                    "model": ADVISOR_MODEL,
                }
            }),
        ])

    print(f"    Running: claude -p ... --model {EXECUTOR_MODEL} ({mode} mode)")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            cwd=workdir, env=env,
            timeout=TIMEOUT_PER_INSTANCE,
        )
        elapsed = time.time() - t0

        # Parse JSON output
        output = {}
        try:
            parsed = json.loads(result.stdout)
            output["success"] = parsed.get("subtype") == "success"
            output["result_text"] = parsed.get("result", "")
            output["session_id"] = parsed.get("session_id", "")
            output["num_turns"] = parsed.get("num_turns", 0)
            output["cost_usd"] = parsed.get("total_cost_usd", 0)
            output["duration_ms"] = parsed.get("duration_ms", 0)
            output["stop_reason"] = parsed.get("stop_reason", "")

            # Extract model usage
            model_usage = parsed.get("modelUsage", {})
            output["model_usage"] = model_usage
        except json.JSONDecodeError:
            output["success"] = False
            output["result_text"] = result.stdout[:2000]
            output["stderr"] = result.stderr[:1000]

        output["elapsed"] = elapsed
        output["exit_code"] = result.returncode
        return output

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return {"success": False, "error": "timeout", "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        return {"success": False, "error": str(e), "elapsed": elapsed}


def run_instance(inst: dict, mode: str, max_turns: int = MAX_TURNS) -> dict:
    """Run a single SWE-bench instance."""
    instance_id = inst["instance_id"]

    # Setup workspace
    ws = setup_workspace(inst)

    # Build prompt
    prompt = build_prompt(inst, mode)

    # Run Claude Code
    info = run_claude_code(prompt, ws, mode, max_turns)

    # Collect diff
    patch = get_diff(ws)

    result = {
        "instance_id": instance_id,
        "model_name_or_path": f"{EXECUTOR_MODEL}+{ADVISOR_MODEL}" if mode == "advisor" else EXECUTOR_MODEL,
        "model_patch": patch,
        "mode": mode,
        "success": info.get("success", False),
        "num_turns": info.get("num_turns", 0),
        "cost_usd": info.get("cost_usd", 0),
        "elapsed": round(info.get("elapsed", 0), 1),
        "stop_reason": info.get("stop_reason", ""),
    }

    if "model_usage" in info:
        result["model_usage"] = info["model_usage"]
    if "error" in info:
        result["error"] = info["error"]

    return result


def save_results(results: list, mode: str):
    """Save results in SWE-bench format."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # predictions.jsonl for SWE-bench evaluation
    pred_file = RESULTS_DIR / f"predictions_{mode}.jsonl"
    with open(pred_file, "w") as f:
        for r in results:
            f.write(json.dumps({
                "instance_id": r["instance_id"],
                "model_name_or_path": r["model_name_or_path"],
                "model_patch": r["model_patch"],
            }, ensure_ascii=False) + "\n")

    # Full results
    full_file = RESULTS_DIR / f"results_{mode}.json"
    with open(full_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary metrics
    patches = sum(1 for r in results if r["model_patch"])
    total_cost = sum(r.get("cost_usd", 0) for r in results)
    total_time = sum(r.get("elapsed", 0) for r in results)
    avg_turns = sum(r.get("num_turns", 0) for r in results) / max(len(results), 1)

    metrics = {
        "mode": mode,
        "executor": EXECUTOR_MODEL,
        "advisor": ADVISOR_MODEL if mode == "advisor" else None,
        "total": len(results),
        "patches": patches,
        "patch_rate": f"{patches}/{len(results)}",
        "total_cost_usd": round(total_cost, 4),
        "total_time_sec": round(total_time, 1),
        "avg_turns": round(avg_turns, 1),
    }
    with open(RESULTS_DIR / f"metrics_{mode}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Claude Code SWE-bench evaluation")
    parser.add_argument("--eval-set", default=str(BASE_DIR / "eval_set_swe_hard6.json"))
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--mode", choices=["solo", "advisor", "both"], default="both")
    args = parser.parse_args()

    with open(args.eval_set) as f:
        instances = json.load(f)
    if args.limit:
        instances = instances[:args.limit]

    # Update max turns
    max_turns = args.max_turns

    modes = ["solo", "advisor"] if args.mode == "both" else [args.mode]

    all_metrics = {}
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Claude Code SWE-bench: {EXECUTOR_MODEL} {'+' + ADVISOR_MODEL if mode == 'advisor' else 'SOLO'}")
        print(f"  Instances: {len(instances)} | Max turns: {MAX_TURNS}")
        print(f"{'='*60}")

        results = []
        for idx, inst in enumerate(instances):
            iid = inst["instance_id"]
            print(f"\n[{idx+1}/{len(instances)}] {iid}")
            t0 = time.time()
            r = run_instance(inst, mode, max_turns)
            elapsed = time.time() - t0
            results.append(r)
            status = "✓ PATCH" if r["model_patch"] else "✗ no patch"
            print(f"    {status} | turns={r['num_turns']} | {r['elapsed']:.0f}s | ${r.get('cost_usd', 0):.4f}")

        metrics = save_results(results, mode)
        all_metrics[mode] = metrics
        print(f"\n  {mode.upper()} Summary: {metrics['patch_rate']} patches | "
              f"${metrics['total_cost_usd']:.4f} | {metrics['total_time_sec']:.0f}s")

    # Print comparison
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("  COMPARISON")
        print(f"{'='*60}")
        for mode, m in all_metrics.items():
            label = f"{EXECUTOR_MODEL} + {ADVISOR_MODEL}" if mode == "advisor" else f"{EXECUTOR_MODEL} solo"
            print(f"  {label}: {m['patch_rate']} | ${m['total_cost_usd']:.4f} | {m['total_time_sec']:.0f}s | avg {m['avg_turns']} turns")


if __name__ == "__main__":
    main()
