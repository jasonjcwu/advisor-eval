"""
SWE-bench Multilingual runner — loads tasks, runs agent, collects results.

Usage:
    python swe_runner.py --executor deepseek-chat --advisor glm-5.1 --subset 18
    python swe_runner.py --matrix  # run full eval matrix
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_loop import AdvisorAgent, run_task
from config import EVAL_PAIRS, get_model, MODELS


DATASET_NAME = "SWE-bench/SWE-bench_Multilingual"
RESULTS_DIR = Path(__file__).parent / "results"


def load_swe_tasks(subset_size: int = None, language: str = None):
    """Load SWE-bench Multilingual tasks from HuggingFace."""
    from datasets import load_dataset
    
    print(f"Loading {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, split="test")
    
    tasks = []
    for item in ds:
        if language and item.get("language", "").lower() != language.lower():
            continue
        tasks.append(item)
    
    if subset_size and subset_size < len(tasks):
        # Stratified subset: pick evenly across languages
        from collections import defaultdict
        by_lang = defaultdict(list)
        for t in tasks:
            lang = t.get("language", "unknown")
            by_lang[lang].append(t)
        
        per_lang = max(1, subset_size // len(by_lang))
        selected = []
        for lang, items in by_lang.items():
            selected.extend(items[:per_lang])
        tasks = selected[:subset_size]
    
    print(f"Loaded {len(tasks)} tasks" + 
          (f" (subset of {subset_size})" if subset_size else ""))
    return tasks


def build_task_prompt(task: dict) -> str:
    """Build the executor's task prompt from a SWE-bench item."""
    repo = task.get("repo", "unknown")
    issue = task.get("problem_statement", "")
    hints = task.get("hints_text", "")
    
    prompt = f"""\
## Task: Fix a bug in {repo}

{issue}

"""
    if hints:
        prompt += f"## Hints\n{hints}\n\n"
    
    prompt += """\
## Instructions
1. First, explore the codebase to understand the structure
2. Identify the root cause of the issue
3. Make minimal, targeted edits to fix it
4. Verify your fix by running relevant tests if possible

The codebase is in the current working directory. Start by listing the files.
"""
    return prompt


def setup_task_workspace(task: dict, base_dir: str = "/tmp/swe-workspaces") -> str:
    """Clone the repo at the right commit for a task."""
    repo = task.get("repo", "")
    base_commit = task.get("base_commit", "")
    instance_id = task.get("instance_id", "unknown")
    
    workspace = os.path.join(base_dir, instance_id.replace("/", "__"))
    
    if os.path.exists(workspace):
        return workspace
    
    os.makedirs(workspace, exist_ok=True)
    
    # Clone repo at specific commit
    clone_url = f"https://github.com/{repo}.git"
    print(f"  Cloning {repo}@{base_commit[:8]}...")
    
    try:
        subprocess.run(
            ["git", "clone", "--quiet", clone_url, workspace],
            check=True, timeout=120, capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", "--quiet", base_commit],
            check=True, timeout=30, capture_output=True,
            cwd=workspace,
        )
    except Exception as e:
        print(f"  ⚠️ Clone failed: {e}")
        # Try shallow clone
        try:
            if os.path.exists(workspace):
                subprocess.run(["rm", "-rf", workspace], check=True)
            subprocess.run(
                ["git", "clone", "--depth=1", "--quiet", clone_url, workspace],
                check=True, timeout=60, capture_output=True,
            )
        except Exception as e2:
            print(f"  ❌ Shallow clone also failed: {e2}")
            return workspace
    
    return workspace


def run_single_task(
    task: dict,
    executor: str,
    advisor: str = None,
    max_turns: int = 15,
    setup_workspace: bool = True,
) -> dict:
    """Run a single SWE-bench task with the advisor agent."""
    instance_id = task.get("instance_id", "unknown")
    
    # Setup workspace
    if setup_workspace:
        workdir = setup_task_workspace(task)
    else:
        workdir = "/tmp/advisor-workspace"
        os.makedirs(workdir, exist_ok=True)
    
    prompt = build_task_prompt(task)
    
    # Run agent
    agent = AdvisorAgent(
        executor_model=executor,
        advisor_model=advisor,
        max_turns=max_turns,
        workdir=workdir,
        verbose=True,
    )
    
    metrics = agent.run(prompt, task_id=instance_id)
    
    # Generate patch from git diff
    patch = ""
    if setup_workspace and os.path.exists(workdir):
        try:
            result = subprocess.run(
                ["git", "diff"],
                capture_output=True, text=True,
                cwd=workdir, timeout=30,
            )
            patch = result.stdout
        except Exception:
            pass
    
    # Build result
    executor_cfg = get_model(executor)
    advisor_cfg = get_model(advisor) if advisor else None
    
    result = {
        "instance_id": instance_id,
        "executor": executor,
        "advisor": advisor,
        "repo": task.get("repo", ""),
        "language": task.get("language", ""),
        "metrics": {
            "executor_input_tokens": metrics.executor_input_tokens,
            "executor_output_tokens": metrics.executor_output_tokens,
            "advisor_input_tokens": metrics.advisor_input_tokens,
            "advisor_output_tokens": metrics.advisor_output_tokens,
            "advisor_calls": metrics.advisor_calls,
            "tool_calls": metrics.tool_calls,
            "total_seconds": round(metrics.total_seconds, 1),
            "num_turns": len([t for t in metrics.turns if t.role == "executor"]),
            "cost_usd": round(metrics.cost_usd(executor_cfg, advisor_cfg), 4),
        },
        "patch": patch,
        "error": metrics.error,
    }
    
    return result


def run_eval_matrix(subset_size: int = 18):
    """Run the full evaluation matrix."""
    tasks = load_swe_tasks(subset_size=subset_size)
    
    all_results = []
    
    for executor, advisor in EVAL_PAIRS:
        pair_name = f"{executor}+{advisor or 'solo'}"
        print(f"\n{'='*60}")
        print(f"Running: {pair_name}")
        print(f"{'='*60}")
        
        pair_results = []
        for i, task in enumerate(tasks):
            instance_id = task.get("instance_id", f"task_{i}")
            print(f"\n[{i+1}/{len(tasks)}] {instance_id} ({task.get('language','?')})")
            
            try:
                result = run_single_task(task, executor, advisor, max_turns=15)
                pair_results.append(result)
                print(f"  → turns={result['metrics']['num_turns']}, "
                      f"advisor_calls={result['metrics']['advisor_calls']}, "
                      f"cost=${result['metrics']['cost_usd']}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                pair_results.append({
                    "instance_id": instance_id,
                    "executor": executor,
                    "advisor": advisor,
                    "error": str(e),
                })
        
        # Save pair results
        pair_file = RESULTS_DIR / f"{pair_name.replace('/', '_')}.json"
        with open(pair_file, "w") as f:
            json.dump(pair_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {pair_file}")
        
        all_results.extend(pair_results)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for executor, advisor in EVAL_PAIRS:
        pair_name = f"{executor}+{advisor or 'solo'}"
        pair_data = [r for r in all_results if r["executor"] == executor and r.get("advisor") == advisor]
        if not pair_data:
            continue
        
        total = len(pair_data)
        errors = sum(1 for r in pair_data if r.get("error"))
        avg_turns = sum(r["metrics"]["num_turns"] for r in pair_data if "metrics" in r) / max(total, 1)
        avg_advisor = sum(r["metrics"]["advisor_calls"] for r in pair_data if "metrics" in r) / max(total, 1)
        total_cost = sum(r["metrics"]["cost_usd"] for r in pair_data if "metrics" in r)
        
        print(f"{pair_name:40s} | {total} tasks | avg_turns={avg_turns:.1f} | "
              f"avg_advisor={avg_advisor:.1f} | cost=${total_cost:.2f} | errors={errors}")
    
    # Save full results
    full_file = RESULTS_DIR / "full_results.json"
    with open(full_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results: {full_file}")


# ─── Demo / Quick Test ──────────────────────────────────────────────

def run_demo():
    """Quick demo with a simple coding task (no SWE-bench)."""
    import tempfile
    
    # Create a workspace with a buggy file
    workdir = tempfile.mkdtemp(prefix="advisor-demo-")
    
    buggy_code = '''\
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    # Bug: this is not fibonacci, it's just n-1 + n-2
    return n - 1 + n - 2

def is_prime(n):
    """Check if n is a prime number."""
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

def sort_list(lst):
    """Sort a list in ascending order."""
    # Bug: returns descending
    return sorted(lst, reverse=True)
'''
    
    with open(os.path.join(workdir, "utils.py"), "w") as f:
        f.write(buggy_code)
    
    task = """\
There are bugs in utils.py. Fix all of them:

1. fibonacci(10) should return 55, not 17
2. sort_list([3,1,2]) should return [1,2,3], not [3,2,1]  
3. is_prime seems correct, but it's very slow for large numbers. Optimize it.

After fixing, verify by running: python -c "from utils import *; print(fibonacci(10), sort_list([3,1,2]))"
"""
    
    print("=" * 60)
    print("DEMO: Simple bug-fix task")
    print("=" * 60)
    
    # Run with and without advisor
    configs = [
        ("deepseek-chat", None, "DeepSeek solo"),
        ("deepseek-chat", "glm-5.1", "DeepSeek + GLM advisor"),
    ]
    
    for executor, advisor, label in configs:
        print(f"\n{'─'*40}")
        print(f"Config: {label}")
        print(f"{'─'*40}")
        
        # Reset workspace
        with open(os.path.join(workdir, "utils.py"), "w") as f:
            f.write(buggy_code)
        
        metrics = run_task(task, executor=executor, advisor=advisor, workdir=workdir)
        
        print(f"\n  Turns: {len([t for t in metrics.turns if t.role=='executor'])}")
        print(f"  Advisor calls: {metrics.advisor_calls}")
        print(f"  Tokens: {metrics.executor_input_tokens + metrics.executor_output_tokens} exec + "
              f"{metrics.advisor_input_tokens + metrics.advisor_output_tokens} advisor")
        print(f"  Time: {metrics.total_seconds:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWE-bench Multilingual Advisor Eval")
    parser.add_argument("--demo", action="store_true", help="Run quick demo")
    parser.add_argument("--matrix", action="store_true", help="Run full eval matrix")
    parser.add_argument("--executor", default="deepseek-chat", help="Executor model")
    parser.add_argument("--advisor", default=None, help="Advisor model")
    parser.add_argument("--subset", type=int, default=18, help="Number of tasks")
    parser.add_argument("--max-turns", type=int, default=15, help="Max agent turns")
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(exist_ok=True)
    
    if args.demo:
        run_demo()
    elif args.matrix:
        run_eval_matrix(subset_size=args.subset)
    else:
        tasks = load_swe_tasks(subset_size=args.subset)
        for i, task in enumerate(tasks[:1]):  # Just run first task
            print(f"\n[{i+1}] {task.get('instance_id','?')}")
            result = run_single_task(task, args.executor, args.advisor, args.max_turns)
            print(json.dumps(result["metrics"], indent=2))
