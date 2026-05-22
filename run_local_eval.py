#!/usr/bin/env python3
"""
Run SWE-bench agent loop on local eval_set_swe.json (20 curated instances).
Does NOT need HuggingFace datasets — loads from local JSON.
"""
import json
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from agent_loop import AdvisorAgent
from config import get_model

EVAL_SET_PATH = PROJECT_DIR / "eval_set_swe.json"
RESULTS_DIR = PROJECT_DIR / "swe_bench_results"

# Which configs to run (from swe_bench_advisor_eval.py AGENT_CONFIGS)
CONFIGS = {
    "ds-flash-solo": ("deepseek-v4-flash", None),
    "ds-flash-ds-advisor": ("deepseek-v4-flash", "deepseek-chat"),
    "glm-air-solo": ("glm-4.5-air", None),
    "glm-air-glm51-advisor": ("glm-4.5-air", "glm-5.1"),
}

def load_local_set(path: Path = EVAL_SET_PATH) -> list:
    with open(path) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} instances from {path}")
    return items

def setup_task_workspace(instance: dict, base_dir: str = "/tmp/swe-workspaces") -> str:
    """Clone repo at base_commit for a SWE-bench task."""
    import subprocess
    repo = instance.get("repo", "")
    base_commit = instance.get("base_commit", "")
    instance_id = instance.get("instance_id", "unknown")
    workspace = os.path.join(base_dir, instance_id.replace("/", "__"))

    if os.path.exists(workspace):
        try:
            subprocess.run(["git", "checkout", "--quiet", base_commit],
                           check=True, timeout=30, capture_output=True, cwd=workspace)
            subprocess.run(["git", "clean", "-fdq"],
                           check=True, timeout=30, capture_output=True, cwd=workspace)
        except Exception:
            pass
        return workspace

    os.makedirs(base_dir, exist_ok=True)
    clone_url = f"https://github.com/{repo}.git"
    print(f"  Cloning {repo}@{base_commit[:8]}...")
    try:
        subprocess.run(["git", "clone", "--quiet", clone_url, workspace],
                       check=True, timeout=120, capture_output=True)
        subprocess.run(["git", "checkout", "--quiet", base_commit],
                       check=True, timeout=30, capture_output=True, cwd=workspace)
    except Exception as e:
        print(f"  ⚠️ Clone failed: {e}")
        if os.path.exists(workspace):
            subprocess.run(["rm", "-rf", workspace], check=True)
        try:
            subprocess.run(["git", "clone", "--depth=1", "--quiet", clone_url, workspace],
                           check=True, timeout=60, capture_output=True)
        except Exception as e2:
            print(f"  ❌ Shallow clone also failed: {e2}")
            os.makedirs(workspace, exist_ok=True)
    return workspace

def build_task_prompt(instance: dict) -> str:
    repo = instance.get("repo", "unknown")
    issue = instance.get("problem_statement", "")
    hints = instance.get("hints_text", "")
    prompt = f"## Task: Fix a bug in {repo}\n\n{issue}\n\n"
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

def get_diff(workdir: str) -> str:
    import subprocess
    if not os.path.exists(workdir):
        return ""
    try:
        result = subprocess.run(["git", "diff"], capture_output=True, text=True,
                                cwd=workdir, timeout=30)
        # Also check for untracked files
        result2 = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True,
                                 cwd=workdir, timeout=30)
        return result.stdout + result2.stdout
    except Exception:
        return ""

def run_one(instance: dict, config_name: str, executor_model: str,
            advisor_model=None, max_turns: int = 15) -> dict:
    iid = instance["instance_id"]
    print(f"\n  [{iid}] config={config_name}")

    # Setup workspace
    workdir = setup_task_workspace(instance)

    # Build prompt
    prompt = build_task_prompt(instance)

    # Create agent
    agent = AdvisorAgent(
        executor_model=executor_model,
        advisor_model=advisor_model,
        max_turns=max_turns,
        workdir=workdir,
        verbose=True,
    )

    # Solo mode: remove ask_advisor tool so the agent doesn't waste turns calling it
    if not advisor_model:
        agent.include_advisor_tool = False

    # Run
    t0 = time.time()
    try:
        metrics = agent.run(prompt, task_id=iid)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"instance_id": iid, "model_patch": "", "error": str(e), "metrics": {}}
    total_time = time.time() - t0

    # Get patch
    patch = get_diff(workdir)

    # Compute cost
    executor_cfg = get_model(executor_model)
    advisor_cfg = get_model(advisor_model) if advisor_model else None
    cost = metrics.cost_usd(executor_cfg, advisor_cfg)

    m = {
        "executor_input_tokens": metrics.executor_input_tokens,
        "executor_output_tokens": metrics.executor_output_tokens,
        "advisor_input_tokens": metrics.advisor_input_tokens,
        "advisor_output_tokens": metrics.advisor_output_tokens,
        "advisor_calls": metrics.advisor_calls,
        "tool_calls": dict(metrics.tool_calls),
        "num_turns": len([t for t in metrics.turns if t.role == "executor"]),
        "total_seconds": round(total_time, 1),
        "cost_usd": round(cost, 4),
        "error": metrics.error,
    }

    status = "✓" if patch else "✗"
    print(f"    {status} turns={m['num_turns']} calls={m['advisor_calls']} "
          f"tok={m['executor_input_tokens']+m['executor_output_tokens']}↓{m['advisor_input_tokens']+m['advisor_output_tokens']}↑ "
          f"${m['cost_usd']:.4f} {m['total_seconds']}s")

    return {
        "instance_id": iid,
        "model_name_or_path": f"{executor_model}+{advisor_model}" if advisor_model else executor_model,
        "model_patch": patch,
        "metrics": m,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="ds-flash-solo,ds-flash-ds-advisor,glm-air-solo,glm-air-glm51-advisor",
                        help="Comma-separated config names")
    parser.add_argument("--max_turns", type=int, default=15)
    parser.add_argument("--run_id", default="loop-20")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N instances")
    parser.add_argument("--eval-set", default=None,
                        help="Path to eval set JSON (default: eval_set_swe.json)")
    args = parser.parse_args()

    eval_path = args.eval_set if args.eval_set else EVAL_SET_PATH
    instances = load_local_set(Path(eval_path))
    if args.limit:
        instances = instances[:args.limit]

    run_dir = RESULTS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    configs_to_run = [c.strip() for c in args.configs.split(",")]

    for config_name in configs_to_run:
        if config_name not in CONFIGS:
            print(f"Unknown config: {config_name}. Known: {list(CONFIGS.keys())}")
            continue

        executor_model, advisor_model = CONFIGS[config_name]
        pred_path = run_dir / f"predictions_{config_name}.jsonl"
        details_dir = run_dir / f"details_{config_name}"
        details_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / f"metrics_{config_name}.json"

        # Resume: skip done
        done_ids = set()
        if pred_path.exists():
            with open(pred_path) as f:
                for line in f:
                    if line.strip():
                        done_ids.add(json.loads(line)["instance_id"])

        remaining = [i for i in instances if i["instance_id"] not in done_ids]
        print(f"\n{'='*60}")
        print(f"[{config_name}] executor={executor_model} advisor={advisor_model or 'none'}")
        print(f"  {len(done_ids)} done, {len(remaining)} remaining")
        print(f"{'='*60}")

        if not remaining:
            print("  All done")
            continue

        totals = {
            "executor_in": 0, "executor_out": 0, "advisor_in": 0, "advisor_out": 0,
            "latency": 0, "cost": 0, "calls": 0, "turns": 0, "patches": 0,
        }

        for idx, inst in enumerate(remaining):
            result = run_one(inst, config_name, executor_model, advisor_model,
                             max_turns=args.max_turns)

            # Save prediction
            pred = {
                "instance_id": result["instance_id"],
                "model_name_or_path": result["model_name_or_path"],
                "model_patch": result["model_patch"],
            }
            with open(pred_path, "a") as f:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

            # Save detail
            detail_path = details_dir / f"{result['instance_id']}.json"
            with open(detail_path, "w") as f:
                json.dump(result, f, indent=2)

            # Accumulate
            m = result["metrics"]
            totals["executor_in"] += m.get("executor_input_tokens", 0)
            totals["executor_out"] += m.get("executor_output_tokens", 0)
            totals["advisor_in"] += m.get("advisor_input_tokens", 0)
            totals["advisor_out"] += m.get("advisor_output_tokens", 0)
            totals["latency"] += m.get("total_seconds", 0)
            totals["cost"] += m.get("cost_usd", 0)
            totals["calls"] += m.get("advisor_calls", 0)
            totals["turns"] += m.get("num_turns", 0)
            if result["model_patch"]:
                totals["patches"] += 1

            # Save running metrics
            done = len(done_ids) + idx + 1
            agg = {
                "config": config_name,
                "executor_model": executor_model,
                "advisor_model": advisor_model,
                "total_instances": len(instances),
                "completed": done,
                "patches_generated": totals["patches"],
                "executor_tokens_in": totals["executor_in"],
                "executor_tokens_out": totals["executor_out"],
                "advisor_tokens_total": totals["advisor_in"] + totals["advisor_out"],
                "total_latency_s": round(totals["latency"], 1),
                "total_cost_usd": round(totals["cost"], 4),
                "total_advisor_calls": totals["calls"],
                "total_turns": totals["turns"],
                "avg_turns": round(totals["turns"] / done, 1),
                "avg_advisor_calls": round(totals["calls"] / done, 1),
            }
            with open(metrics_path, "w") as f:
                json.dump(agg, f, indent=2)

        print(f"\n[{config_name}] Done: {totals['patches']}/{len(instances)} patches, "
              f"${totals['cost']:.4f}, {totals['latency']:.0f}s")
        with open(metrics_path) as f:
            final_metrics = json.load(f)
            print(json.dumps(final_metrics, indent=2))

    # Summary across configs
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for config_name in configs_to_run:
        if config_name not in CONFIGS:
            continue
        mp = run_dir / f"metrics_{config_name}.json"
        if mp.exists():
            with open(mp) as f:
                m = json.load(f)
            label = CONFIGS[config_name][1] or "solo"
            print(f"  [{config_name:25s}] patches={m['patches_generated']}/{m['total_instances']} "
                  f"turns={m['avg_turns']} calls={m['avg_advisor_calls']} "
                  f"cost=${m['total_cost_usd']:.4f} latency={m['total_latency_s']:.0f}s")
    print(f"\nResults in: {run_dir}")

if __name__ == "__main__":
    main()
