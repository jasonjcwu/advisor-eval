#!/usr/bin/env python3
"""
SWE-bench Advisor Evaluation Framework (Agent Loop Edition)

Runs SWE-bench tasks through a multi-turn agent loop:
  - Baseline: executor-only agent (no advisor tool)
  - Experimental: executor agent + advisor tool (calls stronger model on demand)

The agent has tools: file_read, file_edit, bash_run, ask_advisor.
It iterates up to N turns, calling tools as needed, consulting the advisor
when stuck.

Usage:
  # Generate predictions for 5-sample pilot
  python3 swe_bench_advisor_eval.py --mode generate --n 5 --run_id pilot

  # Generate predictions for full Verified (500)
  python3 swe_bench_advisor_eval.py --mode generate --run_id full

  # Evaluate on Modal (needs Modal auth)
  python3 swe_bench_advisor_eval.py --mode evaluate --run_id pilot

  # Summarize results
  python3 swe_bench_advisor_eval.py --mode summarize --run_id pilot
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from agent_loop import AdvisorAgent, RunMetrics, TOOLS, EXECUTOR_SYSTEM_PROMPT, ADVISOR_SYSTEM_PROMPT
from config import ModelConfig, get_model, MODELS, EVAL_PAIRS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
    "multilingual": "SWE-bench/SWE-bench_Multilingual",
}

# Model pair configurations for advisor comparison.
# Each config specifies (executor_model, advisor_model_or_None).
# These map to models defined in config.py.
AGENT_CONFIGS = {
    # ── Solo baselines (no advisor) ──
    "ds-solo": ("deepseek-chat", None),
    "ds-flash-solo": ("deepseek-v4-flash", None),
    "glm-air-solo": ("glm-4.5-air", None),

    # ── Advisor pairs (cheap executor + strong advisor) ──
    # Mirrors Anthropic's Haiku+Opus pattern
    "glm-air-glm51-advisor": ("glm-4.5-air", "glm-5.1"),
    "ds-flash-glm51-advisor": ("deepseek-v4-flash", "glm-5.1"),
    "glm-air-ds-advisor": ("glm-4.5-air", "deepseek-chat"),
    "ds-flash-ds-advisor": ("deepseek-v4-flash", "deepseek-chat"),
}

RESULTS_DIR = Path("/root/advisor-eval/swe_bench_results")

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_name: str, n: Optional[int] = None) -> List[Dict]:
    """Load SWE-bench dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="test")
        items = [dict(row) for row in ds]
    except Exception as e:
        print(f"ERROR loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    if n:
        items = items[:n]
    return items


# ---------------------------------------------------------------------------
# Workspace setup (git clone for SWE-bench)
# ---------------------------------------------------------------------------

def setup_task_workspace(task: dict, base_dir: str = "/tmp/swe-workspaces") -> str:
    """Clone the repo at the right commit for a SWE-bench task."""
    repo = task.get("repo", "")
    base_commit = task.get("base_commit", "")
    instance_id = task.get("instance_id", "unknown")

    workspace = os.path.join(base_dir, instance_id.replace("/", "__"))

    if os.path.exists(workspace):
        # Reset to clean state
        try:
            subprocess.run(
                ["git", "checkout", "--quiet", base_commit],
                check=True, timeout=30, capture_output=True,
                cwd=workspace,
            )
            subprocess.run(
                ["git", "clean", "-fdq"],
                check=True, timeout=30, capture_output=True,
                cwd=workspace,
            )
        except Exception:
            pass
        return workspace

    os.makedirs(base_dir, exist_ok=True)

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
        print(f"  ⚠️ Clone failed: {e}, trying shallow clone...")
        try:
            if os.path.exists(workspace):
                subprocess.run(["rm", "-rf", workspace], check=True)
            subprocess.run(
                ["git", "clone", "--depth=1", "--quiet", clone_url, workspace],
                check=True, timeout=60, capture_output=True,
            )
        except Exception as e2:
            print(f"  ❌ Shallow clone also failed: {e2}")
            os.makedirs(workspace, exist_ok=True)
            return workspace

    return workspace


# ---------------------------------------------------------------------------
# Task prompt building
# ---------------------------------------------------------------------------

def build_task_prompt(task: dict) -> str:
    """Build the executor's task prompt from a SWE-bench item."""
    repo = task.get("repo", "unknown")
    issue = task.get("problem_statement", "")
    hints = task.get("hints_text", "")

    prompt = f"""## Task: Fix a bug in {repo}

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


# ---------------------------------------------------------------------------
# Patch generation (agent loop)
# ---------------------------------------------------------------------------

def generate_patch_agent(
    instance: Dict,
    executor_model: str,
    advisor_model: Optional[str],
    max_turns: int = 15,
) -> Dict:
    """
    Run the agent loop on a SWE-bench instance.

    Returns a dict with:
      - instance_id, model_name_or_path, model_patch (SWE-bench format)
      - metrics (tokens, turns, advisor_calls, latency, cost, tool_calls)
      - raw_output (first 500 chars of final answer)
    """
    instance_id = instance.get("instance_id", "unknown")

    # Setup workspace (git clone)
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

    # Run agent loop
    t0 = time.time()
    try:
        metrics = agent.run(prompt, task_id=instance_id)
    except Exception as e:
        return {
            "instance_id": instance_id,
            "model_name_or_path": executor_model,
            "model_patch": "",
            "error": str(e),
            "metrics": {},
        }
    total_time = time.time() - t0

    # Extract patch from workspace (git diff)
    patch = ""
    if os.path.exists(workdir):
        try:
            result = subprocess.run(
                ["git", "diff"],
                capture_output=True, text=True,
                cwd=workdir, timeout=30,
            )
            patch = result.stdout
        except Exception:
            pass

    # Compute cost
    executor_cfg = get_model(executor_model)
    advisor_cfg = get_model(advisor_model) if advisor_model else None
    cost = metrics.cost_usd(executor_cfg, advisor_cfg)

    # Build model name for SWE-bench
    if advisor_model:
        model_name = f"{executor_model}+{advisor_model}"
    else:
        model_name = executor_model

    return {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch,
        "metrics": {
            "executor_input_tokens": metrics.executor_input_tokens,
            "executor_output_tokens": metrics.executor_output_tokens,
            "advisor_input_tokens": metrics.advisor_input_tokens,
            "advisor_output_tokens": metrics.advisor_output_tokens,
            "advisor_calls": metrics.advisor_calls,
            "tool_calls": metrics.tool_calls,
            "num_turns": len([t for t in metrics.turns if t.role == "executor"]),
            "total_seconds": round(metrics.total_seconds, 1),
            "cost_usd": round(cost, 4),
            "error": metrics.error,
        },
        "raw_output": metrics.final_answer[:500] if metrics.final_answer else "",
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_generate(args):
    """Generate predictions for all agent configs."""
    dataset = DATASETS.get(args.dataset, args.dataset) or args.dataset
    instances = load_dataset(dataset, args.n)
    print(f"Loaded {len(instances)} instances from {dataset}")

    run_dir = RESULTS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Determine which configs to run
    configs_to_run = args.configs.split(",") if args.configs else list(AGENT_CONFIGS.keys())

    for config_name in configs_to_run:
        if config_name not in AGENT_CONFIGS:
            print(f"Unknown config: {config_name}, skipping")
            continue

        executor_model, advisor_model = AGENT_CONFIGS[config_name]

        predictions_path = run_dir / f"predictions_{config_name}.jsonl"
        metrics_path = run_dir / f"metrics_{config_name}.json"

        # Resume: skip already-predicted instances
        done_ids = set()
        if predictions_path.exists():
            with open(predictions_path) as f:
                for line in f:
                    if line.strip():
                        done_ids.add(json.loads(line)["instance_id"])

        remaining = [i for i in instances if i["instance_id"] not in done_ids]
        print(f"\n[{config_name}] executor={executor_model} advisor={advisor_model or 'none'}")
        print(f"  {len(done_ids)} done, {len(remaining)} remaining")

        if not remaining:
            print(f"  All instances already predicted")
            continue

        total_tokens_in = 0
        total_tokens_out = 0
        total_advisor_tokens = 0
        total_latency = 0
        total_cost = 0
        patches_found = 0
        total_advisor_calls = 0
        total_turns = 0

        for idx, instance in enumerate(remaining):
            iid = instance["instance_id"]
            print(f"\n  [{idx+1}/{len(remaining)}] {iid}")

            try:
                result = generate_patch_agent(
                    instance=instance,
                    executor_model=executor_model,
                    advisor_model=advisor_model,
                    max_turns=args.max_turns,
                )

                # Write prediction in SWE-bench format
                pred = {
                    "instance_id": result["instance_id"],
                    "model_name_or_path": result["model_name_or_path"],
                    "model_patch": result["model_patch"],
                }
                with open(predictions_path, "a") as f:
                    f.write(json.dumps(pred, ensure_ascii=False) + "\n")

                # Track metrics
                m = result.get("metrics", {})
                exec_in = m.get("executor_input_tokens", 0)
                exec_out = m.get("executor_output_tokens", 0)
                adv_in = m.get("advisor_input_tokens", 0)
                adv_out = m.get("advisor_output_tokens", 0)

                total_tokens_in += exec_in
                total_tokens_out += exec_out
                total_advisor_tokens += adv_in + adv_out
                total_latency += m.get("total_seconds", 0)
                total_cost += m.get("cost_usd", 0)
                total_advisor_calls += m.get("advisor_calls", 0)
                total_turns += m.get("num_turns", 0)

                if result["model_patch"]:
                    patches_found += 1

                status = "✓" if result["model_patch"] else "✗"
                turns = m.get("num_turns", "?")
                adv_calls = m.get("advisor_calls", 0)
                print(f"    {status} turns={turns} advisor_calls={adv_calls} "
                      f"tokens={exec_in+exec_in}↓{exec_out+adv_out}↑ "
                      f"${m.get('cost_usd', 0):.4f} {m.get('total_seconds', 0)}s")

                # Save per-instance detailed result
                detail_path = run_dir / f"details_{config_name}" / f"{iid}.json"
                detail_path.parent.mkdir(parents=True, exist_ok=True)
                with open(detail_path, "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

            except Exception as e:
                print(f"    ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                # Write empty prediction so we don't retry
                with open(predictions_path, "a") as f:
                    f.write(json.dumps({
                        "instance_id": iid,
                        "model_name_or_path": executor_model,
                        "model_patch": "",
                    }) + "\n")

            # Save aggregate metrics after each instance
            agg_metrics = {
                "config": config_name,
                "executor_model": executor_model,
                "advisor_model": advisor_model,
                "dataset": dataset,
                "total_instances": len(instances),
                "completed": len(done_ids) + idx + 1,
                "patches_generated": patches_found,
                "executor_tokens_in": total_tokens_in,
                "executor_tokens_out": total_tokens_out,
                "advisor_tokens_total": total_advisor_tokens,
                "total_latency_s": round(total_latency, 1),
                "total_cost_usd": round(total_cost, 4),
                "total_advisor_calls": total_advisor_calls,
                "total_turns": total_turns,
                "avg_turns": round(total_turns / max(idx + 1, 1), 1),
                "avg_advisor_calls": round(total_advisor_calls / max(idx + 1, 1), 1),
            }
            with open(metrics_path, "w") as f:
                json.dump(agg_metrics, f, indent=2)

    print("\n=== Generation complete ===")
    print(f"Results in: {run_dir}")


def run_evaluate(args):
    """Submit predictions to SWE-bench evaluation (Modal or local)."""
    run_dir = RESULTS_DIR / args.run_id
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    pred_files = sorted(run_dir.glob("predictions_*.jsonl"))
    if not pred_files:
        print("No prediction files found")
        sys.exit(1)

    for pred_file in pred_files:
        config_name = pred_file.stem.replace("predictions_", "")
        print(f"\nEvaluating {config_name}...")
        print(f"  Predictions: {pred_file}")

        cmd = (
            f"python -m swebench.harness.run_evaluation "
            f"--dataset_name {DATASETS.get(args.dataset, args.dataset)} "
            f"--predictions_path {pred_file} "
            f"--max_workers 4 "
            f"--run_id {args.run_id}_{config_name}"
        )

        if args.modal:
            cmd += " --modal true --parallelism 10"

        print(f"  Command: {cmd}")
        os.system(cmd)


def run_summarize(args):
    """Summarize evaluation results with loop-specific metrics."""
    run_dir = RESULTS_DIR / args.run_id
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Summary for {args.run_id}")
    print(f"{'='*70}\n")

    # Load metrics
    metrics_files = sorted(run_dir.glob("metrics_*.json"))
    rows = []
    for mf in metrics_files:
        config_name = mf.stem.replace("metrics_", "")
        with open(mf) as f:
            metrics = json.load(f)

        # Support both old (single-shot) and new (agent loop) metric formats
        exec_tokens_in = metrics.get("executor_tokens_in", metrics.get("tokens_in", 0))
        exec_tokens_out = metrics.get("executor_tokens_out", metrics.get("tokens_out", 0))
        advisor_tokens = metrics.get("advisor_tokens_total", 0)
        total_cost = metrics.get("total_cost_usd", 0)
        avg_turns = metrics.get("avg_turns", "N/A")
        avg_advisor = metrics.get("avg_advisor_calls", "N/A")

        row = {
            "config": config_name,
            "patches": f"{metrics['patches_generated']}/{metrics['total_instances']}",
            "exec_tokens": f"{exec_tokens_in}↓ {exec_tokens_out}↑",
            "advisor_tokens": str(advisor_tokens),
            "latency": f"{metrics['total_latency_s']}s",
            "cost": f"${total_cost:.4f}",
            "avg_turns": str(avg_turns),
            "avg_advisor": str(avg_advisor),
        }
        rows.append(row)

        print(f"  [{config_name}]")
        print(f"    Patches: {row['patches']}")
        print(f"    Executor tokens: {row['exec_tokens']}")
        print(f"    Advisor tokens:  {row['advisor_tokens']}")
        print(f"    Avg turns:       {row['avg_turns']}")
        print(f"    Avg advisor calls: {row['avg_advisor']}")
        print(f"    Latency:         {row['latency']}")
        print(f"    Cost:            {row['cost']}")
        print()

    # Comparison table: solo vs advisor for same executor
    print(f"\n{'='*70}")
    print("Solo vs Advisor Comparison")
    print(f"{'='*70}\n")

    configs_data = {}
    for mf in metrics_files:
        config_name = mf.stem.replace("metrics_", "")
        with open(mf) as f:
            configs_data[config_name] = json.load(f)

    # Group by executor model (infer from metrics if not in AGENT_CONFIGS)
    by_executor = {}
    for cname, data in configs_data.items():
        # Try new AGENT_CONFIGS first
        if cname in AGENT_CONFIGS:
            executor, advisor = AGENT_CONFIGS[cname]
        else:
            # Infer from old config data
            executor = data.get("executor_model", data.get("config", cname))
            advisor = data.get("advisor_model", None)
        by_executor.setdefault(executor, []).append((cname, advisor, data))

    for executor, entries in by_executor.items():
        print(f"  Executor: {executor}")
        for cname, advisor, m in entries:
            label = f"+ {advisor}" if advisor else "solo"
            patches = f"{m['patches_generated']}/{m['total_instances']}"
            cost = m.get('total_cost_usd', 0)
            turns = m.get('avg_turns', 'N/A')
            adv = m.get('avg_advisor_calls', 'N/A')
            print(f"    {label:30s} | patches={patches} | "
                  f"turns={turns} | advisor_calls={adv} | cost=${cost:.4f}")
        print()

    # Check for evaluation results (resolved rate from SWE-bench harness)
    eval_dirs = sorted(Path("evaluation_results").glob(f"{args.run_id}_*"))
    if eval_dirs:
        print(f"\n{'='*70}")
        print("SWE-bench Evaluation Results")
        print(f"{'='*70}\n")
        for ed in eval_dirs:
            result_file = ed / "results.json"
            if result_file.exists():
                with open(result_file) as f:
                    results = json.load(f)
                config_name = ed.name.replace(f"{args.run_id}_", "")
                resolved = results.get("resolved", 0)
                total = results.get("total", 0)
                pct = 100 * resolved / total if total else 0
                print(f"  [{config_name}] Resolved: {resolved}/{total} ({pct:.1f}%)")

    # Save comparison JSON
    comparison_path = run_dir / "comparison.json"
    with open(comparison_path, "w") as f:
        json.dump({"configs": configs_data, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"\nComparison saved: {comparison_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench Advisor Evaluation (Agent Loop Edition)"
    )
    parser.add_argument(
        "--mode", choices=["generate", "evaluate", "summarize"],
        required=True,
    )
    parser.add_argument("--run_id", required=True, help="Unique run identifier")
    parser.add_argument(
        "--dataset", default="verified",
        help="Dataset: verified, lite, full, multilingual, or HF path",
    )
    parser.add_argument("--n", type=int, default=None, help="Number of instances")
    parser.add_argument(
        "--configs", default=None,
        help="Comma-separated config names (default: all)",
    )
    parser.add_argument(
        "--max_turns", type=int, default=15,
        help="Max agent loop turns per instance (default: 15)",
    )
    parser.add_argument(
        "--modal", action="store_true", help="Use Modal for evaluation",
    )
    args = parser.parse_args()

    if args.mode == "generate":
        run_generate(args)
    elif args.mode == "evaluate":
        run_evaluate(args)
    elif args.mode == "summarize":
        run_summarize(args)


if __name__ == "__main__":
    main()
