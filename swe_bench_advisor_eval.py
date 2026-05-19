#!/usr/bin/env python3
"""
SWE-bench Advisor Evaluation Framework

Generates patch predictions using:
  1. Executor-only (baseline)
  2. Executor + Advisor (experimental)

Then submits to Modal for SWE-bench evaluation.

Usage:
  # Generate predictions for 20-sample pilot
  python3 swe_bench_advisor_eval.py --mode generate --n 20 --run_id pilot

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
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
}

# Model configurations for advisor comparison
MODEL_CONFIGS = {
    # Baseline: executor only (cheap model, no advisor)
    "ds-flash-solo": {
        "executor_model": "deepseek-chat",
        "executor_key_env": "DEEPSEEK_API_KEY",
        "executor_base_url": "https://api.deepseek.com",
        "advisor_model": None,
    },
    # Experimental: cheap executor + strong advisor
    "ds-flash-ds-pro-advisor": {
        "executor_model": "deepseek-chat",
        "executor_key_env": "DEEPSEEK_API_KEY",
        "executor_base_url": "https://api.deepseek.com",
        "advisor_model": "deepseek-reasoner",
        "advisor_key_env": "DEEPSEEK_API_KEY",
        "advisor_base_url": "https://api.deepseek.com",
    },
    "glm4flash-solo": {
        "executor_model": "glm-4-flash",
        "executor_key_env": "GLM_API_KEY",
        "executor_base_url": "https://open.bigmodel.cn/api/paas/v4",
        "advisor_model": None,
    },
    "glm4flash-glm51-advisor": {
        "executor_model": "glm-4-flash",
        "executor_key_env": "GLM_API_KEY",
        "executor_base_url": "https://open.bigmodel.cn/api/paas/v4",
        "advisor_model": "glm-5.1",
        "advisor_key_env": "GLM_API_KEY",
        "advisor_base_url": "https://open.bigmodel.cn/api/paas/v4",
    },
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
# Patch generation (single model call)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_PATCH = """You are an expert software engineer. Given a GitHub issue description
and relevant repository context, generate a minimal patch that resolves the issue.

Rules:
1. Output ONLY a unified diff patch (git diff format)
2. The patch must be complete and apply cleanly
3. Do NOT modify test files unless the issue explicitly requires it
4. Keep changes minimal — fix only what the issue describes
5. If unsure about exact line numbers, make your best guess based on context

Format:
```diff
diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -start,count +start,count @@
 context line
-removed line
+added line
```
"""

SYSTEM_PROMPT_ADVISOR = """You are a senior staff engineer reviewing a bug fix. The executor model
is about to generate a patch for a GitHub issue. Based on the issue description
and repository context, provide concise guidance in under 80 words:

1. Which files likely need changes
2. What the root cause probably is
3. Key edge cases to handle
4. Common pitfalls for this type of fix

Be specific. Do NOT write the full patch — just guide the executor."""

SYSTEM_PROMPT_WITH_ADVICE = """You are an expert software engineer. Given a GitHub issue description,
repository context, and advisor guidance, generate a minimal patch that resolves the issue.

A senior engineer has provided guidance below — follow it closely.

Rules:
1. Output ONLY a unified diff patch (git diff format)
2. The patch must be complete and apply cleanly
3. Do NOT modify test files unless the issue explicitly requires it
4. Keep changes minimal — fix only what the issue describes

Format:
```diff
diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -start,count +start,count @@
 context line
-removed line
+added line
```
"""


def _get_api_key(env_var: str) -> str:
    """Get API key from env, hermes .env, or hermes auth."""
    key = os.environ.get(env_var)
    if key:
        return key
    # Try hermes .env
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.exists():
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{env_var}=") and not line.startswith("#"):
                        key = line.split("=", 1)[1].strip().strip("'\"")
                        if key:
                            return key
        except Exception:
            pass
    # Try hermes auth.json credential pool
    auth_path = Path.home() / ".hermes" / "auth.json"
    if auth_path.exists():
        try:
            auth = json.loads(auth_path.read_text())
            pool = auth.get("credential_pool", {})
            key_map = {
                "DEEPSEEK_API_KEY": ["deepseek", "custom:deepseek"],
                "GLM_API_KEY": ["custom:glmcode"],
            }
            providers = key_map.get(env_var, [])
            for provider in providers:
                creds = pool.get(provider, [])
                if isinstance(creds, list) and creds:
                    k = creds[0].get("api_key", "") if isinstance(creds[0], dict) else ""
                    if k:
                        return k
        except Exception:
            pass
    return ""


def _call_model(model: str, messages: List[Dict], api_key: str,
                base_url: str, temperature: float = 0.2,
                max_tokens: int = 4096, timeout: int = 120) -> Dict:
    """Call an OpenAI-compatible model."""
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    start = time.time()

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    latency = time.time() - start
    content = resp.choices[0].message.content or ""
    # reasoning_content fallback for GLM-5.1 / DS thinking mode
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


def extract_patch(text: str) -> str:
    """Extract unified diff patch from model output."""
    # Try code block first
    patterns = [
        r"```diff\n(.*?)```",
        r"```\n(.*?)```",
        r"(diff --git .+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            patch = m.group(1).strip()
            if patch.startswith("diff --git"):
                return patch
    # Fallback: return entire text if it looks like a diff
    if "diff --git" in text:
        start = text.index("diff --git")
        return text[start:].strip()
    return ""


def generate_patch_baseline(instance: Dict, config: Dict) -> Dict:
    """Generate patch with executor only (no advisor)."""
    problem = instance.get("problem_statement", "")
    repo = instance.get("repo", "")
    hints = instance.get("hints_text", "")

    prompt = f"""Repository: {repo}
Issue: {instance.get('instance_id', '')}

## Problem Statement
{problem}

## Hints from Comments
{hints[:2000] if hints else 'None'}

Generate a patch to fix this issue."""

    api_key = _get_api_key(config["executor_key_env"])
    result = _call_model(
        model=config["executor_model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_PATCH},
            {"role": "user", "content": prompt},
        ],
        api_key=api_key,
        base_url=config["executor_base_url"],
    )

    return {
        "instance_id": instance["instance_id"],
        "model_name_or_path": config["executor_model"],
        "model_patch": extract_patch(result["content"]),
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "latency_s": result["latency_s"],
        "raw_output": result["content"][:500],
    }


def generate_patch_with_advisor(instance: Dict, config: Dict) -> Dict:
    """Generate patch with executor + advisor."""
    problem = instance.get("problem_statement", "")
    repo = instance.get("repo", "")
    hints = instance.get("hints_text", "")

    # Step 1: Ask advisor for guidance
    advisor_prompt = f"""Repository: {repo}
Issue: {instance.get('instance_id', '')}

## Problem Statement
{problem}

## Hints from Comments
{hints[:2000] if hints else 'None'}

The executor is about to generate a patch. What guidance do you have?
(Advisor: please keep your guidance under 80 words — I need a focused starting point, not a comprehensive plan.)"""

    advisor_key = _get_api_key(config["advisor_key_env"])
    advisor_result = _call_model(
        model=config["advisor_model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_ADVISOR},
            {"role": "user", "content": advisor_prompt},
        ],
        api_key=advisor_key,
        base_url=config["advisor_base_url"],
        temperature=0.3,
        max_tokens=512,
    )

    advice = advisor_result["content"]

    # Step 2: Executor generates patch with advisor guidance
    executor_prompt = f"""Repository: {repo}
Issue: {instance.get('instance_id', '')}

## Problem Statement
{problem}

## Hints from Comments
{hints[:2000] if hints else 'None'}

## Senior Engineer's Guidance
{advice}

Generate a patch following this guidance."""

    executor_key = _get_api_key(config["executor_key_env"])
    executor_result = _call_model(
        model=config["executor_model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_WITH_ADVICE},
            {"role": "user", "content": executor_prompt},
        ],
        api_key=executor_key,
        base_url=config["executor_base_url"],
    )

    return {
        "instance_id": instance["instance_id"],
        "model_name_or_path": f"{config['executor_model']}+{config['advisor_model']}",
        "model_patch": extract_patch(executor_result["content"]),
        "tokens_in": executor_result["tokens_in"] + advisor_result["tokens_in"],
        "tokens_out": executor_result["tokens_out"] + advisor_result["tokens_out"],
        "latency_s": round(executor_result["latency_s"] + advisor_result["latency_s"], 1),
        "advisor_advice": advice[:300],
        "advisor_tokens": {
            "in": advisor_result["tokens_in"],
            "out": advisor_result["tokens_out"],
        },
        "executor_tokens": {
            "in": executor_result["tokens_in"],
            "out": executor_result["tokens_out"],
        },
        "raw_output": executor_result["content"][:500],
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_generate(args):
    """Generate predictions for all model configs."""
    dataset = DATASETS.get(args.dataset, args.dataset)
    instances = load_dataset(dataset, args.n)
    print(f"Loaded {len(instances)} instances from {dataset}")

    run_dir = RESULTS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Determine which configs to run
    configs_to_run = args.configs.split(",") if args.configs else list(MODEL_CONFIGS.keys())

    for config_name in configs_to_run:
        config = MODEL_CONFIGS.get(config_name)
        if not config:
            print(f"Unknown config: {config_name}, skipping")
            continue

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
        print(f"\n[{config_name}] {len(done_ids)} done, {len(remaining)} remaining")

        if not remaining:
            print(f"[{config_name}] All instances already predicted")
            continue

        is_advisor = config["advisor_model"] is not None
        gen_fn = generate_patch_with_advisor if is_advisor else generate_patch_baseline
        total_tokens = {"in": 0, "out": 0}
        total_latency = 0
        patches_found = 0

        for idx, instance in enumerate(remaining):
            iid = instance["instance_id"]
            try:
                result = gen_fn(instance, config)

                # Write prediction in SWE-bench format
                pred = {
                    "instance_id": result["instance_id"],
                    "model_name_or_path": result["model_name_or_path"],
                    "model_patch": result["model_patch"],
                }
                with open(predictions_path, "a") as f:
                    f.write(json.dumps(pred, ensure_ascii=False) + "\n")

                # Track metrics
                total_tokens["in"] += result["tokens_in"]
                total_tokens["out"] += result["tokens_out"]
                total_latency += result["latency_s"]
                if result["model_patch"]:
                    patches_found += 1

                status = "✓" if result["model_patch"] else "✗"
                print(f"  [{idx+1}/{len(remaining)}] {status} {iid} "
                      f"({result['tokens_in']}↓{result['tokens_out']}↑, "
                      f"{result['latency_s']}s)")

            except Exception as e:
                print(f"  [{idx+1}/{len(remaining)}] ✗ {iid} ERROR: {e}")
                # Write empty prediction so we don't retry
                with open(predictions_path, "a") as f:
                    f.write(json.dumps({
                        "instance_id": iid,
                        "model_name_or_path": config["executor_model"],
                        "model_patch": "",
                    }) + "\n")

            # Save metrics after each instance
            metrics = {
                "config": config_name,
                "dataset": dataset,
                "total_instances": len(instances),
                "completed": len(done_ids) + idx + 1,
                "patches_generated": patches_found,
                "tokens_in": total_tokens["in"],
                "tokens_out": total_tokens["out"],
                "total_latency_s": round(total_latency, 1),
            }
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

    print("\n=== Generation complete ===")
    print(f"Results in: {run_dir}")


def run_evaluate(args):
    """Submit predictions to SWE-bench evaluation (Modal)."""
    run_dir = RESULTS_DIR / args.run_id
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    # Find all prediction files
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
    """Summarize evaluation results."""
    run_dir = RESULTS_DIR / args.run_id
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    print(f"\n=== Summary for {args.run_id} ===\n")

    # Load metrics
    metrics_files = sorted(run_dir.glob("metrics_*.json"))
    for mf in metrics_files:
        config_name = mf.stem.replace("metrics_", "")
        with open(mf) as f:
            metrics = json.load(f)
        print(f"  [{config_name}]")
        print(f"    Patches: {metrics['patches_generated']}/{metrics['total_instances']}")
        print(f"    Tokens: {metrics['tokens_in']}↓ {metrics['tokens_out']}↑")
        print(f"    Latency: {metrics['total_latency_s']}s total")
        print()

    # Check for evaluation results
    eval_dirs = sorted(Path("evaluation_results").glob(f"{args.run_id}_*"))
    for ed in eval_dirs:
        result_file = ed / "results.json"
        if result_file.exists():
            with open(result_file) as f:
                results = json.load(f)
            config_name = ed.name.replace(f"{args.run_id}_", "")
            resolved = results.get("resolved", 0)
            total = results.get("total", 0)
            print(f"  [{config_name}] Resolved: {resolved}/{total} ({100*resolved/total:.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SWE-bench Advisor Evaluation")
    parser.add_argument("--mode", choices=["generate", "evaluate", "summarize"], required=True)
    parser.add_argument("--run_id", required=True, help="Unique run identifier")
    parser.add_argument("--dataset", default="verified", help="Dataset: verified, lite, full, or HF path")
    parser.add_argument("--n", type=int, default=None, help="Number of instances (default: all)")
    parser.add_argument("--configs", default=None, help="Comma-separated config names (default: all)")
    parser.add_argument("--modal", action="store_true", help="Use Modal for evaluation")
    args = parser.parse_args()

    if args.mode == "generate":
        run_generate(args)
    elif args.mode == "evaluate":
        run_evaluate(args)
    elif args.mode == "summarize":
        run_summarize(args)


if __name__ == "__main__":
    main()
