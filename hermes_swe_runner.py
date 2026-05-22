#!/usr/bin/env python3
"""
Hermes Agent SWE-bench Runner
=============================
Runs SWE-bench tasks through the actual Hermes Agent CLI (`hermes chat -q`),
capturing real token usage, timing, and cost data.

Tracks: solo (no advisor) vs advisor mode, comparing:
  1. Token usage (input/output)
  2. Answer quality (patch correctness)
  3. Speed (wall-clock time)

Usage:
  python3 hermes_swe_runner.py --mode solo --model glm-5.1 --limit 3
  python3 hermes_swe_runner.py --mode advisor --model glm-5.1 --advisor-model glm-5.1 --limit 3
  python3 hermes_swe_runner.py --judge  # Score patches with AI judge
"""

import argparse, json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime

EVAL_SET = "/root/advisor-eval/eval_set_swe_hard6.json"
RESULTS_BASE = "/root/advisor-eval/swe_bench_results"

# SWE-bench task prompt template — simulates what Claude Code sees
TASK_PROMPT_TEMPLATE = """Solve this GitHub issue. Your task is to produce a git diff patch that fixes the bug.

**Repository:** {repo}
**Base commit:** {base_commit}
**Issue:** {instance_id}

{problem_statement}

Instructions:
1. Explore the codebase to understand the structure
2. Identify the root cause of the bug
3. Make minimal, targeted edits
4. Verify your fix doesn't break existing tests
5. Output the final patch as a git diff at the end

Format your final patch between ```diff and ``` markers."""


def load_eval_set(limit=None):
    with open(EVAL_SET) as f:
        instances = json.load(f)
    return instances[:limit] if limit else instances


def extract_patch(text):
    """Extract git diff patch from agent response."""
    if not text:
        return ""
    # Try ```diff ... ``` blocks
    import re
    patterns = [
        r'```diff\n(.*?)```',
        r'```\n(.*?)```',
        r'(diff --git.*?)(?:\n\n|\Z)',
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.DOTALL)
        if matches:
            # Return the longest match (most likely the real patch)
            return max(matches, key=len).strip()
    # Fallback: look for diff --git anywhere
    idx = text.find("diff --git")
    if idx >= 0:
        return text[idx:].strip()
    return ""


def run_hermes_task(inst, mode, model, provider, advisor_model, advisor_provider, workdir, max_turns):
    """Run a single SWE-bench task through Hermes Agent CLI."""
    iid = inst["instance_id"]
    
    # Build the task prompt
    prompt = TASK_PROMPT_TEMPLATE.format(
        repo=inst.get("repo", ""),
        base_commit=inst.get("base_commit", ""),
        instance_id=iid,
        problem_statement=inst.get("problem_statement", "")[:3000],
    )
    
    # Prepare workdir for the repo checkout
    repo_dir = os.path.join(workdir, iid.replace("/", "__"))
    os.makedirs(repo_dir, exist_ok=True)
    
    # Setup git repo if not already done
    if not os.path.exists(os.path.join(repo_dir, ".git")):
        repo_url = inst.get("repo", "")
        bc = inst.get("base_commit", "")
        if repo_url and bc:
            clone_url = f"https://github.com/{repo_url}.git"
            try:
                subprocess.run(["git", "clone", clone_url, repo_dir], 
                             capture_output=True, timeout=120)
                subprocess.run(["git", "checkout", bc], 
                             capture_output=True, cwd=repo_dir, timeout=30)
            except Exception as e:
                print(f"      ⚠️  git setup failed: {e}")
    
    # Build hermes command
    cmd = [
        "hermes", "chat", "-q", prompt, "-Q",
        "-m", model,
        "--max-turns", str(max_turns),
    ]
    if provider:
        cmd.extend(["--provider", provider])
    
    # For advisor mode, configure advisor model via env var
    env = os.environ.copy()
    if mode == "advisor" and advisor_model:
        env["HERMES_ADVISOR_MODEL"] = advisor_model
        env["HERMES_ADVISOR_PROVIDER"] = advisor_provider or ""
        # Enable advisor tool explicitly
        cmd.extend(["-t", "advisor,terminal,file"])
    
    # Solo mode: disable advisor
    if mode == "solo":
        cmd.extend(["-t", "terminal,file"])
    
    print(f"      Running: {' '.join(cmd[:8])}... (workdir={repo_dir})")
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=repo_dir, env=env,
        )
        elapsed = time.time() - start
        
        stdout = result.stdout
        stderr = result.stderr
        
        # Extract session_id from stderr
        session_id = None
        for line in stderr.split("\n"):
            if "session_id:" in line:
                session_id = line.split("session_id:")[-1].strip()
                break
        
        # Extract patch from response
        patch = extract_patch(stdout)
        
        # Try to get session data for token/cost info
        tokens_in = tokens_out = cost = 0
        if session_id:
            try:
                export = subprocess.run(
                    ["hermes", "sessions", "export", session_id],
                    capture_output=True, text=True, timeout=15,
                    cwd="/tmp"
                )
                # Parse exported sessions (JSONL)
                export_file = os.path.join("/tmp", session_id)
                if os.path.exists(export_file):
                    with open(export_file) as f:
                        for line in f:
                            d = json.loads(line)
                            if d.get("id") == session_id:
                                tokens_in = d.get("input_tokens", 0)
                                tokens_out = d.get("output_tokens", 0)
                                cost = d.get("estimated_cost_usd", 0)
                                break
                    os.remove(export_file)
            except Exception as e:
                print(f"      ⚠️  session export failed: {e}")
        
        return {
            "instance_id": iid,
            "session_id": session_id,
            "model": model,
            "mode": mode,
            "advisor_model": advisor_model if mode == "advisor" else None,
            "model_patch": patch,
            "wall_seconds": round(elapsed, 1),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "estimated_cost_usd": cost,
            "has_patch": bool(patch and len(patch.strip()) > 10),
            "response_preview": stdout[-500:] if stdout else "",
        }
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            "instance_id": iid,
            "model": model,
            "mode": mode,
            "advisor_model": advisor_model if mode == "advisor" else None,
            "model_patch": "",
            "wall_seconds": round(elapsed, 1),
            "tokens_in": 0, "tokens_out": 0,
            "estimated_cost_usd": 0,
            "has_patch": False,
            "error": "timeout",
        }


def main():
    parser = argparse.ArgumentParser(description="Hermes Agent SWE-bench Runner")
    parser.add_argument("--mode", choices=["solo", "advisor", "both"], default="both")
    parser.add_argument("--model", default="glm-5.1")
    parser.add_argument("--provider", default="custom:glmcode")
    parser.add_argument("--advisor-model", default="glm-5.1")
    parser.add_argument("--advisor-provider", default="custom:glmcode")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--output", default=None)
    parser.add_argument("--judge", action="store_true", help="Run AI judge on existing results")
    parser.add_argument("--judge-dir", default=None)
    
    args = parser.parse_args()
    
    if args.judge:
        # Judge mode — score existing patches
        judge_dir = args.judge_dir or f"{RESULTS_BASE}/hermes"
        run_judge(judge_dir)
        return
    
    instances = load_eval_set(args.limit)
    print(f"📋 Running {len(instances)} tasks with Hermes Agent")
    print(f"   Model: {args.model} ({args.provider})")
    print(f"   Mode: {args.mode}")
    if args.mode in ["advisor", "both"]:
        print(f"   Advisor: {args.advisor_model} ({args.advisor_provider})")
    print()
    
    # Setup output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.model.replace("/", "_").replace(".", "-")
    
    modes = ["solo", "advisor"] if args.mode == "both" else [args.mode]
    
    for mode in modes:
        out_name = f"hermes-{model_slug}-{mode}"
        out_dir = args.output or os.path.join(RESULTS_BASE, "hermes", out_name)
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"{'='*60}")
        print(f"  Mode: {mode.upper()}")
        print(f"  Output: {out_dir}")
        print(f"{'='*60}")
        
        all_results = []
        for i, inst in enumerate(instances):
            iid = inst["instance_id"]
            print(f"\n  [{i+1}/{len(instances)}] {iid}")
            
            r = run_hermes_task(
                inst, mode, args.model, args.provider,
                args.advisor_model, args.advisor_provider,
                "/tmp/hermes_swe_workdir", args.max_turns,
            )
            all_results.append(r)
            
            status = "✓ patch" if r["has_patch"] else "✗ no patch"
            print(f"      {status} | {r['wall_seconds']}s | {r['tokens_in']}↓/{r['tokens_out']}↑ | ${r.get('estimated_cost_usd',0):.4f}")
        
        # Save results
        pred_file = os.path.join(out_dir, "predictions.jsonl")
        with open(pred_file, "w") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        # Save metrics
        patches = sum(1 for r in all_results if r["has_patch"])
        total_time = sum(r["wall_seconds"] for r in all_results)
        total_in = sum(r["tokens_in"] for r in all_results)
        total_out = sum(r["tokens_out"] for r in all_results)
        total_cost = sum(r.get("estimated_cost_usd", 0) for r in all_results)
        
        metrics = {
            "mode": mode,
            "model": args.model,
            "advisor_model": args.advisor_model if mode == "advisor" else None,
            "tasks": len(all_results),
            "patches_generated": patches,
            "patch_rate": f"{patches}/{len(all_results)}",
            "total_wall_seconds": round(total_time, 1),
            "total_tokens_in": total_in,
            "total_tokens_out": total_out,
            "total_cost_usd": round(total_cost, 4),
            "avg_time_per_task": round(total_time / len(all_results), 1),
        }
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n  Summary: {patches}/{len(all_results)} patches | {total_time:.0f}s | {total_in}↓/{total_out}↑ | ${total_cost:.4f}")
    
    print(f"\n{'='*60}")
    print("Done! Run with --judge to score patches.")


def run_judge(base_dir):
    """Score all patches in directory with AI judge (DS Flash)."""
    import openai
    
    # Load DS key
    with open(os.path.expanduser("~/.hermes/auth.json")) as f:
        auth = json.load(f)
    ds_key = auth["credential_pool"]["deepseek"][0]["access_token"]
    client = openai.OpenAI(api_key=ds_key, base_url="https://api.deepseek.com/v1")
    
    with open(EVAL_SET) as f:
        eval_set = {inst["instance_id"]: inst for inst in json.load(f)}
    
    base = Path(base_dir)
    print(f"🧑‍⚖️ Judging patches in {base}")
    
    for pred_file in sorted(base.rglob("predictions.jsonl")):
        with open(pred_file) as f:
            preds = [json.loads(l) for l in f if l.strip()]
        
        scores = []
        for pred in preds:
            iid = pred["instance_id"]
            patch = pred.get("model_patch", "")
            
            if not patch or len(patch.strip()) < 10:
                scores.append({"instance_id": iid, "综合评分": 0, "分析": "no patch"})
                continue
            
            inst = eval_set.get(iid, {})
            issue = inst.get("problem_statement", "")[:600]
            
            user_msg = ("Evaluate SWE-bench patch quality 0-10. Output ONLY JSON: "
                '{"correctness":N,"minimality":N,"overall":N,"analysis":"one sentence"}'
                f"\n\nIssue: {iid}\n{issue[:400]}\n\nPatch:\n{patch[:1500]}")
            
            try:
                resp = client.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=[
                        {"role":"system","content":"Output ONLY valid JSON."},
                        {"role":"user","content":user_msg},
                    ],
                    max_tokens=1024, temperature=0.1,
                )
                raw = resp.choices[0].message.content
                start, end = raw.find("{"), raw.rfind("}") + 1
                score = json.loads(raw[start:end]) if start >= 0 else {"raw": raw}
                score["instance_id"] = iid
                scores.append(score)
            except Exception as e:
                scores.append({"instance_id": iid, "error": str(e)[:80], "综合评分": 0})
        
        judge_file = pred_file.parent / "judge_scores.jsonl"
        with open(judge_file, "w") as f:
            for s in scores:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        
        valid = [s.get("overall", s.get("综合评分", 0)) for s in scores 
                 if isinstance(s.get("overall", s.get("综合评分", 0)), (int,float)) and s.get("overall", s.get("综合评分", 0)) > 0]
        avg = sum(valid)/len(valid) if valid else 0
        print(f"  {pred_file.parent.name}: {len(valid)} patches, avg={avg:.1f}")
    
    print("Judging complete.")


if __name__ == "__main__":
    main()
