#!/usr/bin/env python3
"""
Self-iterative SWE-bench evaluation framework.

Two parallel tracks:
  1. Claude Code (local) — user runs on laptop with modified source
  2. Hermes Agent (cloud) — runs here with ask_advisor tool

Each track produces patches. AI judge (GLM-4-Flash, free) scores them.
Iteration loop: run → judge → analyze failures → improve prompt → re-run.

Usage:
  # Track 2: Hermes Agent cloud eval
  python3 eval_framework.py --track hermes --pair ds-flash+ds-pro --limit 3

  # AI Judge: score existing patches
  python3 eval_framework.py --judge --results-dir /path/to/results

  # Compare tracks
  python3 eval_framework.py --compare --dirs track1,track2
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────
HERMES_ROOT = "/root/hermes-agent"
EVAL_ROOT = "/root/advisor-eval"
RESULTS_ROOT = os.path.join(EVAL_ROOT, "iteration_results")

if HERMES_ROOT not in sys.path:
    sys.path.insert(0, HERMES_ROOT)
if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

# ── Model configs ────────────────────────────────────────────────────
MODELS = {
    "ds-flash": {
        "model": "deepseek-chat",  # V4 (Flash tier pricing)
        "base_url": "https://api.deepseek.com/v1",
        "api_mode": "openai",
        "cost_per_M_in": 0.1,  # $/M input tokens
        "cost_per_M_out": 0.3,
    },
    "ds-pro": {
        "model": "deepseek-reasoner",  # R1 (Pro tier)
        "base_url": "https://api.deepseek.com/v1",
        "api_mode": "openai",
        "cost_per_M_in": 0.55,
        "cost_per_M_out": 2.19,
    },
    "glm4-flash": {
        "model": "glm-4-flash",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_mode": "openai",
        "cost_per_M_in": 0.0,  # Free!
        "cost_per_M_out": 0.0,
    },
    "glm-5.1": {
        "model": "glm-5.1",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_mode": "openai",
        "cost_per_M_in": 0.5,
        "cost_per_M_out": 0.5,
    },
}

def _load_api_key(name):
    """Load API key from hermes auth.json."""
    try:
        with open(os.path.expanduser("~/.hermes/auth.json")) as f:
            auth = json.load(f)
        pool = auth.get("credential_pool", {})
        for candidate in [name, f"custom:{name}", name.replace("-", "")]:
            creds = pool.get(candidate, [])
            if creds:
                return creds[0].get("access_token", "")
    except Exception:
        pass
    # Fallback env
    env_map = {"ds-flash": "DEEPSEEK_API_KEY", "ds-pro": "DEEPSEEK_API_KEY",
               "glm4-flash": "GLMCODE_API_KEY", "glm-5.1": "GLMCODE_API_KEY"}
    return os.environ.get(env_map.get(name, ""), "")


# ── AI Judge (智谱 AI 裁判员模式) ─────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """\
你是一名公正的代码 patch 评分裁判。你需要在 SWE-bench bug 修复场景下（场景定义：\
给定一个 GitHub issue 和对应代码库，评估生成的 git diff patch 质量），按照以下原则评估。

<评分原则>
根据以下维度对 patch 进行评价，按权重从高到低排序：
1. **正确性** (40%): patch 是否正确修复了 issue 描述的问题？逻辑是否正确？
2. **最小性** (25%): 改动是否最小化？是否有多余的变更？是否改变了不该改的代码？
3. **完整性** (15%): 是否遗漏了必要的修改？相关联的文件/函数是否都处理了？
4. **可测试性** (10%): patch 后代码能否正常编译/运行？是否引入新 bug？
5. **代码风格** (10%): 是否符合项目现有风格？变量命名、注释是否合理？

每个维度的评分范围为 0 至 10 分。

<分值标准>
三档赋分，尽可能严格：
- 第一档 (8-10) 优秀：各方面均严格符合标准，patch 可直接提交
- 第二档 (5-7) 及格：主要方向正确，但有可改进之处
- 第三档 (0-4) 不及格：有重大缺陷，无法解决 issue 或引入新问题

<评分步骤>
1. 阅读 issue 描述，理解需要修复的问题
2. 检查 patch 改动了哪些文件和位置
3. 分析每个改动是否与 issue 相关
4. 检查是否有多余或遗漏的改动
5. 综合给出每个维度的分数

<输出要求>
仅输出 JSON 格式内容，禁止任何无关说明。
"""

JUDGE_USER_TEMPLATE = """\
请评估以下 SWE-bench patch：

## Issue 描述
{issue}

## 项目：{repo}

## 生成的 Patch (git diff)
```diff
{patch}
```

## 参考答案 (gold patch)
```diff
{gold_patch}
```

请按评分维度给出 JSON 评分。"""


def run_judge(results_dir, judge_model="glm4-flash"):
    """Run AI judge on all patches in results_dir."""
    import openai

    cfg = MODELS[judge_model]
    api_key = _load_api_key(judge_model)
    if not api_key:
        print(f"No API key for judge model {judge_model}")
        return

    client = openai.OpenAI(api_key=api_key, base_url=cfg["base_url"])

    # Load eval set for gold patches and issue descriptions
    eval_set_path = os.path.join(EVAL_ROOT, "eval_set_swe_hard6.json")
    with open(eval_set_path) as f:
        eval_set = {inst["instance_id"]: inst for inst in json.load(f)}

    # Find all prediction files
    results_dir = Path(results_dir)
    scored = 0
    for pred_file in sorted(results_dir.rglob("predictions.jsonl")):
        judge_file = pred_file.parent / "judge_scores.jsonl"
        if judge_file.exists():
            existing = set()
            with open(judge_file) as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        existing.add(d["instance_id"])
        else:
            existing = set()

        with open(pred_file) as f:
            predictions = [json.loads(line) for line in f if line.strip()]

        for pred in predictions:
            iid = pred["instance_id"]
            patch = pred.get("model_patch", "")

            if iid in existing:
                continue
            if not patch or len(patch.strip()) < 20:
                # No patch to score
                score = {"instance_id": iid, "综合评分": 0, "error": "no patch generated"}
            else:
                inst = eval_set.get(iid, {})
                issue = inst.get("problem_statement", "")[:2000]
                gold = inst.get("patch", "")[:2000]
                repo = inst.get("repo", "?")

                user_msg = JUDGE_USER_TEMPLATE.format(
                    issue=issue, repo=repo, patch=patch[:3000], gold_patch=gold
                )

                try:
                    resp = client.chat.completions.create(
                        model=cfg["model"],
                        messages=[
                            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        max_tokens=1024,
                        temperature=0.1,
                    )
                    raw = resp.choices[0].message.content
                    # Try to parse JSON from response
                    try:
                        # Find JSON in response
                        start = raw.find("{")
                        end = raw.rfind("}") + 1
                        if start >= 0 and end > start:
                            score = json.loads(raw[start:end])
                        else:
                            score = {"raw": raw}
                    except json.JSONDecodeError:
                        score = {"raw": raw, "parse_error": True}
                    score["instance_id"] = iid
                    score["judge_model"] = cfg["model"]
                except Exception as e:
                    score = {"instance_id": iid, "error": str(e)}

            # Append score
            with open(judge_file, "a") as f:
                f.write(json.dumps(score, ensure_ascii=False) + "\n")
            scored += 1

            total = score.get("综合评分", 0)
            print(f"  {'✅' if isinstance(total, (int, float)) and total >= 5 else '❌'} "
                  f"{iid}: score={total}")

    print(f"\nJudged {scored} patches with {cfg['model']} (cost: $0.00 — free)")
    return scored


# ── Hermes Agent track ───────────────────────────────────────────────
def run_hermes_track(exec_name, adv_name, instances, max_turns=12, iteration=1):
    """Run SWE-bench tasks using Hermes Agent's built-in agent loop + advisor tool."""
    from bench_multi import (
        MODELS as BM_MODELS, setup_workspace, build_prompt, get_diff,
        run_task, _load_ds_key, TOOLS, EXECUTOR_PROMPT, SOLO_PROMPT,
    )
    from tools.advisor_tool import call_advisor, load_advisor_config

    exec_cfg = BM_MODELS.get(exec_name)
    adv_cfg = BM_MODELS.get(adv_name)
    if not exec_cfg:
        exec_cfg = MODELS[exec_name]
    if not adv_cfg:
        adv_cfg = MODELS[adv_name]

    # Resolve API keys
    exec_key = exec_cfg.get("api_key") or _load_ds_key()
    adv_key = adv_cfg.get("api_key") or _load_api_key(adv_name) or exec_key

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(RESULTS_ROOT) / f"iter{iteration}" / f"hermes_{exec_name}+{adv_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"solo": [], "advisor": []}

    for mode, solo in [("solo", True), ("advisor", False)]:
        mode_dir = out_dir / mode
        mode_dir.mkdir(exist_ok=True)

        # Setup advisor config
        cfg = load_advisor_config()
        if not solo:
            cfg.update(
                model=adv_cfg["model"],
                api_key=adv_key,
                base_url=adv_cfg["base_url"],
                enabled=True,
            )

        label = f"hermes-{exec_name}-{mode}"
        print(f"\n{'='*60}")
        print(f"Track: Hermes Agent | {label}")
        print(f"{'='*60}")

        for idx, inst in enumerate(instances):
            iid = inst["instance_id"]
            print(f"\n  [{idx+1}/{len(instances)}] {iid}")

            ws = setup_workspace(inst, base=f"/tmp/swe-{label}")
            prompt = build_prompt(inst, solo=solo)

            t0 = time.time()
            try:
                info = run_task(prompt, ws, max_turns, solo, exec_cfg, cfg)
            except Exception as e:
                import traceback
                traceback.print_exc()
                info = {"error": str(e), "advisor_calls": 0, "tool_calls": {},
                        "turns": 0, "tokens_in": 0, "tokens_out": 0}

            elapsed = time.time() - t0
            patch = get_diff(ws)

            r = {
                "instance_id": iid,
                "model_name_or_path": f"hermes-{exec_name}",
                "model_patch": patch,
                "metrics": {
                    "executor_model": exec_cfg["model"],
                    "advisor_model": adv_cfg["model"] if not solo else None,
                    "mode": mode,
                    "advisor_calls": info.get("advisor_calls", 0),
                    "tool_calls": info.get("tool_calls", {}),
                    "num_turns": info.get("turns", 0),
                    "executor_input_tokens": info.get("tokens_in", 0),
                    "executor_output_tokens": info.get("tokens_out", 0),
                    "total_seconds": round(elapsed, 1),
                }
            }
            results[mode].append(r)

            # Save prediction
            with open(mode_dir / "predictions.jsonl", "a") as f:
                f.write(json.dumps({
                    "instance_id": iid,
                    "model_name_or_path": label,
                    "model_patch": patch,
                }, ensure_ascii=False) + "\n")

            p = "✓" if patch else "✗"
            adv_c = info.get("advisor_calls", 0)
            print(f"    {p} adv={adv_c} turns={info.get('turns',0)} {elapsed:.0f}s")

        # Save metrics
        patches = sum(1 for r in results[mode] if r["model_patch"])
        total_tokens_in = sum(r["metrics"]["executor_input_tokens"] for r in results[mode])
        total_tokens_out = sum(r["metrics"]["executor_output_tokens"] for r in results[mode])

        with open(mode_dir / "metrics.json", "w") as f:
            json.dump({
                "track": "hermes",
                "iteration": iteration,
                "executor": exec_cfg["model"],
                "advisor": adv_cfg["model"] if mode == "advisor" else None,
                "mode": mode,
                "total": len(instances),
                "patches": patches,
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
                "advisor_calls": sum(r["metrics"]["advisor_calls"] for r in results[mode]),
            }, f, indent=2)

    # Summary
    s = sum(1 for r in results["solo"] if r["model_patch"])
    a = sum(1 for r in results["advisor"] if r["model_patch"])
    print(f"\n  Hermes {exec_name}+{adv_name}: solo={s}/{len(instances)} advisor={a}/{len(instances)}")
    return results


# ── Compare tracks ───────────────────────────────────────────────────
def compare_tracks(dirs):
    """Compare results from Claude Code track vs Hermes Agent track."""
    all_data = {}

    for d in dirs.split(","):
        d = Path(d.strip())
        if not d.exists():
            # Try under RESULTS_ROOT
            d = Path(RESULTS_ROOT) / d
        if not d.exists():
            print(f"  ⚠ {d} not found")
            continue

        label = d.name
        solo_patches = advisor_patches = 0
        solo_scores = []
        advisor_scores = []
        total_tokens = 0

        for mode in ["solo", "advisor"]:
            mode_dir = d / mode
            if not mode_dir.exists():
                continue

            # Count patches
            pred_file = mode_dir / "predictions.jsonl"
            if pred_file.exists():
                with open(pred_file) as f:
                    preds = [json.loads(l) for l in f if l.strip()]
                patches = sum(1 for p in preds if p.get("model_patch"))
            else:
                patches = 0

            # Read judge scores
            judge_file = mode_dir / "judge_scores.jsonl"
            scores = []
            if judge_file.exists():
                with open(judge_file) as f:
                    for line in f:
                        if line.strip():
                            s = json.loads(line)
                            total = s.get("综合评分", 0)
                            if isinstance(total, (int, float)):
                                scores.append(total)

            # Read metrics
            metrics_file = mode_dir / "metrics.json"
            tokens = 0
            if metrics_file.exists():
                with open(metrics_file) as f:
                    m = json.load(f)
                tokens = m.get("tokens_in", 0) + m.get("tokens_out", 0)

            if mode == "solo":
                solo_patches = patches
                solo_scores = scores
            else:
                advisor_patches = patches
                advisor_scores = []
            total_tokens += tokens

        all_data[label] = {
            "solo_patches": solo_patches,
            "advisor_patches": advisor_patches,
            "solo_avg_score": sum(solo_scores) / len(solo_scores) if solo_scores else "N/A",
            "advisor_avg_score": sum(advisor_scores) / len(advisor_scores) if advisor_scores else "N/A",
            "tokens": total_tokens,
        }

    # Print comparison table
    print(f"\n{'='*70}")
    print("TRACK COMPARISON")
    print(f"{'='*70}")
    print(f"{'Track':<35} {'Solo':>6} {'Advisor':>8} {'Δ':>5} {'S.Score':>8} {'A.Score':>8}")
    print("-" * 70)
    for label, d in all_data.items():
        s = d["solo_patches"]
        a = d["advisor_patches"]
        delta = a - s
        ss = d["solo_avg_score"]
        sa = d["advisor_avg_score"]
        ss_str = f"{ss:.1f}" if isinstance(ss, float) else ss
        sa_str = f"{sa:.1f}" if isinstance(sa, float) else sa
        print(f"{label:<35} {s:>6} {a:>8} {delta:>+5} {ss_str:>8} {sa_str:>8}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Self-iterative SWE-bench evaluation")
    p.add_argument("--track", choices=["hermes", "claude-code"], default="hermes",
                   help="Which agent framework to evaluate")
    p.add_argument("--pair", default="ds-flash+ds-pro",
                   help="Model pair: executor+advisor (e.g. ds-flash+ds-pro)")
    p.add_argument("--eval-set", default=os.path.join(EVAL_ROOT, "eval_set_swe_hard6.json"))
    p.add_argument("--limit", type=int, default=None, help="Max tasks to run")
    p.add_argument("--max-turns", type=int, default=15, help="Max agent loop turns per task")
    p.add_argument("--iteration", type=int, default=1, help="Iteration number")
    p.add_argument("--judge", action="store_true", help="Run AI judge on existing results")
    p.add_argument("--judge-model", default="glm4-flash", help="Judge model (free: glm4-flash)")
    p.add_argument("--compare", action="store_true", help="Compare tracks")
    p.add_argument("--dirs", default=None, help="Comma-separated result dirs for comparison")
    p.add_argument("--output", default=RESULTS_ROOT, help="Output root directory")
    args = p.parse_args()

    if args.judge:
        targets = [args.dirs] if args.dirs else [args.output]
        for t in targets:
            print(f"\n🧑‍⚖️ Judging patches in {t}")
            run_judge(t, args.judge_model)
    elif args.compare:
        compare_tracks(args.dirs or args.output)
    else:
        # Load eval set
        with open(args.eval_set) as f:
            instances = json.load(f)
        if args.limit:
            instances = instances[:args.limit]

        exec_name, adv_name = args.pair.split("+")
        print(f"Track: {args.track} | Exec: {exec_name} | Advisor: {adv_name} | Tasks: {len(instances)}")

        if args.track == "hermes":
            run_hermes_track(exec_name, adv_name, instances, args.max_turns, args.iteration)
        else:
            print("Claude Code track: run locally with modified source code")
            print("See README-LOCAL.md for setup instructions")
