#!/usr/bin/env python3
"""
SWE-bench Curated Eval Set — 题目筛选 + 存档

从 SWE-bench Verified 500 题中，按难度梯度精选 ~20 题：
  - Easy (<15 min): 7 题  — solo 模型应该能做，baseline
  - Medium (15 min - 1 hour): 7 题  — advisor 可能有帮助
  - Hard (1-4 hours): 6 题  — advisor 应该有明显帮助

筛选标准：
  1. 按难度分层抽样
  2. 覆盖不同 repo（避免单一 repo 偏差）
  3. 排除 >4 hours（3 题，太难，不适合小评测集）
  4. 每个 repo 至少 1 题（如果题量够）

输出：
  - eval_set_swe.json: 精选题目（含筛选元数据）
  - selection_report.json: 筛选过程存档
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset


def select_curated_set(seed=42):
    rng = random.Random(seed)

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    items = [dict(row) for row in ds]

    # Group by difficulty
    by_diff = defaultdict(list)
    for item in items:
        by_diff[item["difficulty"]].append(item)

    easy = by_diff["<15 min fix"]
    medium = by_diff["15 min - 1 hour"]
    hard = by_diff["1-4 hours"]

    print(f"Pool: easy={len(easy)}, medium={len(medium)}, hard={len(hard)}")

    # Strategy: repo diversity within each tier
    def pick_diverse(pool, n, rng):
        """Pick n items from pool, maximizing repo diversity."""
        by_repo = defaultdict(list)
        for item in pool:
            by_repo[item["repo"]].append(item)

        selected = []
        repos = list(by_repo.keys())
        rng.shuffle(repos)

        # Round-robin by repo
        idx = 0
        while len(selected) < n and by_repo:
            repo = repos[idx % len(repos)]
            if by_repo[repo]:
                item = by_repo[repo].pop(0)
                selected.append(item)
            if not by_repo[repo]:
                repos.remove(repo)
                if not repos:
                    break
            idx += 1

        return selected

    easy_picked = pick_diverse(easy, 7, rng)
    medium_picked = pick_diverse(medium, 7, rng)
    hard_picked = pick_diverse(hard, 6, rng)

    all_selected = easy_picked + medium_picked + hard_picked

    # Build output
    eval_set = []
    for item in all_selected:
        eval_set.append({
            "instance_id": item["instance_id"],
            "repo": item["repo"],
            "difficulty": item["difficulty"],
            "base_commit": item["base_commit"],
            "problem_statement": item["problem_statement"],
            "hints_text": item.get("hints_text", ""),
            "FAIL_TO_PASS": item.get("FAIL_TO_PASS", []),
            "PASS_TO_PASS": item.get("PASS_TO_PASS", []),
            "version": item.get("version", ""),
            "created_at": item.get("created_at", ""),
        })

    # Selection report
    repos_in_set = Counter(item["repo"] for item in all_selected)
    report = {
        "selection_method": "stratified sampling by difficulty + repo diversity",
        "seed": seed,
        "source_dataset": "princeton-nlp/SWE-bench_Verified",
        "total_in_source": len(items),
        "selected": len(eval_set),
        "difficulty_distribution": {
            "easy (<15 min)": len(easy_picked),
            "medium (15min-1h)": len(medium_picked),
            "hard (1-4h)": len(hard_picked),
        },
        "repo_distribution": dict(repos_in_set),
        "selection_criteria": [
            "1. Stratified by difficulty: easy 7, medium 7, hard 6",
            "2. Within each tier, round-robin by repo for diversity",
            "3. Excluded >4 hours (only 3 in source, too hard for small eval)",
            "4. Random seed=42 for reproducibility",
        ],
        "instances": [
            {
                "instance_id": item["instance_id"],
                "repo": item["repo"],
                "difficulty": item["difficulty"],
            }
            for item in all_selected
        ],
    }

    return eval_set, report


if __name__ == "__main__":
    out_dir = Path("/root/advisor-eval")

    eval_set, report = select_curated_set()

    # Save eval set
    eval_path = out_dir / "eval_set_swe.json"
    with open(eval_path, "w") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)
    print(f"\nEval set ({len(eval_set)} instances) saved to: {eval_path}")

    # Save selection report
    report_path = out_dir / "selection_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Selection report saved to: {report_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Curated SWE-bench Eval Set")
    print(f"{'='*60}")
    print(f"Total: {len(eval_set)} instances")
    print(f"Difficulty: easy={report['difficulty_distribution']['easy (<15 min)']}, "
          f"medium={report['difficulty_distribution']['medium (15min-1h)']}, "
          f"hard={report['difficulty_distribution']['hard (1-4h)']}")
    print(f"Repos: {len(report['repo_distribution'])}")
    print()
    for item in report["instances"]:
        print(f"  [{item['difficulty']:<20s}] {item['instance_id']}")
