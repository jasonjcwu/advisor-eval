#!/usr/bin/env python3
"""
Pick 6 harder instances from the curated eval set (3 MEDIUM + 3 HARD).
"""
import json, sys
from pathlib import Path

SRC = Path("/root/advisor-eval/eval_set_swe.json")
DST = Path("/root/advisor-eval/eval_set_swe_hard6.json")

with open(SRC) as f:
    all_items = json.load(f)

# Pick: MEDIUM 3 + HARD 3
selected_ids = [
    # MEDIUM
    "psf__requests-2931",
    "sympy__sympy-11618",
    "pydata__xarray-2905",
    # HARD
    "scikit-learn__scikit-learn-25102",
    "django__django-10554",
    "sphinx-doc__sphinx-11510",
]

selected = [i for i in all_items if i["instance_id"] in selected_ids]
# Preserve order
selected.sort(key=lambda i: selected_ids.index(i["instance_id"]))

for item in selected:
    diff_short = {"15 min - 1 hour": "MED", "1-4 hours": "HARD"}[item["difficulty"]]
    print(f"  [{diff_short}] {item['instance_id']}")

with open(DST, "w") as f:
    json.dump(selected, f, indent=2)
print(f"\nSaved {len(selected)} instances to {DST}")
