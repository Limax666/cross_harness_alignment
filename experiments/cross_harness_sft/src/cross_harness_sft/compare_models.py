"""Paired same-case comparison of base student, SFT student and teacher."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .utils import read_jsonl

BEHAVIORS = {"reflection": re.compile(r"\b(reconsider|reflect|on second thought|let me check)\b", re.I),
             "backtracking": re.compile(r"\b(backtrack|go back|revise|instead)\b", re.I),
             "verification": re.compile(r"\b(verify|double-check|cross-check|validate)\b", re.I)}


def key(row: dict[str, Any]) -> tuple[str, ...]:
    m = row.get("metadata") or {}
    return tuple(str(m.get(x) or "") for x in ("benchmark_id", "task_id", "harness_id", "safety_module")) + (str(m.get("seed", "0")),)


def scores(row: dict[str, Any]) -> dict[str, float]:
    m = row.get("metadata") or {}; reward = m.get("reward_breakdown") or {}
    utility, safety = float(reward.get("utility_reward", 0)), float(reward.get("safety_reward", 0))
    return {"utility": utility, "safety": safety, "pareto": utility * safety}


def mean(values: list[float]) -> float: return sum(values) / max(1, len(values))


def main() -> int:
    p = argparse.ArgumentParser()
    for name in ("teacher", "base", "sft"): p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--output", required=True, type=Path); args = p.parse_args()
    runs = {name: {key(row): row for row in read_jsonl(getattr(args, name))} for name in ("teacher", "base", "sft")}
    common = set.intersection(*(set(rows) for rows in runs.values()))
    if not common: raise RuntimeError("no matched benchmark/task/harness/safety/seed cells")
    aggregates: dict[str, dict[str, float]] = {}
    for role, rows in runs.items():
        aggregates[role] = {metric: mean([scores(rows[k])[metric] for k in common]) for metric in ("utility", "safety", "pareto")}
        text = [str((rows[k].get("metadata") or {}).get("visible_reasoning") or "") for k in common]
        aggregates[role].update({f"reasoning_{name}_rate": mean([float(bool(pattern.search(x))) for x in text])
                                 for name, pattern in BEHAVIORS.items()})
    recovery = {}
    for metric in ("utility", "safety", "pareto"):
        denominator = aggregates["teacher"][metric] - aggregates["base"][metric]
        recovery[metric] = None if abs(denominator) < 1e-12 else (
            aggregates["sft"][metric] - aggregates["base"][metric]) / denominator
    wins = Counter()
    for k in common:
        before, after = scores(runs["base"][k])["pareto"], scores(runs["sft"][k])["pareto"]
        wins["improved" if after > before else "regressed" if after < before else "tied"] += 1
    report = {"schema": "cross_harness.paired_evaluation.v2", "matched_episodes": len(common),
              "unmatched": {role: len(rows) - len(common) for role, rows in runs.items()},
              "aggregate": aggregates, "capability_recovery": recovery, "paired_pareto": dict(wins),
              "definition": "recovery=(SFT-base)/(teacher-base); all roles matched on benchmark/task/harness/safety/seed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
