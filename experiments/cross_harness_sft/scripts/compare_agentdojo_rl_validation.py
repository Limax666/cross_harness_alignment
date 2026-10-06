#!/usr/bin/env python3
"""Paired official AgentDojo outcome comparison on frozen validation cases."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[tuple[str, int], dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    indexed = {}
    for row in rows:
        key = (row["task_id"], row["seed"])
        if key in indexed:
            raise ValueError(f"duplicate case and seed: {key}")
        indexed[key] = row
    return indexed


def rate(rows: list[dict], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows)


def summarize(rows: list[dict]) -> dict:
    clean = [row for row in rows if row["task_id"].endswith(":clean")]
    injection = [row for row in rows if ":injection_task_" in row["task_id"]]
    return {
        "episodes": len(rows), "clean_episodes": len(clean), "injection_episodes": len(injection),
        "clean_utility": rate(clean, "official_utility"),
        "injection_utility": rate(injection, "official_utility"),
        "injection_asr": rate(injection, "official_attack_success"),
        "incomplete_rate": sum(row["termination"] != "completed" for row in rows) / len(rows),
        "invalid_call_rate": sum(bool(row["invalid_calls"]) for row in rows) / len(rows),
    }


def compare(base: Path, rl: Path) -> dict:
    before, after = load(base), load(rl)
    if before.keys() != after.keys():
        raise ValueError("SFT and RL evaluations do not have identical cases and seeds")
    keys = sorted(before)
    family_counts = defaultdict(int)
    for key in keys:
        for row in (before[key], after[key]):
            if row.get("status") != "completed":
                raise ValueError(f"evaluation error prevents paired comparison: {key}")
            if row.get("official_utility") not in (0, 1, 0.0, 1.0):
                raise ValueError(f"missing official utility: {key}")
            if ":injection_task_" in row["task_id"] and row.get("official_attack_success") not in (0, 1, 0.0, 1.0):
                raise ValueError(f"missing official injection verdict: {key}")
        family_counts[key[0].rsplit(":", 1)[0]] += 1
    if len(family_counts) < 2:
        raise ValueError("comparison needs at least two held-out families")
    sft = summarize([before[key] for key in keys])
    tuned = summarize([after[key] for key in keys])
    return {"sft": sft, "rl": tuned,
            "delta_rl_minus_sft": {name: tuned[name] - sft[name] for name in sft if name.endswith(("utility", "asr", "rate"))},
            "held_out_families": sorted(family_counts), "paired_cases": len(keys),
            "denominator_policy": "all matched official AgentDojo verdicts; non-completed model turns remain in denominator"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--rl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.sft, args.rl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
