#!/usr/bin/env python3
"""Freeze an AgentDojo comparison set from families held out of SFT training.

The teacher reference is the original, unfiltered Codex collection. Repeated
records of a task must agree on the official verifier scores before deduping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--teacher-raw", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    train_families = {row["meta"]["family"] for row in read_jsonl(args.train)}
    validation_families = {row["meta"]["family"] for row in read_jsonl(args.validation)}
    overlap = train_families & validation_families
    if overlap:
        raise ValueError(f"train/validation family leakage: {sorted(overlap)}")

    by_task: dict[str, dict] = {}
    duplicates = 0
    for row in read_jsonl(args.teacher_raw):
        meta = row.get("metadata") or {}
        if meta.get("benchmark_id") != "agentdojo" or meta.get("semantic_task_family_id") not in validation_families:
            continue
        if meta.get("harness_id") != "codex_native" or meta.get("safety_module") != "none" or meta.get("seed") != 0:
            continue
        if row.get("status") != "completed":
            continue
        reward = meta.get("reward_breakdown") or {}
        if not {"utility_reward", "safety_reward"} <= reward.keys():
            continue
        task_id = str(meta["task_id"])
        record = {
            "task_id": task_id,
            "family": str(meta["semantic_task_family_id"]),
            "task_type": str(meta["task_type"]),
            "seed": 0,
            "teacher_harness": "codex_native",
            "teacher_model": str(meta["teacher_id"]),
            "benchmark_version": str(meta["benchmark_version"]),
            "system_prompt": str(meta["skillrl_system_instruction"]),
            "user_query": str(meta["user_query"]),
            "teacher_utility": float(reward["utility_reward"]),
            "teacher_safety": float(reward["safety_reward"]),
        }
        previous = by_task.get(task_id)
        if previous is not None:
            duplicates += 1
            if previous != record:
                raise ValueError(f"inconsistent duplicate teacher record: {task_id}")
        else:
            by_task[task_id] = record
    if not by_task:
        raise ValueError("no teacher episodes match the held-out families")
    selected = [by_task[key] for key in sorted(by_task)]
    if {row["family"] for row in selected} != validation_families:
        raise ValueError("teacher reference does not cover every held-out family")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "evaluation_plan.jsonl"
    with plan_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema": "agentdojo_9b_comparison.plan.v1",
        "benchmark": "official AgentDojo v1.2.2 via local benchmark worker",
        "task_count": len(selected),
        "heldout_families": sorted(validation_families),
        "train_family_overlap": 0,
        "task_types": dict(Counter(row["task_type"] for row in selected)),
        "source_duplicates_removed": duplicates,
        "teacher_reference": "historical Codex trajectories, not a fresh unified-runner re-evaluation",
        "inputs_sha256": {name: sha256(path) for name, path in (("train", args.train), ("validation", args.validation), ("teacher_raw", args.teacher_raw))},
        "plan_sha256": sha256(plan_path),
    }
    (args.output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
