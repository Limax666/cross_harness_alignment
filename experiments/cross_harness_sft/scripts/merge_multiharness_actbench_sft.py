#!/usr/bin/env python3
"""Merge filtered multi-harness teacher trajectories with safe ActBench rows.

The merge happens after source-specific safety filtering.  It preserves the
family split from the existing corpus and the deterministic ActBench split;
the VeRL builder performs the final tokenizer/runtime length audit.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / "experiments/cross_harness_sft/data/sft"
SOURCE = ROOT / "multi_harness_trl_v1"
ACT = ROOT / "actbench_safe_real_v1_sanitized"
DEST = ROOT / "multi_harness_combined_v1"


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def context(messages: list[dict[str, Any]], fallback: str) -> str:
    first = next((m for m in messages if m.get("role") == "system"), None)
    text = str((first or {}).get("content") or "").strip()
    if "<harness_context>" in text:
        start, end = text.find("<harness_context>"), text.find("</harness_context>")
        if end >= start:
            return text[start : end + len("</harness_context>")]
    return fallback


def act_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    messages = row["messages"]
    meta = row.get("meta") or {}
    harness = str(meta.get("harness") or "unknown")
    # Keep ActBench's official adapter names distinct; this lets the model learn
    # the interface-specific safety behavior rather than collapsing tools.
    record = str(meta.get("trajectory_id") or meta.get("task_id") or index)
    rid = f"actbench:{harness}:{record}"
    return {
        "messages": messages,
        "tools": row.get("tools") or [],
        "metadata": {
            "record_id": rid,
            "benchmark": "actbench",
            "harness_name": harness,
            "harness_context": context(messages, f"<harness_context>\\nname: {harness}\\ninterface: actbench_official_adapter\\n</harness_context>"),
            "source_kind": "actbench_safe_real_trajectory",
            "synthetic": False,
            "task_id": meta.get("task_id"),
            "family": meta.get("suite") or meta.get("task_id"),
            "teacher_model": meta.get("teacher_model"),
            "source_path": "data/sft/actbench_safe_real_v1_sanitized/train.jsonl",
        },
    }


def fingerprint(row: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(row["messages"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    seen: set[str] = set()
    for split in by_split:
        for row in read(SOURCE / f"{split}.jsonl"):
            key = fingerprint(row)
            if key not in seen:
                seen.add(key); by_split[split].append(row)
    for split in ("train", "validation"):
        for i, row in enumerate(read(ACT / f"{split}.jsonl")):
            converted = act_row(row, i)
            key = fingerprint(converted)
            if key not in seen:
                seen.add(key); by_split[split].append(converted)
    for split, rows in by_split.items():
        (DEST / f"{split}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8"
        )
    report = {
        "format": "cross_harness safe complete teacher trajectories v2",
        "rows_by_split": {k: len(v) for k, v in by_split.items()},
        "total_rows": sum(map(len, by_split.values())),
        "harness_by_split": {k: dict(Counter((x.get("metadata") or {}).get("harness_name") for x in v)) for k, v in by_split.items()},
        "sources": {"multi_harness_trl_v1": 1, "actbench_safe_real_v1_sanitized": 1},
    }
    (DEST / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
