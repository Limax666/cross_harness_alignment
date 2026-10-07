#!/usr/bin/env python3
"""Build a family-gated task catalog without copying teacher conversations."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: expected JSON object")
                yield row


def build(release: Path, output: Path) -> dict[str, Any]:
    manifest_path = release / "family_split_manifest.jsonl"
    train_path = release / "preflight_qwen35_2b_base_4096_v1" / "qualified_train.jsonl"
    val_path = release / "preflight_qwen35_2b_base_4096_v1" / "qualified_validation.jsonl"
    required = (manifest_path, train_path, val_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen release inputs: {missing}")

    family_splits: dict[tuple[str, str], str] = {}
    for row in read_jsonl(manifest_path):
        key = (str(row["benchmark_namespace"]), str(row["source_family_id"]))
        if key in family_splits:
            raise ValueError(f"duplicate source family in manifest: {key}")
        family_splits[key] = str(row["split"])

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    input_counts: Counter = Counter()
    unique: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {
        "train": {}, "validation": {}
    }
    for split in ("train", "validation"):
        path = train_path if split == "train" else val_path
        for row in read_jsonl(path):
            input_counts[split] += 1
            if row.get("split") != split:
                raise ValueError(f"row split mismatch in {path}: {row.get('record_id')}")
            benchmark = str(row.get("benchmark") or "")
            harness = str(row.get("harness_name") or "")
            task_id = str(row.get("task_id") or "")
            family = str(row.get("source_family_id") or "")
            if not all((benchmark, harness, task_id, family)):
                raise ValueError(f"missing task identity in {path}: {row.get('record_id')}")
            if family_splits.get((benchmark, family)) != split:
                raise ValueError(f"task family is not {split}: {(benchmark, family)}")
            if row.get("messages") or row.get("teacher_model") or row.get("record_id"):
                # These are deliberately not copied into the catalog. The source
                # accepted/preflight JSONL is an inventory, never an RL prompt set.
                pass
            objective = str(row.get("source_objective") or "").strip()
            context = row.get("harness_context")
            tools = row.get("tools")
            if not objective or not isinstance(context, str) or not context.strip() or not isinstance(tools, list):
                raise ValueError(f"task lacks source prompt or native contract: {benchmark}/{harness}/{task_id}")
            key = (benchmark, harness, task_id)
            task = {
                "benchmark": benchmark,
                "harness_name": harness,
                "task_id": task_id,
                "source_family_id": family,
                "split": split,
                "source_task_type": str(row.get("source_task_type") or ""),
                "primary_behavior_type": str(row.get("primary_behavior_type") or ""),
                "quality_category": str(row.get("quality_category") or ""),
                "source_objective": objective,
                "harness_context": context,
                "tools": tools,
                "source_fixture_path": str(row.get("source_fixture_path") or ""),
                "source_fixture_sha256": str(row.get("source_fixture_sha256") or ""),
                "task_metadata": row.get("task_metadata") or {},
            }
            previous = unique[split].get(key)
            if previous is not None and previous != task:
                raise ValueError(f"conflicting contracts for task cell: {key}")
            unique[split][key] = task

    # Task IDs may be repeated in the input release; count the deduplicated catalog.
    counts = {split: Counter((r["benchmark"], r["harness_name"]) for r in records.values())
              for split, records in unique.items()}
    for split in ("train", "validation"):
        path = output / f"{split}_task_catalog.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for key in sorted(unique[split]):
                stream.write(json.dumps(unique[split][key], ensure_ascii=False, sort_keys=True) + "\n")

    # No test/review task identity or teacher conversation is written to the catalog.
    for records in unique.values():
        for task in records.values():
            if task["split"] not in {"train", "validation"}:
                raise AssertionError("held-out family entered task catalog")
    report = {
        "status": "task_catalog_only_environment_not_yet_qualified",
        "source_release": str(release.resolve()),
        "family_manifest_sha256": sha256(manifest_path),
        "qualified_train_sha256": sha256(train_path),
        "qualified_validation_sha256": sha256(val_path),
        "families": dict(Counter(family_splits.values())),
        "input_rows": dict(input_counts),
        "unique_task_cells": {split: len(unique[split]) for split in unique},
        "source_harness_counts": {
            split: {f"{benchmark}/{harness}": count
                    for (benchmark, harness), count in sorted(counts[split].items())}
            for split in counts
        },
        "excluded_splits": ["test", "review_only"],
        "teacher_messages_in_catalog": False,
        "online_rollout_ready": False,
    }
    (output / "catalog_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.release, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
