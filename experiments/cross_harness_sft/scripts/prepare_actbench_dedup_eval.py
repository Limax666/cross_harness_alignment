#!/usr/bin/env python3
"""Prepare a family-disjoint ActBench evaluation suite for the v7 Qwen3.5-2B SFT.

Families present in either the actual SFT train or validation view are excluded.
The remaining published task pairs are emitted as the suite for all official
ActBench backends; each task_id represents one benign and one attack case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
RELEASE = ROOT / "experiments/cross_harness_sft/data/sft/teacher_release_v7"
VIEW = RELEASE / "full_view_qwen35_2b_base_bs16_v2"
RAW = ROOT / "experiments/cross_harness_sft/data/raw/actbench_public_v1"
OUT = ROOT / "experiments/cross_harness_sft/outputs/actbench/dedup_qwen35_2b_v7"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()

    train_path = VIEW / "train_balanced.jsonl"
    val_path = VIEW / "validation_natural.jsonl"
    split_path = RELEASE / "family_split_manifest.jsonl"
    pairs_path = RAW / "data/task_pairs/test-00000-of-00001.parquet"
    for p in (train_path, val_path, split_path, pairs_path):
        if not p.is_file():
            raise SystemExit(f"Required input missing: {p}")

    excluded = set()
    for p in (train_path, val_path):
        excluded.update(
            row["task_id"] for row in read_jsonl(p)
            if row.get("benchmark") == "actbench" and row.get("task_id")
        )

    split_rows = read_jsonl(split_path)
    assigned_split = {
        row.get("source_family_id"): row.get("split")
        for row in split_rows if row.get("benchmark_namespace") == "actbench"
    }
    train_families = {
        row["task_id"] for row in read_jsonl(train_path)
        if row.get("benchmark") == "actbench" and row.get("task_id")
    }
    val_families = {
        row["task_id"] for row in read_jsonl(val_path)
        if row.get("benchmark") == "actbench" and row.get("task_id")
    }
    # Exclude families actually rendered into the SFT input views. The broader
    # family manifest includes candidate families that failed filtering and were
    # never used to train this checkpoint; excluding those would needlessly shrink
    # the held-out benchmark. Keep assignments as an audit consistency check.
    for task_id in train_families:
        if assigned_split.get(f"actbench:{task_id}") not in (None, "train"):
            raise SystemExit(f"SFT train family has inconsistent split assignment: {task_id}")
    for task_id in val_families:
        if assigned_split.get(f"actbench:{task_id}") not in (None, "validation"):
            raise SystemExit(f"SFT validation family has inconsistent split assignment: {task_id}")
    excluded = train_families | val_families

    table = pq.read_table(pairs_path, columns=["task_id", "scenario_id", "behavior_id", "behavior_type"])
    pairs = table.to_pylist()
    ids = [str(row["task_id"]) for row in pairs]
    if len(ids) != 300 or len(set(ids)) != 300:
        raise SystemExit(f"Expected 300 unique published task pairs, found {len(ids)} rows / {len(set(ids))} IDs")

    heldout = [row for row in pairs if row["task_id"] not in excluded]
    if not heldout:
        raise SystemExit("No deduplicated ActBench families remain")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_ids = [str(row["task_id"]) for row in heldout]
    (output_dir / "task_ids.txt").write_text("\n".join(task_ids) + "\n", encoding="utf-8")
    (output_dir / "suite.csv").write_text(",".join(task_ids), encoding="utf-8")
    manifest = {
        "format": "actbench_family_disjoint_eval_v1",
        "benchmark_revision": "31bd732e9c083ddeb19bf152048386d38e511c90",
        "source_task_pairs": len(pairs),
        "excluded_sft_train_or_validation_families": sorted(excluded),
        "excluded_family_count": len(excluded),
        "excluded_train_family_count": len(train_families),
        "excluded_validation_family_count": len(val_families),
        "heldout_pair_count": len(heldout),
        "heldout_case_count": 2 * len(heldout),
        "rollouts_per_case": 3,
        "cases_per_model_harness": 6 * len(heldout),
        "source_hashes": {
            "train_view": sha256(train_path),
            "validation_view": sha256(val_path),
            "family_split_manifest": sha256(split_path),
            "task_pairs_parquet": sha256(pairs_path),
        },
        "task_pairs": heldout,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in (
        "format", "excluded_family_count", "heldout_pair_count", "heldout_case_count",
        "rollouts_per_case", "cases_per_model_harness",
    )}, indent=2))
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
