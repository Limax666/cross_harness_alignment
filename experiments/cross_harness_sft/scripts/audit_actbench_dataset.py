#!/usr/bin/env python3
"""Download a reproducible ActBench schema sample before SFT conversion.

This deliberately writes raw rows and a schema report only. The subsequent
converter may admit only trajectories with a verified harness identifier,
complete conversation/tool evidence, and a safe realised outcome.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset


def shape(value):
    if isinstance(value, dict):
        return {str(k): shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return type(value).__name__


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--sample-per-config", type=int, default=32)
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = get_dataset_config_names("ZJUICSR/ActBench", revision=args.revision)
    report = {"dataset": "ZJUICSR/ActBench", "revision": args.revision, "configs": {}}
    for config in configs:
        stream = load_dataset("ZJUICSR/ActBench", config, split="test", streaming=True, revision=args.revision)
        rows = []
        field_presence: Counter[str] = Counter()
        for row in stream:
            rows.append(row)
            field_presence.update(row.keys())
            if len(rows) >= args.sample_per_config:
                break
        (args.output_dir / f"{config}.sample.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        report["configs"][config] = {
            "sample_rows": len(rows), "field_presence": dict(field_presence),
            "first_row_schema": shape(rows[0]) if rows else None,
        }
    (args.output_dir / "schema_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
