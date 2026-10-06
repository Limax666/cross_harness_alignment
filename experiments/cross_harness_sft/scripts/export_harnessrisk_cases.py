#!/usr/bin/env python3
"""Export HarnessRisk's Parquet distribution to the official per-case JSON layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(args.parquet, columns=["case_id", "full_case_json"])
    exported = 0
    for case_id, raw_case in zip(table["case_id"].to_pylist(), table["full_case_json"].to_pylist()):
        case = json.loads(raw_case)
        if case.get("case_id") != case_id:
            raise ValueError(f"case_id mismatch for {case_id!r}")
        (args.output_dir / f"{case_id}.json").write_text(
            json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        exported += 1
    print(json.dumps({"exported_cases": exported, "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
