#!/usr/bin/env python3
"""Choose a deterministic, family-balanced subset of HarnessRisk cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", default="harnessrisk-gpt56-sol-real-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for path in sorted(args.data_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        case_id = row.get("case_id")
        if not isinstance(case_id, str):
            continue
        family = case_id.split("_", 1)[0]
        cases.append((family, case_id))
    if args.count <= 0 or args.count > len(cases):
        raise SystemExit(f"--count must be in [1, {len(cases)}]")
    groups: dict[str, list[str]] = defaultdict(list)
    for family, case_id in cases:
        groups[family].append(case_id)
    for family in groups:
        groups[family].sort(key=lambda x: hashlib.sha256(f"{args.seed}:{x}".encode()).hexdigest())
    families = sorted(groups)
    base, remainder = divmod(args.count, len(families))
    selected: list[str] = []
    for index, family in enumerate(families):
        take = base + int(index < remainder)
        if len(groups[family]) < take:
            raise SystemExit(f"family {family} has only {len(groups[family])} cases, need {take}")
        selected.extend(groups[family][:take])
    selected.sort()
    payload = {
        "schema": "cross_harness_sft.harnessrisk_real_subset.v1",
        "seed": args.seed,
        "requested_count": args.count,
        "selected_count": len(selected),
        "families": {family: sum(x.startswith(family + "_") for x in selected) for family in families},
        "case_ids": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
