#!/usr/bin/env python3
"""Write a corrected analytics copy of an RL journal with legacy fixed counts.

The source journal remains immutable. Only fields exactly derivable from its
per-cell reward arrays are changed; training rewards and optimizer metrics are
never recomputed or modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def corrected_rows(rows: list[dict], source: str) -> list[dict]:
    output = []
    for row in rows:
        fixed = dict(row)
        if row.get("record_type") == "run":
            metadata = dict(row.get("metadata", {}))
            metadata["analytics_correction_source"] = source
            metadata["analytics_correction_fields"] = [
                "sampled_groups", "sampled_rollouts", "scored_rollouts",
                "unscorable_rollouts", "zero_variance_group_rate",
            ]
            fixed["metadata"] = metadata
        elif row.get("record_type") == "metrics":
            rewards = row["rewards_by_cell"]
            if not rewards or any(len(values) != 4 for values in rewards.values()):
                raise ValueError("expected G=4 reward arrays for every registered cell")
            groups = len(rewards)
            rollouts = sum(len(values) for values in rewards.values())
            fixed.update({
                "sampled_groups": groups,
                "sampled_rollouts": rollouts,
                "scored_rollouts": rollouts,
                "unscorable_rollouts": 0,
                "zero_variance_group_rate": sum(len(set(values)) == 1 for values in rewards.values()) / groups,
            })
        else:
            raise ValueError("unexpected journal record type")
        output.append(fixed)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    fixed = corrected_rows(rows, str(args.source.resolve()))
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in fixed), encoding="utf-8")
    print(f"corrected {sum(row.get('record_type') == 'metrics' for row in fixed)} metric rows: {args.output}")


if __name__ == "__main__":
    main()
