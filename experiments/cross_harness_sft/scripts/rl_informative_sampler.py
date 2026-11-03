"""Balanced CHS-PO v4 sampling with a frozen train-only signal snapshot.

Per source/harness/arm, half of the cells retain the v3 deterministic
coverage rotation. The other half favor train cells whose recent G=4 outcomes
still vary, while every eligible cell retains positive selection probability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from rl_multiharness_stratified_sampler import StratifiedTaskSampler, GROUPS_PER_STRATUM


def priority_snapshot(metrics_path: Path, *, last_step: int, window: int = 30) -> dict:
    if last_step < 1 or window < 1:
        raise ValueError("invalid historical window")
    rows = []
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("record_type") == "metrics" and row.get("phase") == "train" and last_step - window < row["step"] <= last_step:
            rows.append(row)
    if not rows or rows[-1]["step"] != last_step:
        raise ValueError("priority snapshot must end at the resumed checkpoint")
    by_cell: dict[str, list[tuple[bool, bool]]] = {}
    for row in rows:
        for cell, scores in row["rewards_by_cell"].items():
            if len(scores) != 4:
                raise ValueError("historical GRPO group is not G=4")
            by_cell.setdefault(cell, []).append((len(set(scores)) > 1,
                                                   any(score < 0.95 for score in scores)))
    weights = {}
    for cell, samples in by_cell.items():
        variable = sum(int(v) for v, _ in samples) / len(samples)
        nonmax = sum(int(v) for _, v in samples) / len(samples)
        weights[cell] = round(1.0 + 5.0 * variable + 2.0 * nonmax, 6)
    return {"source_metrics": str(metrics_path.resolve()),
            "source_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            "last_step": last_step, "window": window,
            "formula": "1 + 5*p(variable G4 reward) + 2*p(any reward below safe-success floor 0.95)",
            "unseen_weight": 2.0, "weights": weights}


class InformativeTaskSampler(StratifiedTaskSampler):
    def __init__(self, dataframe, snapshot: dict, *, seed: int = 267):
        super().__init__(dataframe, seed=seed)
        if snapshot.get("last_step", 0) < 1 or not isinstance(snapshot.get("weights"), dict):
            raise ValueError("invalid priority snapshot")
        self.weights = snapshot["weights"]
        self.unseen_weight = float(snapshot["unseen_weight"])
        if not 0 < self.unseen_weight <= 8:
            raise ValueError("unseen-cell weight outside bound")
        self.index_to_cell = {int(index): row["cell_id"] for index, row in enumerate(dataframe)}
        if any(not isinstance(weight, (int, float)) or not 1 <= weight <= 8
               for weight in self.weights.values()):
            raise ValueError("priority weight outside bound")
        if not set(self.weights).issubset(set(self.index_to_cell.values())):
            raise ValueError("priority snapshot contains non-train cells")

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        chosen = []
        for stratum in sorted(self.pools):
            pool = self.pools[stratum]
            # Two systematic coverage slots per stratum; this cannot be
            # displaced by the priority score or by prior model outcomes.
            start = epoch * 2
            uniform = []
            for position in range(start, start + 2):
                cycle, offset = divmod(position, len(pool))
                order = list(range(len(pool)))
                random.Random(f"{self.seed}:{stratum}:uniform:{cycle}").shuffle(order)
                candidate = pool[order[offset]][0]
                if candidate in uniform:
                    candidate = next(item[0] for item in pool if item[0] not in uniform)
                uniform.append(candidate)
            remaining = [item[0] for item in pool if item[0] not in uniform]
            rng = random.Random(f"{self.seed}:{stratum}:priority:{epoch}")
            priority = []
            for _ in range(GROUPS_PER_STRATUM - len(uniform)):
                weights = [self.weights.get(self.index_to_cell[index], self.unseen_weight)
                           for index in remaining]
                selected = rng.choices(remaining, weights=weights, k=1)[0]
                priority.append(selected)
                remaining.remove(selected)
            chosen.extend(uniform + priority)
        random.Random(f"{self.seed}:batch:{epoch}").shuffle(chosen)
        if len(chosen) != 32 or len(set(chosen)) != 32:
            raise ValueError("v4 sampler produced duplicate or missing task cells")
        return iter(chosen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--last-step", type=int, required=True)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = priority_snapshot(args.metrics, last_step=args.last_step, window=args.window)
    args.output.write_text(json.dumps(snapshot, sort_keys=True, indent=2) + "\n")
    print(f"Wrote {len(snapshot['weights'])} train-cell priority weights to {args.output}")
