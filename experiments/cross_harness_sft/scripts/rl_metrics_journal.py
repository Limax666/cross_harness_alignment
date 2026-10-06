"""Append compact, auditable metrics from *online* PPO/GRPO updates.

The journal is intentionally separate from rollout traces. Keep full episode
evidence in the run's audit store and append one small summary row per update.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOP_LEVEL = {
    "step", "phase", "sampled_groups", "sampled_rollouts", "scored_rollouts",
    "unscorable_rollouts", "optimizer", "harness_metrics",
}


def _finite_tree(value: Any, path: str = "metrics") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} must be finite")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")
        return
    raise TypeError(f"unsupported metric type at {path}: {type(value).__name__}")


class RLMetricsJournal:
    """Write a run header once and one compact metric record per update/eval."""

    def __init__(self, path: str | Path, metadata: dict[str, Any]):
        required = {
            "run_id", "algorithm", "model_checkpoint", "reference_checkpoint",
            "reward_version", "verifier_version", "harnesses", "train_split_hash",
        }
        missing = sorted(required - metadata.keys())
        if missing:
            raise ValueError(f"missing immutable run metadata: {', '.join(missing)}")
        if metadata.get("rollout_mode") != "online_current_policy":
            raise ValueError("RL metrics require rollout_mode=online_current_policy")
        _finite_tree(metadata)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(metadata["run_id"])
        self._harnesses = set(metadata["harnesses"])
        if self.path.exists() and self.path.stat().st_size:
            first = json.loads(self.path.open(encoding="utf-8").readline())
            if first.get("record_type") != "run" or first.get("metadata") != metadata:
                raise ValueError("refusing to append to a journal with different run metadata")
        else:
            self._append({
                "record_type": "run", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata,
            })
        self._last_step: dict[str, int] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("record_type") == "metrics":
                        phase = row["phase"]
                        self._last_step[phase] = max(self._last_step.get(phase, 0), int(row["step"]))

    def _append(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            output.flush()

    def log(self, metrics: dict[str, Any]) -> None:
        missing = sorted(TOP_LEVEL - metrics.keys())
        if missing:
            raise ValueError(f"missing update fields: {', '.join(missing)}")
        phase = metrics["phase"]
        if phase not in {"train", "validation"}:
            raise ValueError("phase must be train or validation")
        step = metrics["step"]
        if not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if step < self._last_step.get(phase, 0):
            raise ValueError(f"{phase} step moved backwards")
        for key in ("sampled_groups", "sampled_rollouts", "scored_rollouts", "unscorable_rollouts"):
            if not isinstance(metrics[key], int) or metrics[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if metrics["scored_rollouts"] + metrics["unscorable_rollouts"] > metrics["sampled_rollouts"]:
            raise ValueError("scored + unscorable rollouts exceeds sampled rollouts")
        if not isinstance(metrics["optimizer"], dict) or not isinstance(metrics["harness_metrics"], dict):
            raise ValueError("optimizer and harness_metrics must be objects")
        if set(metrics["harness_metrics"]) - self._harnesses:
            raise ValueError("metric row includes a harness absent from immutable run metadata")
        _finite_tree(metrics)
        row = {
            "record_type": "metrics", "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), **metrics,
        }
        self._append(row)
        self._last_step[phase] = step
