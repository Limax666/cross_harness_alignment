#!/usr/bin/env python3
"""Stop a named CHS-PO tmux session on a severe policy-health regression."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path


def inspect(row: dict) -> list[str]:
    if row.get("record_type") != "metrics" or row.get("phase") != "train":
        return []
    optimizer = row.get("optimizer", {})
    reasons = []
    kl = optimizer.get("actor/ppo_kl", optimizer.get("approx_kl"))
    entropy = optimizer.get("actor/entropy")
    mean_length = optimizer.get("mean_response_tokens")
    mean_probability = optimizer.get("mean_token_probability")
    for name, value in (("ppo_kl", kl), ("entropy", entropy),
                        ("mean_response_tokens", mean_length),
                        ("mean_token_probability", mean_probability)):
        if value is not None and not math.isfinite(float(value)):
            reasons.append(f"nonfinite_{name}")
    if kl is not None and float(kl) > 0.05:
        reasons.append(f"ppo_kl={float(kl):.6g}>0.05")
    if entropy is not None and float(entropy) > 3.0:
        reasons.append(f"entropy={float(entropy):.6g}>3.0")
    if (mean_length is not None and mean_probability is not None
            and float(mean_length) > 800 and float(mean_probability) < 0.01):
        reasons.append("long_low_probability_outputs")

    arms = [arm for harness in row.get("harness_metrics", {}).values()
            for arm in harness.values()]
    if arms:
        utility = sum(float(arm.get("utility", 0)) for arm in arms) / len(arms)
        timeout = sum(float(arm.get("timeout_rate", 0)) for arm in arms) / len(arms)
        if utility < 0.25 and timeout > 0.75:
            reasons.append(f"utility={utility:.3f}<0.25_and_timeout={timeout:.3f}>0.75")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    seen = 0
    while True:
        if not args.metrics.exists():
            time.sleep(args.poll_seconds)
            continue
        lines = args.metrics.read_text().splitlines()
        for line in lines[seen:]:
            if not line.strip():
                continue
            row = json.loads(line)
            reasons = inspect(row)
            seen += 1
            if reasons:
                alert = {"step": row.get("step"), "reasons": reasons, "row": row}
                (args.run_dir / "health_stop.json").write_text(
                    json.dumps(alert, indent=2, sort_keys=True) + "\n")
                subprocess.run(["tmux", "kill-session", "-t", args.session], check=False)
                return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
