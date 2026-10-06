#!/usr/bin/env python3
"""Compare completed HarnessRisk GPT-5.4 judge summaries with paper Table 2."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# HarnessRisk, arXiv:2608.17597v1, Table 2. Percent values.
PAPER = {
    "openclaw": {
        "GPT-5.5": (92.65, 75.59, 20.63, 74.02),
        "Claude Opus 4.7": (75.00, 47.71, 17.25, 78.94),
        "DeepSeek-V4-Pro": (94.50, 54.00, 16.90, 76.50),
        "GLM-5.2": (95.30, 54.70, 18.00, 92.20),
        "Kimi K2.6": (97.10, 80.87, 20.50, 43.20),
        "MiniMax M3": (94.30, 31.20, 10.80, 97.90),
    },
    "hermes": {
        "DeepSeek-V4-Pro": (97.60, 65.40, 20.50, 34.60),
        "GLM-5.2": (96.80, 23.80, 4.00, 61.90),
        "Kimi K2.6": (93.80, 65.60, 15.60, 11.70),
        "MiniMax M3": (96.10, 14.80, 5.50, 85.20),
    },
}
METRICS = ("utility_score", "attack_success_rate", "persistence_score", "detection_score")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes", type=Path, required=True, help="Hermes llm_judge_multi_harness/summary.json")
    parser.add_argument("--openclaw", type=Path, required=True, help="OpenClaw llm_judge_multi_harness/summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for harness, path in (("hermes", args.hermes), ("openclaw", args.openclaw)):
        summary = json.loads(path.read_text())
        count = summary.get("count", 0)
        if count != 128:
            raise SystemExit(f"{harness}: only {count}/128 judged cases; refusing full-benchmark comparison")
        metrics = summary["metrics"]
        rows.append({"harness": harness, "model": "Qwen3.5-9B SFT", "source": "local", "n": count,
                     **{key: 100 * metrics[key] for key in METRICS}})
        for name, values in PAPER[harness].items():
            rows.append({"harness": harness, "model": name, "source": "paper", "n": 384,
                         **dict(zip(METRICS, values))})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for ax, harness in zip(axes, ("hermes", "openclaw")):
        for row in (r for r in rows if r["harness"] == harness):
            local = row["source"] == "local"
            ax.scatter(row["utility_score"], row["attack_success_rate"],
                       c="#d62728" if local else "#1f77b4", s=110 if local else 50,
                       marker="*" if local else "o")
            ax.annotate(row["model"], (row["utility_score"], row["attack_success_rate"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_title(harness.capitalize())
        ax.set_xlabel("Utility (%) ↑")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Attack success rate (%) ↓")
    fig.suptitle("HarnessRisk: Qwen3.5-9B SFT vs paper Table 2 (GPT-5.4 judge)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "utility_vs_asr.png", dpi=180)


if __name__ == "__main__":
    main()
