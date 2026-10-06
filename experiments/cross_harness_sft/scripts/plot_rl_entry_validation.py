#!/usr/bin/env python3
"""Plot genuine disjoint-family Base/SFT RL-entry evaluation, NEVER RL training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(directory: Path) -> dict:
    result = {"kind": "family_disjoint_rl_entry_validation_not_rl_training", "source": {}, "arms": {}}
    for role in ("base", "sft"):
        path = directory / f"{role}_v3.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if len(rows) != 12 or any(row.get("status") != "completed" for row in rows):
            raise ValueError(f"incomplete validation result: {path}")
        if len({(row["task_id"], row["seed"]) for row in rows}) != 12:
            raise ValueError(f"duplicate or missing validation cell/seed: {path}")
        result["source"][role] = str(path)
        for arm in ("clean", "injection_task_1"):
            group = [row for row in rows if row["task_id"].endswith(":" + arm)]
            if len(group) != 6 or {row["family"] for row in group} != {"workspace:user_task_3", "workspace:user_task_17"}:
                raise ValueError(f"wrong held-out validation group: {role}/{arm}")
            result["arms"][f"{role}/{arm}"] = {
                "n": len(group),
                "official_utility_successes": sum(row["official_utility"] == 1 for row in group),
                "official_attack_successes": sum(row["official_attack_success"] is True for row in group),
                "episodes_with_invalid_calls": sum(row["invalid_calls"] > 0 for row in group),
                "episodes_with_native_tool_execution": sum(row["tools_executed"] > 0 for row in group),
                "generation_truncations": sum(row["termination"] == "generation_token_limit" for row in group),
            }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    data = summarize(args.directory)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = args.output / "rl_entry_validation_v3_summary.json"
    if summary.exists():
        raise FileExistsError(summary)
    summary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Base clean", "SFT clean", "Base injection", "SFT injection"]
    keys = ["base/clean", "sft/clean", "base/injection_task_1", "sft/injection_task_1"]
    metrics = [("official_utility_successes", "Official utility"),
               ("episodes_with_native_tool_execution", "Native tool executed"),
               ("episodes_with_invalid_calls", "Invalid call episodes"),
               ("official_attack_successes", "Official attack success")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = ["#405e84", "#58a181", "#405e84", "#58a181"]
    for ax, (field, title) in zip(axes.ravel(), metrics):
        values = [data["arms"][key][field] for key in keys]
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.set_ylim(0, 6.8)
        ax.set_ylabel("episodes / 6")
        ax.tick_params(axis="x", labelrotation=17)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + .12, str(value), ha="center")
    fig.suptitle("Family-disjoint RL entry validation (not RL training); workspace tasks 3, 17")
    chart = args.output / "rl_entry_validation_v3.png"
    if chart.exists():
        raise FileExistsError(chart)
    fig.savefig(chart, dpi=160)
    print(json.dumps({"summary": str(summary), "plot": str(chart), "kind": data["kind"]}))


if __name__ == "__main__":
    main()
