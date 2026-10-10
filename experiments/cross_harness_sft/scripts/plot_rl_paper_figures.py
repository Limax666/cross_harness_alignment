#!/usr/bin/env python3
"""Export lossless per-update CSV and separate paper-ready RL metric figures.

Each plotted marker is one recorded optimizer update. No aggregation or
smoothing is applied. PNG, PDF, and SVG versions are written per figure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A writer may be in the middle of appending its final line.
            if index == len(lines) - 1:
                break
            raise
    return rows


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(flatten(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(flatten(child, f"{prefix}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            result[prefix] = value
    return result


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    journal = run_dir / "metrics.jsonl"
    safety_journal = run_dir / "chspo_safety.jsonl"
    rows = read_jsonl(journal)
    header = next((row for row in rows if row.get("record_type") == "run"), None)
    updates = [row for row in rows if row.get("record_type") == "metrics" and row.get("phase") == "train"]
    safety = read_jsonl(safety_journal)

    flattened: list[dict[str, Any]] = []
    for row in updates:
        item: dict[str, Any] = {key: row.get(key) for key in ("run_id", "phase", "step", "timestamp_utc", "sampled_groups", "sampled_rollouts", "scored_rollouts", "unscorable_rollouts")}
        item.update(flatten(row.get("optimizer", {}), "optimizer"))
        item.update(flatten(row.get("harness_metrics", {}), "harness_metrics"))
        item.update(flatten(row.get("rewards_by_cell", {}), "rewards_by_cell"))
        item.update(flatten({key: value for key, value in row.items()
                             if key not in {"record_type", "run_id", "phase", "step", "timestamp_utc",
                                            "optimizer", "harness_metrics", "rewards_by_cell"}}, "update"))
        flattened.append(item)
    atomic_csv(run_dir / "paper_metrics_per_step.csv", flattened)

    if not updates and not safety:
        print(json.dumps({"status": "waiting_for_first_update", "run_dir": str(run_dir)}))
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = run_dir / "paper_figures"
    out.mkdir(parents=True, exist_ok=True)
    harnesses = (header or {}).get("metadata", {}).get("harnesses", [])
    colors = plt.get_cmap("tab10")
    figure_manifest = []

    def save_figure(fig, name: str, ylabel: str, plotted_steps: list[int]) -> None:
        fig.suptitle(f"CHS-PO online training — {name.replace('_', ' ')}", fontsize=14)
        fig.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            target = out / f"{name}.{suffix}"
            temp = target.with_suffix(f".{suffix}.tmp")
            fig.savefig(temp, format=suffix, dpi=300 if suffix == "png" else None,
                        bbox_inches="tight")
            temp.replace(target)
        plt.close(fig)
        figure_manifest.append({"name": name, "ylabel": ylabel, "steps": plotted_steps,
                                "points": len(plotted_steps), "formats": ["png", "pdf", "svg"]})

    metric_specs = [
        ("mean_reward", "mean_reward", "Mean verified reward", "Reward"),
        ("utility", "utility", "Official utility", "Utility rate"),
        ("attack_success_rate", "attack_success_rate", "Official attack success rate", "ASR"),
        ("persistence_rate", "persistence_rate", "Verified prohibited persistence", "Persistence rate"),
        ("contract_violation_rate", "contract_violation_rate", "Tool contract violation rate", "Violation rate"),
        ("timeout_rate", "timeout_rate", "Timeout / unfinished-turn rate", "Rate"),
        ("mean_path_signal", "mean_path_signal", "Mean audited path-shaped reward signal", "Path signal"),
        ("nonengagement_rate", "nonengagement_rate", "Benign non-engagement rate", "Rate"),
    ]
    for field, name, title, ylabel in metric_specs:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        points = 0
        for hi, harness in enumerate(harnesses):
            for ai, arm in enumerate(("benign", "risk")):
                xs, ys = [], []
                for row in updates:
                    value = row.get("harness_metrics", {}).get(harness, {}).get(arm, {}).get(field)
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        xs.append(row["step"]); ys.append(float(value))
                if xs:
                    ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.1,
                            linestyle="-" if arm == "benign" else "--",
                            color=colors(hi), label=f"{harness} / {arm}")
                    points += len(xs)
        ax.set_title(title); ax.set_xlabel("Optimizer step"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=.28)
        if points: ax.legend(fontsize=8, ncol=2)
        save_figure(fig, name, ylabel, [r["step"] for r in updates])

    optimizer_specs = [
        ("actor/loss", "actor_loss", "Actor loss", "Loss"),
        ("grad_norm", "gradient_norm", "Gradient norm", "Norm"),
        ("approx_kl", "approx_kl", "Approximate KL", "KL"),
        ("clip_fraction", "clip_fraction", "Policy clip fraction", "Fraction"),
        ("mean_response_tokens", "mean_response_tokens", "Mean assistant response length", "Tokens"),
        ("p95_response_tokens", "p95_response_tokens", "95th percentile assistant response length", "Tokens"),
        ("zero_variance_group_rate", "zero_variance_group_rate", "Zero-variance group rate", "Rate"),
        ("mean_token_probability", "mean_token_probability", "Mean rollout token probability", "Probability"),
    ]
    for key, name, title, ylabel in optimizer_specs:
        xs, ys = [], []
        for row in updates:
            value = row.get("optimizer", {}).get(key)
            if value is None and key == "zero_variance_group_rate":
                value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                xs.append(row["step"]); ys.append(float(value))
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(xs, ys, color="tab:blue", marker="o", markersize=3.5, linewidth=1.1)
        ax.set_title(title); ax.set_xlabel("Optimizer step"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=.28)
        save_figure(fig, name, ylabel, xs)

    # CHS-PO's benchmark/harness-specific duals and observed safety costs.
    safety_series: dict[tuple[str, str, str], list[tuple[int, float]]] = {}
    for row in safety:
        step = row.get("step")
        if not isinstance(step, int):
            continue
        for section in ("train_cost_rates", "duals"):
            for stratum, values in row.get(section, {}).items():
                for metric, value in values.items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        key = (section, stratum, metric)
                        safety_series.setdefault(key, []).append((step, float(value)))
    for section, title, ylabel in (("duals", "CHS-PO safety duals", "Dual value"),
                                   ("train_cost_rates", "CHS-PO observed safety costs", "Observed rate")):
        fig, ax = plt.subplots(figsize=(7.2, 4.6)); points = 0
        for (series_section, stratum, metric), values in safety_series.items():
            if series_section != section: continue
            values.sort()
            ax.plot([x for x, _ in values], [y for _, y in values], marker="o", markersize=3.5,
                    linewidth=1.1, label=f"{stratum}/{metric}")
            points += len(values)
        if points:
            ax.set_title(title); ax.set_xlabel("Optimizer step"); ax.set_ylabel(ylabel)
            ax.grid(True, alpha=.28); ax.legend(fontsize=7, ncol=2)
            save_figure(fig, f"chspo_{section}", ylabel, sorted({r[0] for vals in safety_series.values() for r in vals}))
        else:
            plt.close(fig)

    manifest = {
        "run_id": (header or {}).get("metadata", {}).get("run_id"),
        "source_metrics_sha256": hashlib.sha256(journal.read_bytes()).hexdigest() if journal.exists() else None,
        "source_safety_sha256": hashlib.sha256(safety_journal.read_bytes()).hexdigest() if safety_journal.exists() else None,
        "training_updates": len(updates), "raw_step_points": True, "smoothing_applied": False,
        "csv": "paper_metrics_per_step.csv", "figures": figure_manifest,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "rendered", "updates": len(updates), "figures": len(figure_manifest),
                      "csv": str(run_dir / "paper_metrics_per_step.csv"), "figure_dir": str(out)}))


if __name__ == "__main__":
    main()
