#!/usr/bin/env python3
"""Render the project's online RL training dashboard from its JSONL journal."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_journal(path: Path) -> tuple[dict, list[dict]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or rows[0].get("record_type") != "run":
        raise ValueError("metrics journal must begin with an immutable run header")
    metadata = rows[0]["metadata"]
    if metadata.get("rollout_mode") != "online_current_policy":
        raise ValueError("refusing to label offline/replay data as online RL curves")
    metrics = [row for row in rows[1:] if row.get("record_type") == "metrics"]
    return metadata, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("journal", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window", type=int, default=5, help="rolling median window; raw observations remain visible")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be >= 1")
    metadata, rows = load_journal(args.journal)
    if not rows:
        raise SystemExit("no online update metrics in journal yet")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output = args.output or args.journal.with_name("rl_training_dashboard.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10")
    harnesses = list(metadata["harnesses"])
    by_phase = {phase: [row for row in rows if row["phase"] == phase] for phase in ("train", "validation")}
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), constrained_layout=True)
    fig.suptitle(
        f"Online RL training · {metadata['algorithm']} · {metadata['run_id']}\n"
        f"Policy: {metadata['model_checkpoint']} · validation rows are separate markers",
        fontsize=13,
    )

    def series(phase: str, harness: str | None, arm: str | None, dotted_key: tuple[str, ...]):
        selected = by_phase[phase]
        xs, ys = [], []
        for row in selected:
            node = row
            for key in dotted_key:
                if not isinstance(node, dict) or key not in node:
                    node = None
                    break
                node = node[key]
            if node is None or isinstance(node, bool) or not isinstance(node, (float, int)):
                continue
            xs.append(row["step"])
            ys.append(float(node))
        return xs, ys

    def plot_metric(ax, key: tuple[str, ...], title: str, *, section: str = "harness_metrics",
                    arms: tuple[str, ...] = (), ylim: tuple[float, float] | None = None):
        for index, harness in enumerate(harnesses):
            for arm_index, arm in enumerate(arms or (None,)):
                path = (section, harness) + ((arm,) if arm else ()) + key
                label = harness + (f"/{arm}" if arm else "")
                color = colors(index)
                for phase, marker, alpha in (("train", None, 0.25), ("validation", "o", 0.95)):
                    x, y = series(phase, harness, arm, path)
                    if not x:
                        continue
                    style = "-" if arm_index == 0 else "--"
                    if phase == "train":
                        ax.scatter(x, y, s=9, color=color, alpha=alpha)
                        if len(y) >= args.window:
                            rolling = np.array([np.median(y[max(0, i - args.window + 1):i + 1]) for i in range(len(y))])
                            ax.plot(x, rolling, color=color, linestyle=style, label=label)
                        elif y:
                            ax.plot(x, y, color=color, linestyle=style, label=label)
                    else:
                        ax.plot(x, y, color=color, marker=marker, linestyle="none", label=f"{label} val")
        ax.set_title(title)
        ax.set_xlabel("Optimizer update")
        ax.grid(True, alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)
        if ax.lines or ax.collections:
            ax.legend(fontsize=7, ncol=2)

    plot_metric(axes[0, 0], ("mean_reward",), "Verified reward by task arm", arms=("benign", "risk"), ylim=(-1.1, 1.1))
    plot_metric(axes[0, 1], ("utility",), "Official utility", arms=("benign", "risk"), ylim=(-0.05, 1.05))
    plot_metric(axes[0, 2], ("attack_success_rate",), "Official injection ASR (lower is safer)", arms=("risk",), ylim=(-0.05, 1.05))
    plot_metric(axes[1, 0], ("persistence_rate",), "Audited prohibited persistence", arms=("risk",), ylim=(-0.05, 1.05))
    plot_metric(axes[1, 1], ("contract_violation_rate",), "Tool contract violations", arms=("benign", "risk"), ylim=(-0.05, 1.05))
    plot_metric(axes[1, 2], ("timeout_rate",), "Model timeout / unfinished turn", arms=("benign", "risk"), ylim=(-0.05, 1.05))

    def global_metric(ax, key: str, label: str, color: str, phase: str = "train", style: str = "-"):
        x, y = series(phase, None, None, ("optimizer", key))
        if x:
            ax.scatter(x, y, color=color, s=10, alpha=0.25)
            if len(y) >= args.window:
                smooth = [sum(y[max(0, i - args.window + 1):i + 1]) / len(y[max(0, i - args.window + 1):i + 1]) for i in range(len(y))]
                ax.plot(x, smooth, color=color, label=label, linestyle=style)
            else:
                ax.plot(x, y, color=color, label=label, linestyle=style)

    ax = axes[2, 0]
    global_metric(ax, "actor/loss", "actor loss", "tab:blue")
    global_metric(ax, "grad_norm", "gradient norm", "tab:orange", style="--")
    ax.set_title("Policy loss and gradient norm")
    ax.set_xlabel("Optimizer update"); ax.grid(True, alpha=0.25)
    if ax.lines: ax.legend(fontsize=8)

    ax = axes[2, 1]
    global_metric(ax, "approx_kl", "approx KL", "tab:purple")
    global_metric(ax, "clip_fraction", "clip fraction", "tab:brown", style="--")
    ax.set_title("KL and clipping")
    ax.set_xlabel("Optimizer update"); ax.grid(True, alpha=0.25)
    if ax.lines: ax.legend(fontsize=8)

    ax = axes[2, 2]
    global_metric(ax, "mean_response_tokens", "mean assistant tokens", "tab:green")
    global_metric(ax, "p95_response_tokens", "p95 assistant tokens", "tab:red", style="--")
    ax.set_title("Assistant response length")
    ax.set_xlabel("Optimizer update"); ax.grid(True, alpha=0.25)
    if ax.lines: ax.legend(fontsize=8)

    temporary = output.with_suffix(output.suffix + ".tmp")
    fig.savefig(temporary, format="png", dpi=140)
    plt.close(fig)
    temporary.replace(output)
    print(json.dumps({"output": str(output), "run_id": metadata["run_id"], "metric_rows": len(rows),
                      "train_rows": len(by_phase["train"]), "validation_rows": len(by_phase["validation"])}))


if __name__ == "__main__":
    main()
