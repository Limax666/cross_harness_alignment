#!/usr/bin/env python3
"""Plot training metrics from a VeRL SFT console log."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIELD = re.compile(r"([\w/()]+):([-+\d.eE]+)")
STEP = re.compile(r"\bstep:(\d+)\s+-\s+")


def read_metrics(path: Path):
    train, validation = {}, {}
    for line in path.read_text(errors="replace").replace("\r", "\n").splitlines():
        match = STEP.search(line)
        if match is None:
            continue
        step = int(match.group(1))
        fields = {key: float(value) for key, value in FIELD.findall(line[match.end() :])}
        if "train/loss" in fields:
            train[step] = fields
        if "val/loss" in fields:
            validation[step] = fields
    if not train or not validation:
        raise ValueError("The log must contain training and validation step records")
    return train, validation


def plot(train, validation, output: Path, epoch_steps: int):
    steps = sorted(train)
    val_steps = sorted(validation)
    best = min(val_steps, key=lambda step: validation[step]["val/loss"])

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    figure.suptitle(
        "Qwen3.5-2B-Base full-parameter SFT | 267 updates, 3 epochs",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(steps, [train[s]["train/loss"] for s in steps], color="#b7c1c9", lw=1, alpha=0.75, label="Per-step loss")
    ax.plot(steps, [train[s]["train/loss_ma32"] for s in steps], color="#1167a4", lw=2.5, label="32-step moving average")
    ax.set(title="Training loss", ylabel="Assistant-token loss")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(val_steps, [validation[s]["val/loss"] for s in val_steps], "o-", ms=3.5, lw=2, color="#d27a22")
    ax.scatter(best, validation[best]["val/loss"], color="#a83732", s=55, zorder=3)
    ax.annotate(
        f"Best: {validation[best]['val/loss']:.3f} (step {best})",
        (best, validation[best]["val/loss"]),
        xytext=(8, 16),
        textcoords="offset points",
        color="#a83732",
    )
    ax.set(title="Held-out validation loss", ylabel="Loss")

    ax = axes[1, 0]
    ax.plot(steps, [train[s]["train/lr"] * 1e6 for s in steps], color="#258c65", lw=2)
    ax.set(title="Learning-rate schedule", ylabel="Learning rate (×10⁻⁶)")

    ax = axes[1, 1]
    ax.plot(steps, [train[s]["train/grad_norm"] for s in steps], color="#7860a4", lw=1.5)
    ax.set(title="Gradient norm", ylabel="Norm")

    for ax in axes.flat:
        for boundary in range(epoch_steps, max(steps), epoch_steps):
            ax.axvline(boundary + 0.5, color="#6e7781", ls="--", lw=0.9, alpha=0.6)
        ax.grid(color="#e8edf0", lw=0.7)
        ax.set(xlabel="Optimizer step", xlim=(1, max(steps)))

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_csv(train, validation, output: Path):
    fieldnames = ["step", "train_loss", "train_loss_ma32", "validation_loss", "learning_rate", "grad_norm", "global_tokens", "max_memory_allocated_gb", "max_memory_reserved_gb"]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for step, metrics in sorted(train.items()):
            writer.writerow({
                "step": step,
                "train_loss": metrics.get("train/loss"),
                "train_loss_ma32": metrics.get("train/loss_ma32"),
                "validation_loss": validation.get(step, {}).get("val/loss"),
                "learning_rate": metrics.get("train/lr"),
                "grad_norm": metrics.get("train/grad_norm"),
                "global_tokens": metrics.get("train/global_tokens"),
                "max_memory_allocated_gb": metrics.get("perf/max_memory_allocated_gb"),
                "max_memory_reserved_gb": metrics.get("perf/max_memory_reserved_gb"),
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path, help="PNG output path; PDF and CSV use the same basename")
    parser.add_argument("--epoch-steps", type=int, default=89)
    args = parser.parse_args()
    train, validation = read_metrics(args.log)
    plot(train, validation, args.output, args.epoch_steps)
    write_csv(train, validation, args.output.with_suffix(".csv"))
    best = min(validation, key=lambda step: validation[step]["val/loss"])
    print(f"train_steps={len(train)} validation_points={len(validation)} best_step={best} best_validation_loss={validation[best]['val/loss']:.6f}")
    print(args.output)


if __name__ == "__main__":
    main()
