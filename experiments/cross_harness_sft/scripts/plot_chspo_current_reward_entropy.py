#!/usr/bin/env python3
"""Plot measured reward and entropy along the retained v4 checkpoint lineage."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_chspo_step120_reward_entropy import ROOT, OUT, RUNS, SEGMENTS, STEP_RE, REWARD_RE


def main():
    latest = RUNS / "multiharness_chspo_r20g_v4_from130_20261012"
    committed = [json.loads(line) for line in (latest / "metrics.jsonl").read_text().splitlines()
                 if line.strip()]
    steps = {int(row["step"]) for row in committed
             if row.get("record_type") == "metrics" and row.get("phase") == "train"}
    end = max(steps)
    segments = (*SEGMENTS,
                ("v3_resume", 121, 130, RUNS / "multiharness_chspo_r20g_step120_v3_resume_20261012/launch.log"),
                ("v4", 131, end, latest / "launch.log"))
    rows, sources = {}, []
    for run, first, last, path in segments:
        raw = path.read_bytes()
        sources.append({"run": run, "first_step": first, "last_step": last,
                        "path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(raw).hexdigest()})
        found = {}
        for line in raw.decode("utf-8", errors="replace").splitlines():
            match = STEP_RE.search(line)
            if not match or not first <= int(match[1]) <= last:
                continue
            step = int(match[1])
            reward = REWARD_RE.search(line)
            if reward is None or step in found:
                raise ValueError(f"missing reward or duplicate step: {run}/{step}")
            found[step] = {"step": step, "run": run, "reward": float(reward[1]),
                           "entropy": float(match[2])}
        if set(found) != set(range(first, last + 1)):
            raise ValueError(f"incomplete measured segment: {run}")
        rows.update(found)
    if set(rows) != set(range(1, end + 1)) or steps != set(range(131, end + 1)):
        raise ValueError("checkpoint lineage has missing committed updates")
    data = [rows[step] for step in range(1, end + 1)]
    stem = f"chspo_step0_{end}"
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{stem}_reward_entropy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("step", "run", "reward", "entropy"))
        writer.writeheader()
        writer.writerows(data)
    (OUT / f"{stem}_sources.json").write_text(json.dumps({
        "last_completed_step": end, "step_0": "SFT initialization; unmeasured",
        "excluded_branch": "v3 resume steps 131-138; rolled back to checkpoint 130",
        "objective_change_after_steps": [70, 130], "sources": sources,
    }, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    for field, label, color in (("reward", "Mean trajectory reward", "#156c9e"),
                                ("entropy", "Actor token entropy (nats)", "#a35c16")):
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        xs = [row["step"] for row in data]
        ys = [row[field] for row in data]
        ax.axvspan(130.5, end + .5, color="#d7eee6", alpha=.45)
        ax.plot(xs, ys, color=color, alpha=.35, lw=1.1, label="Per-update value")
        for first, last in ((1, 70), (71, 130), (131, end)):
            values = [rows[step][field] for step in range(first, last + 1)]
            smooth = [sum(values[max(0, i-9):i+1]) / len(values[max(0, i-9):i+1])
                      for i in range(len(values))]
            ax.plot(range(first, last + 1), smooth, color=color, lw=2.2,
                    label="10-update moving average" if first == 1 else None)
        for boundary, text, c in ((70.5, "v3", "#956273"), (130.5, "v4", "#297b60")):
            ax.axvline(boundary, color=c, lw=1, ls="--")
            ax.text(boundary + 1.5, .97, text, transform=ax.get_xaxis_transform(),
                    va="top", color=c)
        ax.scatter([end], [ys[-1]], color=color, s=35, zorder=5)
        ax.annotate(f"Step {end}: {ys[-1]:.3f}", (end, ys[-1]),
                    xytext=(-6, 14), textcoords="offset points", ha="right", fontsize=10)
        ax.set(xlim=(0, end + .5), xlabel="Optimizer step", ylabel=label,
               title=f"CHS-PO training {field} | retained lineage through step {end}")
        ax.grid(color="#dce3e8", alpha=.8, lw=.7)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right" if field == "reward" else "upper left", frameon=False)
        fig.text(.1, .027,
                 "Step 0 is unmeasured. Dashed lines: objective changes. Failed v3 steps 131-138 are excluded.\n"
                 "Reward scales across objective versions are not directly comparable; curves show training metrics.",
                 fontsize=8, color="#53636e")
        fig.subplots_adjust(left=.1, right=.97, top=.9, bottom=.18)
        for extension in ("png", "pdf"):
            fig.savefig(OUT / f"{stem}_{field}.{extension}", dpi=220, facecolor="white")
        plt.close(fig)
    print(json.dumps({"last_step": end, "reward": data[-1]["reward"],
                      "entropy": data[-1]["entropy"], "output_stem": str(OUT / stem)}))


if __name__ == "__main__":
    main()
