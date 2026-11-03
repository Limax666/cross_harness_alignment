#!/usr/bin/env python3
"""Plot the actual checkpoint lineage through CHS-PO optimizer step 120.

The five source runs have overlapping step numbers. Only the segments that
are ancestors of the step-120 checkpoint are admitted here. Step 0 is the SFT
initialization and has no online reward or actor-entropy observation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/rl"
OUT = ROOT / "reports/figures"
SEGMENTS = (
    ("r20b", 1, 10, RUNS / "multiharness_chspo_r20b_ramfix_gpu12_20261010.log"),
    ("r20e", 11, 20, RUNS / "multiharness_chspo_r20e_audit_retry_gpu12_20261010/launch.log"),
    ("r20f", 21, 60, RUNS / "multiharness_chspo_r20f_dnsretry_gpu12_20261010/launch.log"),
    ("r20g", 61, 70, RUNS / "multiharness_chspo_r20g_zero_signal_resample_gpu12_20261011/launch.log"),
    ("v3", 71, 120, RUNS / "multiharness_chspo_r20g_step70_v3_resume_20261011/launch.log"),
)
STEP_RE = re.compile(r"\bstep:(\d+)\s+-\s+actor/entropy:([\d.eE+-]+)")
REWARD_RE = re.compile(r"\bcritic/rewards/mean:([\d.eE+-]+)")


def main() -> None:
    rows = {}
    sources = []
    for run, first, last, path in SEGMENTS:
        raw = path.read_bytes()
        sources.append({"run": run, "first_step": first, "last_step": last,
                        "path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(raw).hexdigest()})
        found = {}
        for line in raw.decode("utf-8", errors="replace").splitlines():
            step_match = STEP_RE.search(line)
            if not step_match:
                continue
            step = int(step_match.group(1))
            if not first <= step <= last:
                continue
            reward_match = REWARD_RE.search(line)
            if reward_match is None:
                raise ValueError(f"Missing reward in {run} step {step}")
            if step in found:
                raise ValueError(f"Duplicate step {step} in {run}")
            found[step] = {"step": step, "run": run,
                           "reward": float(reward_match.group(1)),
                           "entropy": float(step_match.group(2))}
        missing = set(range(first, last + 1)) - found.keys()
        if missing:
            raise ValueError(f"Missing {run} steps: {sorted(missing)}")
        rows.update(found)
    if set(rows) != set(range(1, 121)):
        raise ValueError("Step-120 lineage is not complete")

    OUT.mkdir(parents=True, exist_ok=True)
    data = [rows[step] for step in range(1, 121)]
    with (OUT / "chspo_step0_120_reward_entropy.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("step", "run", "reward", "entropy"))
        writer.writeheader()
        writer.writerows(data)
    (OUT / "chspo_step0_120_sources.json").write_text(
        json.dumps({"step_0": "SFT checkpoint; no online reward or actor entropy measured",
                    "reward_definition": "VeRL critic/rewards/mean, mean over 128 rollout trajectories per update",
                    "entropy_definition": "VeRL actor/entropy from the same optimizer update",
                    "objective_change_after_step": 70,
                    "sources": sources}, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    for field, title, ylabel, color in (
        ("reward", "CHS-PO online reward | steps 0–120", "Mean trajectory reward", "#156c9e"),
        ("entropy", "CHS-PO policy entropy | steps 0–120", "Actor token entropy", "#a35c16"),
    ):
        fig, ax = plt.subplots(figsize=(10.5, 5.4))
        xs = [row["step"] for row in data]
        ys = [row[field] for row in data]
        ax.plot(xs, ys, color=color, lw=1.65, marker="o", ms=2.15, alpha=.9)
        ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], s=35, color=color, zorder=4)
        for boundary in (10, 20, 60):
            ax.axvline(boundary + .5, color="#aab4bc", lw=.8, ls=":", zorder=0)
        ax.axvline(70.5, color="#a43c4a", lw=1.2, ls="--", zorder=0)
        ax.text(72, .96, "v3 objective from step 71", transform=ax.get_xaxis_transform(),
                color="#a43c4a", va="top", fontsize=9)
        ax.set(xlim=(0, 120.5), xlabel="Optimizer step", ylabel=ylabel, title=title)
        ax.set_xticks(range(0, 121, 10))
        ax.grid(color="#dce3e8", alpha=.8, lw=.7)
        ax.set_axisbelow(True)
        ax.annotate(f"step 1: {ys[0]:.3f}", (1, ys[0]), xytext=(8, 15),
                    textcoords="offset points", fontsize=9, color=color)
        ax.annotate(f"step 120: {ys[-1]:.3f}", (120, ys[-1]), xytext=(-8, 14),
                    textcoords="offset points", ha="right", fontsize=9, color=color)
        fig.text(.12, .025, "Step 0 = SFT initialization (no rollout metric). Dotted lines = checkpoint resumes; dashed line = objective change.",
                 fontsize=8.4, color="#53636e")
        fig.subplots_adjust(left=.105, right=.965, top=.88, bottom=.17)
        for suffix in ("png", "pdf"):
            fig.savefig(OUT / f"chspo_step0_120_{field}.{suffix}", dpi=220)
        plt.close(fig)
    print(json.dumps({"steps": len(data), "range": [1, 120],
                      "reward_first_last": [data[0]["reward"], data[-1]["reward"]],
                      "entropy_first_last": [data[0]["entropy"], data[-1]["entropy"]],
                      "output_dir": str(OUT)}))


if __name__ == "__main__":
    main()
