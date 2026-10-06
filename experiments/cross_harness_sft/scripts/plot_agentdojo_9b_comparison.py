#!/usr/bin/env python3
"""Create paired AgentDojo comparison metrics, figures and an audit manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score(row: dict, role: str, metric: str) -> float:
    if role == "teacher":
        utility, safety = float(row["teacher_utility"]), float(row["teacher_safety"])
    else:
        reward = row["reward_breakdown"]
        utility, safety = float(reward["utility_reward"]), float(reward["safety_reward"])
    return {"utility": utility, "safety": safety, "joint": utility * safety}[metric]


def ratio(teacher: float, base: float, sft: float) -> float | None:
    denominator = teacher - base
    return None if abs(denominator) < 1e-12 else (sft - base) / denominator


def bootstrap_recovery(rows: list[dict], metric: str, injection_only: bool, samples: int = 2000) -> list[float]:
    selected = [row for row in rows if not injection_only or row["teacher"]["task_type"] == "injection"]
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_family[row["teacher"]["family"]].append(row)
    families = sorted(by_family)
    if len(families) < 2:
        return []
    generator = random.Random(42)
    values: list[float] = []
    for _ in range(samples):
        draw = [item for family in generator.choices(families, k=len(families)) for item in by_family[family]]
        means = {role: mean([score(item[role], role, metric) for item in draw]) for role in ("teacher", "base", "sft")}
        value = ratio(means["teacher"], means["base"], means["sft"])
        if value is not None:
            values.append(value)
    return sorted(values)


def quantile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    return sorted_values[min(len(sorted_values) - 1, int(fraction * len(sorted_values)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir
    plan_path = output_dir / "evaluation_plan.jsonl"
    plan = {row["task_id"]: row for row in read_jsonl(plan_path)}
    if not plan:
        raise ValueError("evaluation plan is missing or empty")
    by_role = {}
    for role in ("base", "sft"):
        by_role[role] = {row["task_id"]: row for row in read_jsonl(output_dir / f"{role}.jsonl") if row.get("status") == "completed"}
    common = sorted(set(plan) & set(by_role["base"]) & set(by_role["sft"]))
    paired = [{"task_id": key, "teacher": plan[key], "base": by_role["base"][key], "sft": by_role["sft"][key]} for key in common]
    with (output_dir / "paired_task_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["task_id", "family", "task_type"] + [f"{role}_{metric}" for role in ("teacher", "base", "sft") for metric in ("utility", "safety", "joint")]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in paired:
            record = {"task_id": item["task_id"], "family": item["teacher"]["family"], "task_type": item["teacher"]["task_type"]}
            record.update({f"{role}_{metric}": score(item[role], role, metric) for role in ("teacher", "base", "sft") for metric in ("utility", "safety", "joint")})
            writer.writerow(record)
    all_metrics: dict[str, dict] = {}
    for metric, injection_only in (("utility", False), ("safety", True), ("joint", False)):
        selected = [row for row in paired if not injection_only or row["teacher"]["task_type"] == "injection"]
        means = {role: mean([score(row[role], role, metric) for row in selected]) for role in ("teacher", "base", "sft")}
        recovery = ratio(means["teacher"], means["base"], means["sft"]) if selected else None
        bootstrap = bootstrap_recovery(paired, metric, injection_only)
        all_metrics[metric] = {"n": len(selected), "mean": means, "recovery": recovery,
                               "recovery_family_bootstrap_95ci": [quantile(bootstrap, .025), quantile(bootstrap, .975)],
                               "bootstrap_defined_draws": len(bootstrap)}
    paired_changes = {}
    for metric in ("utility", "safety", "joint"):
        subset = [row for row in paired if metric != "safety" or row["teacher"]["task_type"] == "injection"]
        counts = Counter("improved" if score(row["sft"], "sft", metric) > score(row["base"], "base", metric)
                         else "regressed" if score(row["sft"], "sft", metric) < score(row["base"], "base", metric)
                         else "tied" for row in subset)
        paired_changes[metric] = {key: counts[key] for key in ("improved", "regressed", "tied")}

    report = {
        "schema": "agentdojo_9b_comparison.report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if len(common) == len(plan) else "provisional",
        "planned_tasks": len(plan), "paired_tasks": len(common),
        "completed_by_role": {role: len(rows) for role, rows in by_role.items()},
        "missing_from_paired": {role: sorted(set(plan) - set(rows)) for role, rows in by_role.items()},
        "metrics": all_metrics, "paired_changes": paired_changes,
        "definition": "capability_recovery=(SFT-base)/(historical_teacher-base) on identical AgentDojo task IDs and seed 0; undefined when denominator is zero",
        "evaluation_scope": "seven SFT validation families; safety is reported on injection tasks only",
        "interpretation_limit": "teacher scores are historical Codex runs; students share a local Qwen tool loop, not Codex. Teacher selection came from the original successful trajectory corpus.",
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    figures: list[str] = []
    if paired and all_metrics["safety"]["n"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        colors = {"teacher": "#756bb1", "base": "#9ecae1", "sft": "#31a354"}
        roles = ("teacher", "base", "sft")
        labels = ("Utility (all)", "Safety (injection)", "Joint success (all)")
        metrics = ("utility", "safety", "joint")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})
        x = np.arange(3)
        for index, role in enumerate(roles):
            values = [all_metrics[metric]["mean"][role] for metric in metrics]
            positions = x + (index - 1) * .24
            bars = axes[0].bar(positions, values, .22, color=colors[role], label=role.upper())
            for bar, value in zip(bars, values):
                if math.isfinite(value):
                    axes[0].text(bar.get_x() + bar.get_width() / 2, value + .025, f"{value:.2f}", ha="center", fontsize=8)
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0, 1.12); axes[0].set_ylabel("Official AgentDojo pass rate")
        axes[0].legend(loc="upper right"); axes[0].grid(axis="y", alpha=.2)
        recoveries = [all_metrics[metric]["recovery"] for metric in metrics]
        for i, value in enumerate(recoveries):
            if value is None:
                axes[1].text(i, 0, "undefined", rotation=90, va="bottom", ha="center")
            else:
                axes[1].bar(i, value, color="#31a354" if value >= 0 else "#de2d26")
                axes[1].text(i, value, f"{value:.1%}", va="bottom" if value >= 0 else "top", ha="center", fontsize=9)
        axes[1].axhline(0, color="black", linewidth=.8)
        axes[1].axhline(1, color="gray", linestyle="--", linewidth=.8)
        axes[1].set_xticks(x, ("Utility", "Safety", "Joint")); axes[1].set_ylabel("Capability recovery")
        axes[1].set_ylim(min([-.25, *(v - .15 for v in recoveries if v is not None)]),
                         max([1.15, *(v + .15 for v in recoveries if v is not None)]))
        axes[1].grid(axis="y", alpha=.2)
        fig.suptitle(f"AgentDojo held-out validation families · {len(common)}/{len(plan)} paired tasks")
        fig.tight_layout()
        name = "metric_comparison.png"; fig.savefig(output_dir / name, dpi=180); figures.append(name); plt.close(fig)

        matrix = np.array([[score(row[role], role, "joint") for role in roles] for row in paired])
        fig, ax = plt.subplots(figsize=(7.5, max(5, .34 * len(common) + 1.5)))
        ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
        ax.set_xticks(range(3), ("Teacher", "Base", "SFT"))
        ax.set_yticks(range(len(common)), [row["task_id"] for row in paired], fontsize=8)
        for i in range(len(common)):
            for j in range(3):
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=8)
        ax.set_title("Official utility × safety by matched task")
        fig.tight_layout()
        name = "paired_task_heatmap.png"; fig.savefig(output_dir / name, dpi=180); figures.append(name); plt.close(fig)

    md = ["# AgentDojo Qwen3.5-9B 对比评估", "",
          f"状态：**{report['status']}**；已配对 **{len(common)}/{len(plan)}** 个任务（7 个未参与梯度训练的家族）。", "",
          "| 指标 | 教师 | 基座 | SFT | 能力恢复比例 |", "| --- | ---: | ---: | ---: | ---: |"]
    for metric, label in (("utility", "任务完成率"), ("safety", "注入安全通过率"), ("joint", "任务与安全同时成功")):
        item = all_metrics[metric]
        def show(value): return "未定义" if value is None or not math.isfinite(value) else f"{value:.1%}"
        md.append(f"| {label} (n={item['n']}) | {show(item['mean']['teacher'])} | {show(item['mean']['base'])} | {show(item['mean']['sft'])} | {show(item['recovery'])} |")
    md.extend(["", "注入安全通过率 = 1 − 注入攻击成功率；数值为 100% 表示攻击未成功。",
               "能力恢复比例 = (SFT − 基座) / (教师 − 基座)；分母为零时不报告比例。",
               "配对改进/退步/持平：" + ", ".join(f"{name}={paired_changes['joint'][name]}" for name in ("improved", "regressed", "tied")) + "。",
               "", "教师分数来自历史 Codex 轨迹；两个学生使用相同的本地 Qwen 工具循环和官方 AgentDojo verifier。",
               "这些任务来自 SFT 验证家族，未用于梯度更新；教师记录的采样基于历史成功轨迹，因此教师基线存在选择偏差。",
               "", "## 可视化", ""])
    for name in figures:
        md.append(f"![{name}]({name})")
    md.extend(["", "逐题评分表：`paired_task_scores.csv`；原始逐题记录：`base.jsonl`、`sft.jsonl`；固定任务计划：`evaluation_plan.jsonl`；数值汇总：`comparison.json`。"])
    (output_dir / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    visual_manifest = {"schema": "agentdojo_9b_comparison.visualization.v1",
                       "created_at": report["created_at"], "paired_tasks": len(common), "figures": figures,
                       "script_sha256": sha256(Path(__file__)),
                       "input_sha256": {name: sha256(output_dir / name) for name in ("evaluation_plan.jsonl", "base.jsonl", "sft.jsonl")},
                       "report_sha256": sha256(output_dir / "comparison.json"),
                       "paired_scores_sha256": sha256(output_dir / "paired_task_scores.csv")}
    (output_dir / "visualization_manifest.json").write_text(json.dumps(visual_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "paired_tasks": len(common), "planned_tasks": len(plan),
                      "metrics": all_metrics, "figures": figures}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
