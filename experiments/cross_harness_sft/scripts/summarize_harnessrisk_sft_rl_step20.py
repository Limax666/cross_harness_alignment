#!/usr/bin/env python3
"""Compare complete HarnessRisk v7 SFT and R11 step-20 RL evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
BENCH = EXP.parents[1] / "HarnessRisk" / "data" / "HarnessRisk"
SFT_ROOT = EXP / "outputs" / "harnessrisk_qwen35_2b_full_v7" / "runs"
RL_ROOT = EXP / "outputs" / "harnessrisk_qwen35_2b_rl_step20" / "runs"
SFT_TAG = {"hermes": "v7p6", "nanobot": "v7p8", "openclaw": "v7p6"}
HARNESSES = tuple(SFT_TAG)
METRICS = (
    ("utility_score", "Utility ↑"),
    ("attack_success_rate", "ASR ↓"),
    ("persistence_score", "Persistence ↓"),
    ("detection_score", "Detection ↑"),
)
JUDGE_MODEL = "openai/gpt-5.4-nano"


def expected_cases() -> set[str]:
    cases = set()
    for path in BENCH.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("case_id"), str):
            cases.add(data["case_id"])
    if len(cases) != 128:
        raise ValueError(f"Expected 128 HarnessRisk cases, found {len(cases)}")
    return cases


def read_arm(root: Path, tag: str, role: str, harness: str, cases: set[str]) -> dict:
    trials: dict[str, dict[int, dict]] = {case_id: {} for case_id in cases}
    status_counts: dict[str, int] = {}
    timeout_count = 0
    for repetition in range(1, 4):
        batch = root / f"{harness}-{role}" / f"{tag}-{role}-{harness}-r{repetition}"
        manifest = batch / "batch_manifest.jsonl"
        judge_manifest = batch / "llm_judge_multi_harness" / "manifest.json"
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        judged = json.loads(judge_manifest.read_text(encoding="utf-8"))
        for name, items in (("trajectory", rows), ("judge", judged)):
            case_ids = [item.get("case_id") for item in items]
            if len(case_ids) != 128 or set(case_ids) != cases or len(set(case_ids)) != 128:
                raise ValueError(f"Incomplete or duplicate {name} records: {batch}")
        for row in rows:
            if row.get("repetition") != repetition:
                raise ValueError(f"Repetition mismatch: {batch} {row.get('case_id')}")
            status = str(row.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
            timeout_count += row.get("exit_code") == 124
        for row in judged:
            if row.get("status") not in ("judged", "existing"):
                raise ValueError(f"Unscored case: {batch} {row.get('case_id')}")
            verdict = row.get("judge") or {}
            deployment = (verdict.get("_llm_metadata") or {}).get("deployment")
            if deployment != JUDGE_MODEL:
                raise ValueError(f"Judge mismatch: {batch} {row.get('case_id')}: {deployment}")
            if any(verdict.get(key) not in (0, 1) for key, _ in METRICS):
                raise ValueError(f"Non-binary or missing score: {batch} {row.get('case_id')}")
            trials[row["case_id"]][repetition] = {key: verdict[key] for key, _ in METRICS}
    if any(len(reps) != 3 for reps in trials.values()):
        raise ValueError(f"Missing repetitions for {role}/{harness}")
    return {"trials": trials, "status_counts": status_counts, "timeout_count": timeout_count}


def count_metric(arm: dict, key: str) -> int:
    return sum(verdict[key] for reps in arm["trials"].values() for verdict in reps.values())


def any_at_3(arm: dict, key: str) -> int:
    return sum(any(verdict[key] for verdict in reps.values()) for reps in arm["trials"].values())


def pct(count: int, denominator: int) -> str:
    return f"{count}/{denominator} ({100 * count / denominator:.2f}%)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-tag", default="rl20p1")
    parser.add_argument("--output", type=Path, default=EXP / "reports" / "harnessrisk_sft_vs_rl_step20.md")
    args = parser.parse_args()
    cases = expected_cases()
    arms = {}
    for harness in HARNESSES:
        arms[(harness, "sft")] = read_arm(SFT_ROOT, SFT_TAG[harness], "sft", harness, cases)
        arms[(harness, "rl")] = read_arm(RL_ROOT, args.rl_tag, "rl", harness, cases)

    lines = [
        "# HarnessRisk：全参数 SFT 与在线 RL step-20 对照",
        "",
        "SFT 为 Qwen3.5-2B v7 step 267；SFT+RL 是从该权重继续在线 GRPO 20 步的 R11 合并模型。",
        "每个模型×Harness 均使用同一 128 case、3 次独立 rollout（384 条），单 case 超时 300 秒；"
        f"四项指标均由 HarnessRisk LLM judge `{JUDGE_MODEL}` 评分。失败和超时轨迹保留在分母。",
        "",
        "| Harness | 指标 | SFT | SFT+RL step 20 | 变化（百分点） |",
        "|---|---|---:|---:|---:|",
    ]
    totals = {role: {key: 0 for key, _ in METRICS} for role in ("sft", "rl")}
    payload = {"judge_model": JUDGE_MODEL, "rl_tag": args.rl_tag, "harnesses": {}}
    for harness in HARNESSES:
        sft, rl = arms[(harness, "sft")], arms[(harness, "rl")]
        payload["harnesses"][harness] = {}
        for key, label in METRICS:
            a, b = count_metric(sft, key), count_metric(rl, key)
            totals["sft"][key] += a
            totals["rl"][key] += b
            lines.append(f"| {harness} | {label} | {pct(a, 384)} | {pct(b, 384)} | {(b-a)/384*100:+.2f} |")
            payload["harnesses"][harness][key] = {"sft": a, "rl": b, "denominator": 384}
    lines.extend([
        "",
        "四项指标方向分别为 Utility↑、ASR↓、Persistence↓、Detection↑。下表的 `any@3` 指同一 case 的三轮中至少一次为 1。",
        "",
        "| Harness | Utility any@3：SFT → RL | ASR any@3：SFT → RL | 超时：SFT → RL |",
        "|---|---:|---:|---:|",
    ])
    for harness in HARNESSES:
        sft, rl = arms[(harness, "sft")], arms[(harness, "rl")]
        lines.append(
            f"| {harness} | {pct(any_at_3(sft, 'utility_score'), 128)} → "
            f"{pct(any_at_3(rl, 'utility_score'), 128)} | "
            f"{pct(any_at_3(sft, 'attack_success_rate'), 128)} → "
            f"{pct(any_at_3(rl, 'attack_success_rate'), 128)} | "
            f"{sft['timeout_count']} → {rl['timeout_count']} |"
        )
        payload["harnesses"][harness]["run_status"] = {
            "sft": {k: sft[k] for k in ("status_counts", "timeout_count")},
            "rl": {k: rl[k] for k in ("status_counts", "timeout_count")},
        }
    lines.extend([
        "",
        "## 判断",
        "",
    ])
    for harness in HARNESSES:
        before = {key: count_metric(arms[(harness, "sft")], key) for key, _ in METRICS}
        after = {key: count_metric(arms[(harness, "rl")], key) for key, _ in METRICS}
        improving = [label for key, label in METRICS if (after[key] - before[key]) * (1 if '↑' in label else -1) > 0]
        regressing = [label for key, label in METRICS if (after[key] - before[key]) * (1 if '↑' in label else -1) < 0]
        lines.append(f"- **{harness}**：改善 {', '.join(improving) or '无'}；退化 {', '.join(regressing) or '无'}。")
    lines.extend([
        "",
        "OpenClaw 的初始化文件此前被 judge 计作 Persistence 证据；该列须经初始化文件排除后再判断真实持久化变化。"
        "低 ASR 若伴随低 Utility，不能单独解释为安全能力提升。论文使用 GPT-5.4，当前 judge 是 GPT-5.4-nano，"
        "因此本表仅用于同一 judge 下的 SFT→RL 配对对照。",
        "",
        "原始记录：`outputs/harnessrisk_qwen35_2b_full_v7/runs/` 与 "
        "`outputs/harnessrisk_qwen35_2b_rl_step20/runs/`。",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
