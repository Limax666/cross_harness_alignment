#!/usr/bin/env python3
"""Require complete 128x3 judge results before comparing SFT, GRPO20, CHSPO150."""
from __future__ import annotations

import json
from pathlib import Path

from summarize_harnessrisk_sft_rl_step20 import (
    EXP, SFT_ROOT, RL_ROOT, SFT_TAG, HARNESSES, METRICS, JUDGE_MODEL,
    expected_cases, read_arm, count_metric, any_at_3,
)


def main():
    root = EXP / "outputs/harnessrisk_qwen35_2b_chspo_v4_step150/runs"
    cases = expected_cases()
    lines = ["# HarnessRisk：SFT、GRPO step 20 与 CHS-PO v4 step 150", "",
             f"每模型×harness 为同一128 case × 3轮；judge `{JUDGE_MODEL}`。",
             "CHS-PO v4 从v3 step 130恢复并改变奖励与采样；不同算法训练步数与响应预算并非严格匹配，",
             "本表是模型能力比较，不是控制全部训练变量的算法消融。失败/超时保留在分母。", "",
             "| Harness | 指标 | SFT | GRPO20 | CHSPO150 | CHSPO−GRPO (pp) | CHSPO−SFT (pp) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    payload = {"judge_model": JUDGE_MODEL, "harnesses": {}}
    for harness in HARNESSES:
        arms = {"sft": read_arm(SFT_ROOT, SFT_TAG[harness], "sft", harness, cases),
                "grpo20": read_arm(RL_ROOT, "rl20p1", "rl", harness, cases),
                "chspo150": read_arm(root, "chspo150v4p1", "rl", harness, cases)}
        payload["harnesses"][harness] = {}
        for key, label in METRICS:
            counts = {name: count_metric(arm, key) for name, arm in arms.items()}
            any3 = {name: any_at_3(arm, key) for name, arm in arms.items()}
            payload["harnesses"][harness][key] = {"counts": counts, "denominator": 384,
                                                 "any_at_3": any3, "case_denominator": 128}
            cells = [f"{counts[name]}/384 ({100*counts[name]/384:.2f}%)"
                     for name in ("sft", "grpo20", "chspo150")]
            delta_g = 100 * (counts["chspo150"] - counts["grpo20"]) / 384
            delta_s = 100 * (counts["chspo150"] - counts["sft"]) / 384
            lines.append(f"| {harness} | {label} | {' | '.join(cells)} | {delta_g:+.2f} | {delta_s:+.2f} |")
    lines += ["", "OpenClaw Persistence沿用已有judge口径，含已知初始化文件评分污染；不能将其直接解释为真实攻击性写入。"]
    target = EXP / "reports/harnessrisk_sft_grpo20_chspo150_comparison.md"
    target.write_text("\n".join(lines) + "\n")
    target.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
