#!/usr/bin/env python3
"""Summarize native SafeClawArena verdicts without imputing missing cases."""

from __future__ import annotations

import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "outputs/safeclawarena_chspo150_eval"
REPORT = EXP / "reports/safeclawarena_heldout_sft_vs_chspo150_20261015.md"


def stage(platform: str, role: str) -> tuple[int, list[dict]]:
    base = OUT / platform / role
    manifest = base / "run_manifest.json"
    if not manifest.is_file():
        return 0, []
    case_ids = json.loads(manifest.read_text())["case_ids"]
    rows = []
    for rep in (1, 2, 3):
        for case_id in case_ids:
            path = base / f"rep{rep}" / f"{case_id}.json"
            if not path.is_file():
                continue
            try:
                result = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(result.get("score"), (int, float)) or not isinstance(result.get("attack_succeeded"), bool):
                continue
            replies = [s.get("agent_response", "") for s in result.get("sessions", [])]
            result["protocol_error"] = any("Tool call rejected:" in str(x) for x in replies)
            result["case_id"] = case_id
            result["rep"] = rep
            rows.append(result)
    return len(case_ids) * 3, rows


def pct(rows: list[dict], key: str) -> str:
    if not rows:
        return "—"
    count = sum(bool(r.get(key)) for r in rows)
    return f"{count}/{len(rows)} ({count / len(rows):.1%})"


def main() -> None:
    lines = ["# SafeClawArena 留出家族：SFT 与 CHS-PO step-150（进行中）", "",
             "使用 v7 family split 中未进入 SFT 的 PSE:2.4（10 题）与 SSI:1.2（20 题），"
             "每题 3 次独立复位；NemoClaw、SecLaw 都使用官方 `judge.py` 判分。"
             "两组只比较安全攻击指标；SafeClawArena 没有良性任务，不能据此报告 Utility。", "",
             "| Harness | 模型 | 有效判分 / 计划 | 平均安全分 ↑ | 攻击成功率 ↓ | Detection ↑ | 工具协议错误 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    full = True
    for platform in ("nemoclaw", "seclaw"):
        for role in ("sft", "chspo150"):
            planned, rows = stage(platform, role)
            if planned == 0:
                planned = 90
            full &= len(rows) == planned
            mean_score = f"{sum(float(r['score']) for r in rows)/len(rows):.3f}" if rows else "—"
            lines.append(f"| {platform} | {role} | {len(rows)}/{planned} | {mean_score} | "
                         f"{pct(rows, 'attack_succeeded')} | {pct(rows, 'detection')} | {pct(rows, 'protocol_error')} |")
    lines.extend(["", "工具协议错误指模型输出了不属于该 harness 的工具名或不合 schema 的调用；"
                  "它可能让攻击失败，但不能证明模型完成了合法任务。", "",
                  "当前结论：" + ("四组均完整，可做同题配对分析。" if full else "矩阵尚未完整，不能宣称 CHS-PO 涨点或降 ASR。"), ""])
    REPORT.write_text("\n".join(lines))
    print(REPORT)


if __name__ == "__main__":
    main()
