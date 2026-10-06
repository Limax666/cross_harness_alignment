#!/usr/bin/env python3
"""Audit and remove unusable AgentHarm native trajectories.

Only completed test_public records with the official verdict, complete harness
context, and a clean parser state remain.  Failed-attempt files and raw CLI
event dumps are removed because the collector will retry every missing task.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = ROOT / "experiments" / "cross_harness_sft" / "outputs"
REPORT = ROOT / "experiments" / "cross_harness_sft" / "reports" / "agentharm_trajectory_audit_20260923.json"


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def usable(row: dict[str, Any]) -> tuple[bool, str]:
    metadata = row.get("metadata") or {}
    metrics = metadata.get("env_metrics") or {}
    messages = row.get("messages") or []
    system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    roles = {m.get("role") for m in messages if isinstance(m, dict)}
    if row.get("schema") != "cross_harness.native_trajectory.v2": return False, "not_native_schema"
    if not str(metadata.get("benchmark_version", "")).endswith("agentharm/test_public"): return False, "not_test_public"
    if row.get("status") != "completed" or not str(row.get("response") or "").strip(): return False, "incomplete"
    if not metadata.get("harness_id") or not metadata.get("task_id"): return False, "missing_provenance"
    if "<harness_context>" not in str(system) or not {"system", "user", "assistant"}.issubset(roles): return False, "missing_messages"
    if not isinstance(metadata.get("tool_history"), list): return False, "missing_tool_audit"
    if metrics.get("final_success") is not True: return False, "official_failure"
    if metrics.get("parser_problem") is True: return False, "parser_problem"
    if not (metadata.get("verifier_details") or {}).get("judge_backend"): return False, "missing_official_verdict"
    return True, "usable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="rewrite outputs and delete rejected artifacts")
    args = parser.parse_args()
    report: dict[str, Any] = {"criterion": "completed test_public + official success + parser clean + harness/tool provenance", "directories": {}}
    for directory in sorted(OUTPUTS.glob("agentharm*")):
        if not directory.is_dir():
            continue
        kept: dict[Path, list[dict[str, Any]]] = {}
        rejected = Counter()
        kept_by_task = Counter()
        input_rows = 0
        for path in directory.glob("*.jsonl"):
            if path.name.endswith(".errors.jsonl"):
                continue
            rows = read_rows(path)
            input_rows += len(rows)
            seen: set[tuple[str, str]] = set()
            for row in rows:
                ok, reason = usable(row)
                if ok:
                    key = (str((row.get("metadata") or {}).get("harness_id")), str((row.get("metadata") or {}).get("task_id")))
                    if key in seen:
                        rejected["duplicate_task"] += 1
                        continue
                    seen.add(key)
                    kept.setdefault(path, []).append(row)
                    kept_by_task[key[0]] += 1
                else:
                    rejected[reason] += 1
        report["directories"][directory.name] = {
            "input_rows": input_rows,
            "kept": sum(len(rows) for rows in kept.values()),
            "kept_by_harness": dict(kept_by_task),
            "rejected_by_reason": dict(rejected),
        }
        if args.apply:
            for path in directory.glob("*.jsonl"):
                if path.name.endswith(".errors.jsonl"):
                    path.unlink()
                    continue
                rows = kept.get(path, [])
                if rows:
                    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
                else:
                    path.unlink()
            raw = directory / "raw"
            if raw.exists():
                shutil.rmtree(raw)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
