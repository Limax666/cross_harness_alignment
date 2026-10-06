#!/usr/bin/env python3
"""Check that each requested AgentHarm test_public task has a recorded outcome."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/cross_harness_sft"
PROMPTS = EXP / "data/raw/agentharm_prompts_v2.jsonl"
OUTPUTS = EXP / "outputs"
HARNESS_DIRS = {
    "codex": "agentharm_codex_gpt56_sol_native",
    "claude_code": "agentharm_claudecode_glm53flash_native",
    "hermes": "agentharm_hermes_glm53flash_native",
    "nanobot": "agentharm_nanobot_glm53flash_native",
}


def load(path: Path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--harnesses", default=",".join(HARNESS_DIRS))
    p.add_argument("--require-complete", action="store_true")
    args = p.parse_args()
    requested = [x.strip() for x in args.harnesses.split(",") if x.strip()]
    unknown = sorted(set(requested) - set(HARNESS_DIRS))
    if unknown:
        raise SystemExit(f"Unknown AgentHarm harnesses: {unknown}")
    expected: set[str] = set()
    for row in load(PROMPTS) or []:
        if row.get("split") != "test_public":
            continue
        kind = str(row.get("task") or "")
        task_id = str(row.get("id") or "")
        prefix = "benign" if kind == "benign" else "harmful" if kind == "harmful" else ""
        if prefix and task_id:
            expected.add(f"{prefix}:{task_id}")
    if not expected:
        raise SystemExit(f"No test_public task IDs found in {PROMPTS}")

    report: dict[str, Any] = {"expected_tasks": len(expected), "harnesses": {}, "complete": True}
    for harness in requested:
        out_dir = OUTPUTS / HARNESS_DIRS[harness]
        success_ids: set[str] = set()
        error_ids: set[str] = set()
        for row in load(out_dir / "test_public_v1.jsonl") or []:
            task_id = str((row.get("metadata") or {}).get("task_id") or "")
            if task_id:
                success_ids.add(task_id)
        for row in load(out_dir / "test_public_v1.errors.jsonl") or []:
            task_id = str((row.get("case") or {}).get("task_id") or "")
            if task_id:
                error_ids.add(task_id)
        recorded = success_ids | error_ids
        missing = sorted(expected - recorded)
        result = {
            "success_rows_unique_task_ids": len(success_ids),
            "error_task_ids": len(error_ids),
            "recorded_any_outcome": len(recorded & expected),
            "missing_count": len(missing),
            "missing_sample": missing[:12],
            "complete": not missing,
        }
        report["harnesses"][harness] = result
        report["complete"] = report["complete"] and result["complete"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_complete and not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
