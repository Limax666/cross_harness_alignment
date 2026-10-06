#!/usr/bin/env python3
"""Re-score stored native AgentHarm traces with the current judge, without teacher reruns.

This reconstructs the official AgentHarm scoring transcript from the audited
tool calls and observations.  It writes new rows; original collection files
are untouched.  Requires the AgentHarm Inspect environment and judge API key.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/cross_harness_sft"
sys.path.insert(0, str(EXP / "src"))
from cross_harness_sft.backends.agentharm import AgentHarmDriver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-config", type=Path, default=EXP / "configs/agentharm_worker_test_public.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N eligible rows (0 means all)")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Rejudging must write a separate output file")
    driver = AgentHarmDriver(str(args.worker_config))
    approved_model = str(driver.judge_config.get("model") or "")
    done = set()
    if args.output.exists():
        done = {json.loads(line)["id"] for line in args.output.open(encoding="utf-8") if line.strip()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"rescored": 0, "rejected": 0, "skipped": 0, "errors": 0}
    error_path = args.output.with_suffix(".errors.jsonl")
    with args.input.open(encoding="utf-8") as source, args.output.open("a", encoding="utf-8") as target, error_path.open("a", encoding="utf-8") as errors:
        for line in source:
            if not line.strip(): continue
            row = json.loads(line)
            meta = row.get("metadata") or {}
            if row.get("id") in done or not str(meta.get("benchmark_version", "")).endswith("agentharm/test_public"):
                counts["skipped"] += 1; continue
            details = meta.get("verifier_details") or {}
            if details.get("judge_backend") == driver.judge_backend and details.get("judge_model") == approved_model:
                counts["skipped"] += 1; continue
            task_id = str(meta.get("task_id") or "")
            try:
                print(f"[rejudge] start id={row.get('id')} task={task_id} judge={approved_model}", flush=True)
                episode = driver.start(task_id, int(meta.get("seed") or 0))["episode"]
                episode.trace = [{"name": call["tool_name"], "arguments": call["arguments"],
                                  "result": call.get("tool_result"),
                                  "error": str(call.get("observation")) if call.get("invalid_call") else None}
                                 for call in meta.get("tool_history") or []]
                verdict = driver.finish(episode, str(row.get("response") or ""))
                updated = json.loads(json.dumps(row, ensure_ascii=False))
                updated_meta = updated["metadata"]
                updated_meta["verifier_details"] = verdict["details"]
                updated_meta["env_metrics"].update({
                    "utility_reward": float(verdict["utility_reward"]),
                    "safety_reward": float(verdict["safety_reward"]),
                    "final_success": bool(verdict["final_success"]),
                    "risk_success": bool(verdict["risk_success"]),
                })
                updated_meta["reward_breakdown"] = {
                    "utility_reward": float(verdict["utility_reward"]),
                    "safety_reward": float(verdict["safety_reward"]),
                }
                target.write(json.dumps(updated, ensure_ascii=False) + "\n")
                target.flush()
                done.add(row["id"])
                counts["rescored" if verdict["final_success"] else "rejected"] += 1
                print(f"[rejudge] done id={row.get('id')} final_success={verdict['final_success']}", flush=True)
                if args.limit and counts["rescored"] + counts["rejected"] >= args.limit:
                    break
            except Exception as exc:
                counts["errors"] += 1
                print(f"[rejudge] error id={row.get('id')} {type(exc).__name__}: {exc}", flush=True)
                errors.write(json.dumps({"id": row.get("id"), "task_id": task_id,
                                         "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False) + "\n")
                errors.flush()
                if args.limit and counts["errors"] >= args.limit:
                    break
    print(json.dumps({"output": str(args.output), **counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
