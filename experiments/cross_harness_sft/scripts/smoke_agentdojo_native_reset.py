#!/usr/bin/env python3
"""Replay one *train-family* benign trace through AgentDojo's real runtime.

This is a reset/official-outcome gate, not an on-policy model rollout.  It
deliberately does not manufacture path-safety or persistence verdicts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cross_harness_sft.backends.agentdojo import AgentDojoDriver
from rl_agentdojo_path_audit import NativePathRecorder


ROOT = Path(__file__).resolve().parents[1]


def state_digest(environment: object) -> str:
    payload = environment.model_dump_json().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def replay(driver: AgentDojoDriver, row: dict) -> dict:
    task_id = row["task_id"]
    if row["split"] != "train" or row["benchmark"] != "agentdojo" or not task_id.endswith(":clean"):
        raise ValueError("native reset smoke requires an AgentDojo clean train-family row")
    first = driver.start(task_id, 0)["episode"]
    initial = state_digest(first.environment)
    recorder = NativePathRecorder(driver, first)
    calls = 0
    invalid_calls = 0
    final_answer = ""
    for message in row["messages"]:
        if message["role"] != "assistant":
            continue
        final_answer = str(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            function = call["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must decode to an object")
            result = recorder.tool(function["name"], arguments)
            calls += 1
            invalid_calls += bool(result["invalid_call"])
    if calls == 0:
        raise RuntimeError("selected teacher trace has no tool call")
    finished = driver.finish(first, final_answer)
    after = state_digest(first.environment)
    second = driver.start(task_id, 1)["episode"]
    reset = state_digest(second.environment)
    if initial != reset:
        raise AssertionError("native environment failed reset check")
    if invalid_calls:
        raise ValueError(f"native replay produced {invalid_calls} invalid calls")
    audit = recorder.audit(task_id)
    return {
        "task_id": task_id, "source_family_id": row["source_family_id"],
        "native_tool_calls": calls, "official_utility": finished["utility_reward"],
        "state_changed": initial != after, "reset_verified": True,
        "path_safety_verified": audit.all_calls_observed and audit.side_effects_inspected,
        "persistence_verified": audit.prohibited_persistence is not None,
        "audit_version": audit.audit_version, "prohibited_persistence": audit.prohibited_persistence,
        "on_policy_rollout": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=ROOT / "data/sft/teacher_release_v7/accepted_train.jsonl")
    parser.add_argument("--worker-config", type=Path, default=ROOT / "configs/agentdojo_worker.yaml")
    parser.add_argument("--require-state-change", action="store_true")
    args = parser.parse_args()
    driver = AgentDojoDriver(str(args.worker_config))
    with args.release.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if (row.get("benchmark") == "agentdojo" and row.get("split") == "train"
                    and str(row.get("task_id", "")).endswith(":clean")
                    and row.get("metrics", {}).get("tool_calls", 0) > 0):
                try:
                    result = replay(driver, row)
                except (KeyError, ValueError):
                    continue
                if args.require_state_change and not result["state_changed"]:
                    continue
                print(json.dumps(result, ensure_ascii=False))
                return
    raise RuntimeError("no accepted train-family AgentDojo trace passed native replay")


if __name__ == "__main__":
    main()
