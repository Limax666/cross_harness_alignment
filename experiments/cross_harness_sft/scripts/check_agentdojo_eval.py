#!/usr/bin/env python3
"""Fail-closed gate for AgentDojo native-harness smoke and completed runs."""
import argparse
import json
from collections import Counter
from pathlib import Path


def records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check(root, task_ids, seeds, smoke):
    expected = {(role, harness, task, seed)
                for role in ("base", "sft")
                for harness in ("codex_native", "claude_code")
                for task in task_ids for seed in seeds}
    found = Counter()
    problems = []
    for role in ("base", "sft"):
        for harness in ("codex_native", "claude_code"):
            group = root / role / harness
            rows = records(group / "raw_native.jsonl")
            errors = records(group / "raw_native.errors.jsonl")
            if errors:
                problems.append(f"{role}/{harness}: {len(errors)} runtime errors; first={str(errors[0].get('error'))[:250]}")
            tool_evidence = 0
            model_invalid_calls = 0
            task_types = set()
            for row in rows:
                meta = row.get("metadata") or {}
                key = (role, harness, meta.get("task_id"), meta.get("seed"))
                found[key] += 1
                task_types.add(meta.get("task_type"))
                metrics = meta.get("env_metrics") or {}
                controlled_incomplete = (
                    row.get("status") == "incomplete" and bool(metrics.get("hit_max_turns"))
                )
                if row.get("status") != "completed" and not controlled_incomplete:
                    problems.append(f"{key}: status={row.get('status')}")
                if meta.get("scoring_status") != "scored" or not all(
                    field in metrics and metrics[field] is not None
                    for field in ("utility_reward", "safety_reward", "final_success", "risk_success")):
                    problems.append(f"{key}: official verifier missing")
                # AgentDojo counts a model-generated unknown/malformed tool name as an
                # invalid call. That is benchmark behavior (and can reduce utility),
                # not a transport/parser failure. Only parser_problem indicates the
                # gateway failed to decode the CLI's tool-call protocol.
                if metrics.get("parser_problem"):
                    problems.append(f"{key}: tool-call parser failure")
                model_invalid_calls += int(metrics.get("invalid_tool_call_count") or 0)
                bad = (row.get("native_stderr") or "") + "\n" + (row.get("native_stdout") or "")
                if any(term in bad.lower() for term in ("unsupported custom tool call", "deprecated", "unsupported call")):
                    problems.append(f"{key}: CLI protocol/model error")
                calls = meta.get("tool_history") or []
                tool_evidence += sum(not call.get("invalid_call") and call.get("observation") is not None for call in calls)
            if smoke:
                if task_types != {"clean", "injection"}:
                    problems.append(f"{role}/{harness}: smoke lacks clean + injection")
                if tool_evidence == 0:
                    problems.append(f"{role}/{harness}: no real tool call and result")
            print(f"{role}/{harness}: rows={len(rows)} errors={len(errors)} real_tool_results={tool_evidence} model_invalid_calls={model_invalid_calls}")
    missing = expected - set(found)
    unexpected = set(found) - expected
    duplicates = {key: n for key, n in found.items() if n != 1}
    if missing: problems.append(f"missing {len(missing)} planned trajectories; first={next(iter(missing))}")
    if unexpected: problems.append(f"unexpected {len(unexpected)} trajectories; first={next(iter(unexpected))}")
    if duplicates: problems.append(f"duplicate {len(duplicates)} trajectories; first={next(iter(duplicates.items()))}")
    for problem in problems[:25]:
        print("FAIL:", problem)
    if len(problems) > 25:
        print(f"... {len(problems) - 25} further problems")
    return not problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    task_ids = manifest["smoke_task_ids"] if args.smoke else manifest["task_ids"]
    seeds = [0] if args.smoke else [0, 1, 2]
    raise SystemExit(0 if check(args.root, task_ids, seeds, args.smoke) else 1)


if __name__ == "__main__":
    main()
