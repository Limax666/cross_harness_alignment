#!/usr/bin/env python3
"""Build a strict, family-disjoint SFT corpus from the two native collections."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SOURCES = {
    "codex": ("codex_native", "gpt-5.6-sol"),
    "claude": ("claude_code_openrouter", "openai/gpt-5.6-sol"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject(row: dict[str, Any], source: str, max_turns: int, max_response_chars: int) -> str | None:
    meta = row.get("metadata")
    if not isinstance(meta, dict):
        return "missing_metadata"
    harness, teacher = SOURCES[source]
    if meta.get("benchmark_id") != "agentdojo":
        return "wrong_benchmark"
    if meta.get("harness_id") != harness or meta.get("teacher_id") != teacher:
        return "wrong_teacher_or_harness"
    if meta.get("native_harness") is not True or meta.get("controlled_teacher") is not False:
        return "not_native_harness"
    required = ("benchmark_version", "task_id", "semantic_task_family_id", "harness_version", "harness_sha256", "safety_module")
    if any(not meta.get(key) for key in required):
        return "missing_provenance"
    if row.get("status") != "completed" or row.get("remove_sample"):
        return "not_completed"
    metrics = meta.get("env_metrics")
    rewards = meta.get("reward_breakdown")
    if not isinstance(metrics, dict) or not isinstance(rewards, dict):
        return "missing_verifier"
    if metrics.get("parser_problem"):
        return "parser_problem"
    if int(metrics.get("invalid_tool_call_count") or 0):
        return "invalid_tool_call"
    if metrics.get("hit_max_turns") or int(metrics.get("model_turn_count") or 0) > max_turns:
        return "too_many_turns"
    if metrics.get("explicit_finish") is not True:
        return "no_explicit_finish"
    if int(row.get("response_length") or 0) > max_response_chars:
        return "response_too_long"
    if metrics.get("risk_success") or rewards.get("safety_reward") != 1.0:
        return "safety_failure"
    if rewards.get("utility_reward") != 1.0 or metrics.get("final_success") is not True:
        return "utility_failure"
    if not str(meta.get("visible_reasoning") or "").strip():
        return "missing_visible_reasoning"
    if not row.get("native_events") or not meta.get("tool_history"):
        return "missing_audited_history"
    if not str(meta.get("skillrl_system_instruction") or "").strip() or not str(meta.get("skillrl_user_content") or "").strip():
        return "missing_exact_context"
    tools = meta.get("available_tools")
    if not isinstance(tools, list) or not tools:
        return "missing_tool_registry"
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return "missing_messages"
    if [messages[0].get("role"), messages[1].get("role"), messages[-1].get("role")] != ["system", "user", "assistant"]:
        return "invalid_roles"
    if any(not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"} for message in messages):
        return "invalid_messages"
    final = str(messages[-1].get("content") or "")
    if not re.search(r"<think>\s*\S[\s\S]*?</think>", final):
        return "missing_final_rationale"
    if not final.split("</think>", 1)[-1].strip() or "<tool_call>" in final:
        return "invalid_final_answer"
    return None


def normalise_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        clean = {key: value for key, value in message.items() if key in {"role", "content", "tool_calls", "tool_call_id", "name"}}
        if clean.get("tool_calls"):
            calls = []
            for call in clean["tool_calls"]:
                call = dict(call)
                function = dict(call.get("function") or {})
                if isinstance(function.get("arguments"), str):
                    function["arguments"] = json.loads(function["arguments"])
                if not isinstance(function.get("arguments"), dict):
                    raise ValueError("non_object_tool_arguments")
                call["function"] = function
                calls.append(call)
            clean["tool_calls"] = calls
        result.append(clean)
    return result


def bucket(family: str, seed: str, fraction: float) -> str:
    value = int.from_bytes(hashlib.sha256((seed + "\0" + family).encode()).digest()[:8], "big") / 2**64
    return "validation" if value < fraction else "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-input", required=True, type=Path)
    parser.add_argument("--claude-input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", default="agentdojo-cross-harness-sft-v1")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-response-chars", type=int, default=16384)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be between zero and one")
    paths = {"codex": args.codex_input, "claude": args.claude_input}
    rows = {"train": [], "validation": []}
    reasons: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    seen: set[str] = set()
    duplicates = 0
    for source, path in paths.items():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source_counts[source] += 1
                row = json.loads(line)
                reason = reject(row, source, args.max_turns, args.max_response_chars)
                if reason:
                    reasons[f"{source}:{reason}"] += 1
                    continue
                try:
                    messages = normalise_messages(row["messages"])
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    reasons[f"{source}:invalid_tool_calls:{exc}"] += 1
                    continue
                meta = row["metadata"]
                fingerprint = hashlib.sha256(json.dumps({"messages": messages, "tools": meta["available_tools"]}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                if fingerprint in seen:
                    duplicates += 1
                    continue
                seen.add(fingerprint)
                family = str(meta["semantic_task_family_id"])
                item = {"messages": messages, "tools": meta["available_tools"], "meta": {
                    "episode_id": row.get("episode_id"), "harness": source, "family": family,
                    "task_id": meta["task_id"], "task_type": meta.get("task_type"), "safety_module": meta["safety_module"]}}
                rows[bucket(family, args.split_seed, args.validation_fraction)].append(item)
    train_families = {item["meta"]["family"] for item in rows["train"]}
    validation_families = {item["meta"]["family"] for item in rows["validation"]}
    if train_families & validation_families:
        raise RuntimeError("semantic family overlap")
    if not rows["train"] or not rows["validation"]:
        raise RuntimeError("filtering yielded an empty split")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = {"schema": "agentdojo.cross_harness.filter.v1", "source_counts": dict(source_counts),
              "accepted_before_dedup": sum(map(len, rows.values())) + duplicates, "exact_duplicates": duplicates,
              "train": len(rows["train"]), "validation": len(rows["validation"]), "family_overlap": 0,
              "rejection_reasons": dict(sorted(reasons.items())), "inputs_sha256": {name: sha256(path) for name, path in paths.items()},
              "outputs_sha256": {split: sha256(args.output_dir / f"{split}.jsonl") for split in rows},
              "harness_distribution": {split: dict(Counter(item["meta"]["harness"] for item in items)) for split, items in rows.items()}}
    (args.output_dir / "filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
