#!/usr/bin/env python3
"""Rule-filter mixed teacher trajectories and emit TRL conversational SFT JSONL.

The input corpus is never modified.  Accepted real and oracle-derived synthetic rows
share exactly the same eligibility and harness-balancing treatment; provenance remains
in metadata for later audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT / "experiments/cross_harness_sft/data/raw/multi_harness_teacher_v1/trajectories.jsonl"

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)(api[_ -]?key|authorization|bearer)\s*[:=]\s*[^\s,;]{8,}"),
]
ABS_PATH = re.compile(r"/data/home/[^\s'\"`]+")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_harnessrisk_scores() -> dict[str, dict[str, dict[str, float]]]:
    """Return harness -> run_dir absolute string -> official metrics."""
    outputs = PROJECT / "experiments/cross_harness_sft/outputs"
    summaries = {
        "hermes": outputs / "harnessrisk_hermes_gpt56_sol_half/runs/gpt56-sol-hermes-real64-20260916T081345Z/evaluation_summary.json",
        "nanobot": outputs / "harnessrisk_nanobot_gpt56_sol_half/runs/gpt56-sol-nanobot-real64-20260916T081640Z/evaluation_summary.json",
    }
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for harness, path in summaries.items():
        if not path.exists():
            continue
        for run in load_json(path).get("runs", []):
            metrics = run.get("metrics", {})
            if run.get("run_dir") and metrics:
                result[harness][str(Path(run["run_dir"]).resolve())] = metrics
    return result


def stable_split(benchmark: str, family: str) -> str:
    """Family-disjoint, deterministic 80/10/10 assignment."""
    # HarnessRisk has six official top-level case families.  Assigning them explicitly
    # keeps every harness represented in train/validation/test after synthetic completion.
    if benchmark == "harnessrisk":
        harnessrisk = {
            "action": "train", "daily": "train", "memory": "train", "recovery": "train",
            "setup": "validation", "skill": "test",
        }
        if family in harnessrisk:
            return harnessrisk[family]
    value = int(hashlib.sha256(f"multiharness-sft-v1:{benchmark}:{family}".encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "validation" if value < 90 else "test"


def sanitize_text(value: str) -> tuple[str, int]:
    count = 0
    value, changed = ABS_PATH.subn("<workspace>", value)
    count += changed
    for pattern in SECRET_PATTERNS:
        value, changed = pattern.subn("<REDACTED_SECRET>", value)
        count += changed
    return value, count


def sanitize_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    clean = json.loads(json.dumps(messages, ensure_ascii=False))
    redactions = 0
    for message in clean:
        content = message.get("content")
        if isinstance(content, str):
            message["content"], n = sanitize_text(content)
            redactions += n
    return clean, redactions


def normalize_tool_call_arguments(messages: list[dict[str, Any]]) -> list[str]:
    """Match Qwen's current chat-template contract: arguments are mappings."""
    errors = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                errors.append("invalid_tool_call_function")
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    errors.append("invalid_tool_call_arguments_json")
                    continue
            if not isinstance(function.get("arguments"), dict):
                errors.append("non_mapping_tool_call_arguments")
    return sorted(set(errors))


def valid_messages(messages: Any) -> list[str]:
    if not isinstance(messages, list) or not messages:
        return ["missing_messages"]
    reasons = []
    roles = []
    for message in messages:
        if not isinstance(message, dict):
            reasons.append("non_dict_message")
            continue
        role = message.get("role")
        roles.append(role)
        if role not in ALLOWED_ROLES:
            reasons.append("invalid_role")
        if role != "tool" and not message.get("content") and not message.get("tool_calls"):
            reasons.append("empty_non_tool_message")
    if "user" not in roles or "assistant" not in roles:
        reasons.append("missing_user_or_assistant")
    if roles and roles[-1] != "assistant":
        reasons.append("final_not_assistant")
    return sorted(set(reasons))


def reconstruct_harnessrisk(row: dict[str, Any]) -> tuple[list[dict[str, str]] | None, list[str]]:
    result = row["raw_payload"].get("harness_result") or {}
    turns = result.get("turns")
    if not isinstance(turns, list) or len(turns) != 3:
        return None, ["missing_three_turn_replay"]
    messages: list[dict[str, str]] = []
    for turn in turns:
        user = turn.get("message") if isinstance(turn, dict) else None
        assistant = turn.get("response_text") if isinstance(turn, dict) else None
        if not isinstance(user, str) or not user.strip() or not isinstance(assistant, str) or not assistant.strip():
            return None, ["empty_owner_or_assistant_turn"]
        messages.extend(({"role": "user", "content": user}, {"role": "assistant", "content": assistant}))
    return messages, []


def agentdojo_rules(row: dict[str, Any]) -> list[str]:
    raw = row["raw_payload"]
    meta = raw.get("metadata", {})
    reward = meta.get("reward_breakdown", {})
    reasons = []
    if row.get("collection_status") != "completed": reasons.append("not_completed")
    if reward.get("utility_reward") != 1.0: reasons.append("utility_not_verified")
    if reward.get("safety_reward") != 1.0: reasons.append("safety_not_verified")
    if raw.get("response_length", 0) <= 0: reasons.append("empty_final_response")
    return reasons


def harnessrisk_rules(row: dict[str, Any], score_index: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    if row.get("collection_status") != "completed":
        return ["not_completed"]
    if not row.get("has_observable_turn_content"):
        return ["no_observable_turn_content"]
    harness = row["harness_name"]
    source = str((PROJECT / row["source_path"]).resolve())
    metrics = score_index.get(harness, {}).get(source)
    if metrics is None:
        return ["missing_official_evaluation"]
    reasons = []
    if metrics.get("utility_score") != 1.0: reasons.append("utility_not_verified")
    if metrics.get("attack_success_rate") != 0.0: reasons.append("attack_succeeded")
    if metrics.get("persistence_score") != 0.0: reasons.append("persistence_detected")
    return reasons


def synthetic_rules(row: dict[str, Any]) -> list[str]:
    payload = row["raw_payload"]
    contract = payload.get("quality_contract", {})
    reasons = []
    if not (payload.get("synthetic") is True and payload.get("oracle_derived") is True): reasons.append("invalid_synthetic_provenance")
    if contract.get("required_attack_success_rate") != 0.0: reasons.append("synthetic_contract_asr")
    if contract.get("required_utility_score") != 1.0: reasons.append("synthetic_contract_utility")
    if contract.get("required_persistence_score") != 0.0: reasons.append("synthetic_contract_persistence")
    return reasons


def convert_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    if row["benchmark"] == "agentdojo":
        base_messages = row.get("messages")
    elif row["source_kind"] == "synthetic_oracle_candidate":
        base_messages = row.get("messages")
    else:
        base_messages, errors = reconstruct_harnessrisk(row)
        if errors:
            return None, errors
    errors = valid_messages(base_messages)
    if errors:
        return None, errors
    # Qwen3.5 accepts a system instruction only at the beginning.  Preserve the
    # source policy by merging it with the model-visible harness contract rather
    # than emitting two consecutive system turns.
    messages = json.loads(json.dumps(base_messages, ensure_ascii=False))
    if messages and messages[0].get("role") == "system":
        source_policy = str(messages[0].get("content") or "").strip()
        messages[0]["content"] = row["harness_context"] + ("\n\n" + source_policy if source_policy else "")
    else:
        messages.insert(0, {"role": "system", "content": row["harness_context"]})
    errors = normalize_tool_call_arguments(messages)
    if errors:
        return None, errors
    messages, redactions = sanitize_messages(messages)
    tool_schemas = (row.get("tools") or []) if row["benchmark"] == "agentdojo" else []
    return {
        "messages": messages,
        "tools": tool_schemas,
        "metadata": {
            "record_id": row["record_id"], "benchmark": row["benchmark"],
            "harness_name": row["harness_name"], "harness_context": row["harness_context"],
            "source_kind": row["source_kind"], "synthetic": row["synthetic"],
            "task_id": row.get("task_id"), "family": row.get("family"),
            "teacher_model": row.get("teacher_model"), "source_path": row["source_path"],
            "redaction_count": redactions, "tool_audit_count": len(row.get("tools") or []),
        },
    }, []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "experiments/cross_harness_sft/data/sft/multi_harness_trl_v1")
    args = parser.parse_args()
    score_index = load_harnessrisk_scores()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        if row["source_kind"] == "synthetic_oracle_candidate": reasons = synthetic_rules(row)
        elif row["benchmark"] == "agentdojo": reasons = agentdojo_rules(row)
        else: reasons = harnessrisk_rules(row, score_index)
        converted = None
        if not reasons:
            converted, reasons = convert_row(row)
        audit = {"record_id": row["record_id"], "benchmark": row["benchmark"], "harness_name": row["harness_name"], "source_kind": row["source_kind"], "synthetic": row["synthetic"], "accepted": not reasons, "reasons": reasons}
        audits.append(audit)
        if converted is not None and not reasons:
            converted["metadata"]["split"] = stable_split(row["benchmark"], row.get("family") or "unknown")
            accepted.append(converted)
    # Equal total harness mass in every split; retained only as metadata for the TRL sampler.
    counts = Counter((x["metadata"]["split"], x["metadata"]["harness_name"]) for x in accepted)
    for row in accepted:
        meta = row["metadata"]
        meta["sampling_weight"] = 1.0 / counts[(meta["split"], meta["harness_name"])]
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted: by_split[row["metadata"]["split"]].append(row)
    for split in ("train", "validation", "test"):
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in by_split[split]: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "audit.jsonl").open("w", encoding="utf-8") as handle:
        for audit in audits: handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
    rejected = [a for a in audits if not a["accepted"]]
    with (args.output_dir / "rejected.jsonl").open("w", encoding="utf-8") as handle:
        for audit in rejected: handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
    report = {
        "input": str(args.input), "eligibility": "real and synthetic rows use equal admission and harness balancing; provenance retained in metadata",
        "input_rows": len(audits), "accepted_rows": len(accepted), "rejected_rows": len(rejected),
        "accepted_by_harness": dict(Counter(x["metadata"]["harness_name"] for x in accepted)),
        "accepted_by_source_kind": dict(Counter(x["metadata"]["source_kind"] for x in accepted)),
        "accepted_by_split": {s: dict(Counter(x["metadata"]["harness_name"] for x in rows)) for s, rows in by_split.items()},
        "rejection_reasons": dict(Counter(reason for a in rejected for reason in a["reasons"])),
        "trl_format": {"conversation_column": "messages", "tools_column": "tools", "metadata_column": "metadata", "assistant_only_loss": True},
    }
    (args.output_dir / "filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
