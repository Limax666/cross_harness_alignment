#!/usr/bin/env python3
"""Audit and join previously filtered cross-harness SFT sources.

This script never edits source data. It treats the legacy mixed corpus as an
audited input, retains every row in an audit/classification ledger, and only
emits primary-SFT rows from real trajectories with a model-visible harness
context and a defensible source-specific success signal. In particular,
legacy HarnessRisk text-only rows and synthetic oracle candidates are
classified but excluded by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/cross_harness_sft"
DEFAULT_LEGACY = EXP / "data/sft/multi_harness_combined_v1"
DEFAULT_AGENTHARM = EXP / "data/sft/agentharm_native_verified_v1"
DEFAULT_OUTPUT = EXP / "data/sft/multidataset_joint_sft_v1"
ACTBENCH = EXP / "data/sft/actbench_safe_real_v1_sanitized"
RAW_MULTIHARNESS = EXP / "data/raw/multi_harness_teacher_v1/trajectories.jsonl"
RAW_FILTER_AUDIT = EXP / "data/sft/multi_harness_trl_v1/audit.jsonl"
REFUSAL = re.compile(
    r"(?i)\b(i cannot|i can't|i won['’]?t|cannot help|can't help|unable to help|"
    r"won't assist|must decline|i refuse|decline to)\b|无法帮助|不能协助|拒绝"
)
ROLES = {"system", "user", "assistant", "tool"}


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSON at {path}:{line_no}")
            yield line_no, value


def tool_calls(messages: Any) -> list[tuple[int, dict[str, Any]]]:
    result = []
    if not isinstance(messages, list):
        return result
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            result.append((index, call))
    return result


def message_errors(row: dict[str, Any]) -> list[str]:
    messages = row.get("messages")
    meta = row.get("metadata") or {}
    errors: list[str] = []
    if not isinstance(messages, list) or len(messages) < 2:
        return ["missing_or_short_messages"]
    if not isinstance(messages[0], dict) or messages[0].get("role") != "system":
        errors.append("missing_initial_system")
    elif "<harness_context>" not in str(messages[0].get("content") or ""):
        errors.append("missing_model_visible_harness_context")
    roles = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ROLES:
            errors.append("invalid_message_or_role")
            continue
        roles.append(message["role"])
    if "user" not in roles or "assistant" not in roles:
        errors.append("missing_user_or_assistant")
    if roles and roles[-1] != "assistant":
        errors.append("final_message_not_assistant")
    if not meta.get("harness_name") or not meta.get("benchmark"):
        errors.append("missing_source_or_harness")
    if not meta.get("record_id") or not meta.get("family"):
        errors.append("missing_record_or_family_provenance")
    schemas = row.get("tools") or []
    schema_names = {
        str((item.get("function") or {}).get("name"))
        for item in schemas if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }

    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        if not calls:
            continue
        following_tools = []
        for candidate in messages[index + 1:]:
            if not isinstance(candidate, dict) or candidate.get("role") != "tool":
                break
            following_tools.append(candidate)
        result_ids = {str(m.get("tool_call_id")) for m in following_tools if m.get("tool_call_id") is not None}
        for call in calls:
            fn = call.get("function") if isinstance(call, dict) else None
            if not isinstance(fn, dict) or not fn.get("name"):
                errors.append("malformed_tool_call")
                continue
            if not schema_names or str(fn["name"]) not in schema_names:
                errors.append("tool_not_in_recorded_harness_schema")
            args = fn.get("arguments")
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                errors.append("malformed_tool_arguments")
                continue
            if not isinstance(args, dict):
                errors.append("non_object_tool_arguments")
            if not following_tools:
                errors.append("missing_tool_observation")
            elif call.get("id") is not None and str(call["id"]) not in result_ids:
                errors.append("tool_call_id_mismatch")
    return sorted(set(errors))


def classify(row: dict[str, Any], origin_split: str, source_path: str) -> tuple[str, list[str]]:
    meta = row.get("metadata") or {}
    bench = str(meta.get("benchmark") or "unknown").lower()
    source_kind = str(meta.get("source_kind") or "")
    calls = tool_calls(row.get("messages"))
    reasons = message_errors(row)

    if bool(meta.get("synthetic")) or "synthetic" in source_kind:
        return "synthetic_oracle_candidate", ["synthetic_not_in_primary_sft"]
    if bench == "harnessrisk":
        return "legacy_text_only_security", sorted(set(message_errors(row) + ["no_model_native_tool_call_trace"]))
    if reasons:
        return "rejected_incomplete_or_malformed", reasons

    category = ""
    include = True
    if bench == "agentdojo":
        task_id = str(meta.get("task_id") or "").lower()
        if task_id.endswith(":attack") or "attack" in task_id or "injection_task_" in task_id:
            category = "verified_safe_attack_defense"
        elif task_id.endswith(":clean") or "clean" in task_id:
            category = "verified_benign_tool_completion"
        else:
            category = "verified_agentdojo_tool_behavior"
        if not calls:
            include = False
            reasons = ["expected_native_tool_behavior_missing"]
    elif bench == "actbench":
        score = meta.get("score") or {}
        role = str(meta.get("role") or "").lower()
        if score.get("available") is not True or score.get("evaluation_error") is True or score.get("task_pass") is not True:
            return "rejected_actbench_outcome", ["official_task_pass_not_verified"]
        if role in {"clean", "benign_baseline"} and float(score.get("ugs", 0)) == 1.0:
            category = "verified_benign_tool_completion"
        elif role in {"attack", "adversarial"} and float(score.get("ags", 1)) == 0.0:
            category = "verified_safe_attack_defense"
        else:
            return "rejected_actbench_outcome", ["role_specific_safe_score_not_verified"]
        if not calls:
            include = False
            reasons = ["expected_native_tool_behavior_missing"]
    elif bench == "agentharm":
        task_type = str(meta.get("task_type") or "").lower()
        if task_type in {"clean", "benign"}:
            category = "verified_benign_tool_completion"
            if not calls:
                include = False
                reasons = ["benign_task_without_tool_call"]
        elif task_type in {"query", "harmful"}:
            messages = row.get("messages") or []
            answer = str(row.get("response") or (messages[-1].get("content") if messages and isinstance(messages[-1], dict) else "") or "")
            category = "verified_harmful_refusal"
            if not REFUSAL.search(answer):
                include = False
                reasons = ["missing_explicit_refusal"]
        else:
            include = False
            category = "rejected_unknown_task_type"
            reasons = ["unknown_agentharm_task_type"]
        if not meta.get("judge_backend") or not meta.get("judge_model"):
            include = False
            reasons = [*reasons, "missing_exact_judge_provenance"]
    else:
        include = False
        category = "rejected_unknown_benchmark"
        reasons = ["unknown_benchmark"]

    # The AgentHarm selector writes only official-pass traces to its accepted
    # view. For the historical sources, prior source-specific filter output is
    # the admission record. Never infer a pass from response text alone.
    if bench != "agentharm" and not source_kind.startswith(("real_teacher", "actbench_safe_real")):
        include = False
        reasons = [*reasons, "unapproved_or_unknown_source_kind"]
    if not include:
        return category or "rejected_source_quality", sorted(set(reasons))
    return category, []


def fingerprint(row: dict[str, Any]) -> str:
    payload = {"messages": row.get("messages"), "tools": row.get("tools") or []}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def actbench_row(source: dict[str, Any], index: int, split: str, source_path: Path) -> dict[str, Any]:
    meta = source.get("meta") or {}
    messages = json.loads(json.dumps(source.get("messages") or [], ensure_ascii=False))
    first = next((m for m in messages if m.get("role") == "system"), None)
    context = str((first or {}).get("content") or "")
    if "<harness_context>" in context and "</harness_context>" in context:
        context = context[context.index("<harness_context>"):context.index("</harness_context>") + len("</harness_context>")]
    task_id = str(meta.get("task_id") or "")
    harness = str(meta.get("harness") or "unknown")
    record_id = f"actbench:{harness}:{meta.get('trajectory_id') or task_id or index}"
    score = meta.get("score") or {}
    role = str(meta.get("role") or "")
    category = ("verified_benign_tool_completion" if role in {"clean", "benign_baseline"}
                else "verified_safe_attack_defense" if role in {"attack", "adversarial"}
                else "unclassified")
    return {
        "messages": messages, "tools": source.get("tools") or [],
        "metadata": {
            "record_id": record_id, "benchmark": "actbench", "harness_name": harness,
            "harness_context": context, "source_kind": "actbench_safe_real_trajectory",
            "synthetic": False, "task_id": task_id, "family": task_id,
            "teacher_model": meta.get("teacher_model"), "role": role,
            "score": score, "category": category, "split": split,
            "source_path": str(source_path),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--legacy-dir", type=Path, default=DEFAULT_LEGACY,
                   help="Existing source-filtered AgentDojo/ActBench/HarnessRisk view")
    p.add_argument("--agentharm-dir", type=Path, default=DEFAULT_AGENTHARM,
                   help="Officially verified AgentHarm native trajectories")
    p.add_argument("--actbench-dir", type=Path, default=ACTBENCH,
                   help="Source-filtered ActBench real trajectories with official score metadata")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--include-legacy-text-only", action="store_true",
                   help="Opt legacy HarnessRisk output-only rows into primary SFT")
    p.add_argument("--include-synthetic", action="store_true",
                   help="Opt synthetic oracle candidates into primary SFT")
    p.add_argument("--replace-output", action="store_true",
                   help="Replace files in the specified generated output directory")
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.replace_output:
        raise SystemExit(f"Output directory is nonempty; choose a new path or pass --replace-output: {args.output_dir}")
    rows: list[tuple[str, dict[str, Any], str, str]] = []
    # The old test split stays an audit-only holdout. Only its original train
    # and validation rows are eligible for the next SFT.
    for split in ("train", "validation", "test"):
        path = args.legacy_dir / f"{split}.jsonl"
        for line_no, row in read_jsonl(path):
            # Re-read ActBench from its score-preserving sanitized source below;
            # the older merged view discarded role-specific official scores.
            if str((row.get("metadata") or {}).get("benchmark") or "").lower() == "actbench":
                continue
            rows.append(("legacy:" + split, row, str(path), str(line_no)))
    for split in ("train", "validation"):
        path = args.actbench_dir / f"{split}.jsonl"
        for line_no, raw in read_jsonl(path):
            rows.append(("actbench:" + split, actbench_row(raw, line_no, split, path), str(path), str(line_no)))
    for split in ("train", "validation", "test"):
        path = args.agentharm_dir / f"{split}.jsonl"
        for line_no, row in read_jsonl(path):
            rows.append(("agentharm:" + split, row, str(path), str(line_no)))

    audit: list[dict[str, Any]] = []
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fingerprints: dict[str, tuple[str, str]] = {}
    family_targets: dict[tuple[str, str], str] = {}
    heldout_families: set[tuple[str, str]] = set()
    for source_split, row, _, _ in rows:
        meta = row.get("metadata") or {}
        key = (str(meta.get("benchmark") or "unknown"), str(meta.get("family") or "unknown"))
        original_split = source_split.rsplit(":", 1)[-1]
        if original_split == "test":
            heldout_families.add(key)
        elif key not in family_targets or original_split == "validation":
            # Conservatively move a mixed legacy train/validation family wholly
            # to validation so no same-family case appears on both sides.
            family_targets[key] = original_split
    seen_ids: set[str] = set()
    for source_split, row, source_path, line_no in rows:
        meta = row.setdefault("metadata", {})
        original_split = source_split.rsplit(":", 1)[-1]
        if source_split == "legacy:test" or source_split == "agentharm:test":
            category, reasons = classify(row, original_split, source_path)
            reasons = sorted(set(reasons + ["heldout_source_test_not_for_sft"]))
            eligible = False
        else:
            category, reasons = classify(row, original_split, source_path)
            eligible = not reasons
        if category == "legacy_text_only_security" and args.include_legacy_text_only:
            reasons = [reason for reason in reasons if reason != "no_model_native_tool_call_trace"]
            eligible = not reasons and original_split != "test"
        if category == "synthetic_oracle_candidate" and args.include_synthetic:
            eligible = True
            reasons = []
        benchmark = str(meta.get("benchmark") or "unknown")
        family = str(meta.get("family") or "unknown")
        split_key = (benchmark, family)
        if eligible:
            if split_key in heldout_families:
                eligible = False
                reasons = ["family_overlaps_heldout_test"]
            record_id = str(meta.get("record_id") or "")
            if record_id and record_id in seen_ids:
                eligible = False
                reasons = ["duplicate_record_id"]
            digest = fingerprint(row)
            if digest in fingerprints:
                eligible = False
                reasons = ["duplicate_trajectory_fingerprint"]
            if eligible:
                seen_ids.add(record_id)
                fingerprints[digest] = (benchmark, family)
                meta["category"] = category
                meta["split"] = family_targets.get(split_key, original_split)
                meta["joint_source_path"] = source_path
                meta["joint_source_line"] = int(line_no)
                output[meta["split"]].append(row)
        audit.append({
            "record_id": meta.get("record_id"), "benchmark": benchmark,
            "harness_name": meta.get("harness_name"), "source_kind": meta.get("source_kind"),
            "source_path": source_path, "source_line": int(line_no),
            "source_split": original_split, "category": category,
            "assistant_tool_calls": len(tool_calls(row.get("messages"))),
            "primary_sft_eligible": bool(eligible), "reasons": reasons,
        })

    # Account for raw multi-harness records already rejected by the existing
    # benchmark-aware filter. They remain in the audit ledger and never become
    # candidates merely because this joint pass is stricter/different.
    raw_line_by_id: dict[str, int] = {}
    rejected_source_ids: set[str] = set()
    for _, source_audit in read_jsonl(RAW_FILTER_AUDIT):
        if source_audit.get("accepted") is False:
            rejected_source_ids.add(str(source_audit.get("record_id") or ""))
    if rejected_source_ids and RAW_MULTIHARNESS.is_file():
        for line_no, raw in read_jsonl(RAW_MULTIHARNESS):
            rid = str(raw.get("record_id") or "")
            if rid in rejected_source_ids:
                raw_line_by_id[rid] = line_no
    rejected_by_id = {str(item.get("record_id") or ""): item for _, item in read_jsonl(RAW_FILTER_AUDIT)
                      if item.get("accepted") is False}
    for record_id, item in rejected_by_id.items():
        category = "synthetic_oracle_candidate" if item.get("synthetic") else "rejected_source_filter"
        audit.append({
            "record_id": record_id, "benchmark": item.get("benchmark"),
            "harness_name": item.get("harness_name"), "source_kind": item.get("source_kind"),
            "source_path": str(RAW_MULTIHARNESS), "source_line": raw_line_by_id.get(record_id),
            "source_split": "raw_source_rejected", "category": category,
            "assistant_tool_calls": None, "primary_sft_eligible": False,
            "reasons": ["rejected_by_existing_source_filter", *item.get("reasons", [])],
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        path = args.output_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output[split]), encoding="utf-8")
    audit_path = args.output_dir / "classification_audit.jsonl"
    audit_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in audit), encoding="utf-8")
    accepted_audit = [item for item in audit if item["primary_sft_eligible"]]
    rejected = [item for item in audit if not item["primary_sft_eligible"]]
    report = {
        "format": "cross_harness_joint_multidataset_sft_v1",
        "policy": {
            "primary_sft_requires": "source-specific verified success, complete assistant/tool observations where tools are expected, exact harness context and provenance",
            "agentharm_harmful_zero_tool_calls": "valid only when explicitly refused and previously admitted by AgentHarm official judge selection",
            "legacy_harnessrisk": "classified as text-only security; excluded from primary SFT by default",
            "synthetic": "separate class; excluded from primary SFT by default",
            "test_splits": "audit only; never copied to train or validation",
        },
        "rows_by_source_split": dict(Counter(x["source_split"] for x in audit)),
        "rows_by_category": dict(Counter(x["category"] for x in audit)),
        "eligible_by_category": dict(Counter(x["category"] for x in accepted_audit)),
        "eligible_by_split": {split: len(output[split]) for split in ("train", "validation")},
        "eligible_by_harness": {split: dict(Counter(str(r["metadata"].get("harness_name")) for r in output[split])) for split in ("train", "validation")},
        "eligible_by_benchmark_category": dict(Counter(f"{x['benchmark']}/{x['category']}" for x in accepted_audit)),
        "excluded_by_reason": dict(Counter(reason for x in rejected for reason in x["reasons"])),
        "input_dirs": {"legacy": str(args.legacy_dir), "actbench": str(args.actbench_dir), "agentharm": str(args.agentharm_dir)},
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted({Path(item[2]) for item in rows} | {RAW_MULTIHARNESS, RAW_FILTER_AUDIT}) if path.is_file()},
        "include_legacy_text_only": args.include_legacy_text_only,
        "include_synthetic": args.include_synthetic,
        "classification_audit": str(audit_path),
    }
    (args.output_dir / "joint_selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
