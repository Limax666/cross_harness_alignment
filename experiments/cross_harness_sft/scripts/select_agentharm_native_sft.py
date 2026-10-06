#!/usr/bin/env python3
"""Select auditable, native AgentHarm teacher traces for a future SFT run.

Reads collection outputs without modifying them.  The split is by original
behavior family across every harness, so prompt variants cannot cross splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from filter_multiharness_to_trl_sft import sanitize_text

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/cross_harness_sft"
ERROR_OBSERVATION = re.compile(r"(?i)\b(error|exception|traceback|not found|failed|permission denied|invalid)\b")
REFUSAL = re.compile(r"(?i)\b(i can(?:not|'t)|i won'?t|cannot help|can't help|unable to help|won't assist|must decline|i refuse)\b|无法帮助|不能协助|拒绝")


def read_rows(path: Path, snapshot: bytes):
    for line_number, line in enumerate(snapshot.decode("utf-8").splitlines(), 1):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {path}:{line_number}") from exc


def split_for_family(family: str) -> str:
    # AgentHarm IDs are <original behavior>-<augmentation>. Keep all variants
    # and both benign/harmful arms of an original behavior in one split.
    original = family.split("-", 1)[0]
    bucket = int(hashlib.sha256(f"agentharm-family-v1:{original}".encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def quality_reasons(row: dict[str, Any], allowed_judges: set[str]) -> list[str]:
    meta = row.get("metadata") or {}
    metrics = meta.get("env_metrics") or {}
    verdict = meta.get("verifier_details") or {}
    messages = row.get("messages") or []
    audit = meta.get("tool_history")
    reasons: list[str] = []
    if row.get("schema") != "cross_harness.native_trajectory.v2" or row.get("status") != "completed":
        reasons.append("incomplete_native_run")
    if not str(meta.get("benchmark_version", "")).endswith("agentharm/test_public"):
        reasons.append("wrong_source_split")
    if not meta.get("harness_id") or not meta.get("task_id") or not meta.get("family_key"):
        reasons.append("missing_provenance")
    if verdict.get("judge_backend") not in allowed_judges:
        reasons.append("unapproved_judge")
    if metrics.get("final_success") is not True or metrics.get("risk_success") is not False:
        reasons.append("unsafe_or_unsuccessful")
    if metrics.get("parser_problem") or metrics.get("hit_max_turns") or not metrics.get("explicit_finish"):
        reasons.append("truncated_or_parser_error")
    if metrics.get("invalid_tool_call_count") or not isinstance(audit, list):
        reasons.append("invalid_tool_audit")
        audit = []
    if not isinstance(messages, list) or len(messages) < 3 or not isinstance(messages[0], dict):
        return sorted(set(reasons + ["incomplete_messages"]))
    system = str(messages[0].get("content") or "")
    if messages[0].get("role") != "system" or "<harness_context>" not in system:
        reasons.append("missing_harness_context")
    if messages[1].get("role") != "user" or not str(messages[1].get("content") or "").strip():
        reasons.append("missing_user_prompt")
    if messages[-1].get("role") != "assistant" or not str(messages[-1].get("content") or "").strip():
        reasons.append("missing_final_answer")
    if not isinstance(meta.get("available_tools"), list) or not meta["available_tools"]:
        reasons.append("missing_native_tool_schema")
    names = {str((t.get("function") or {}).get("name")) for t in meta.get("available_tools") or [] if isinstance(t, dict)}
    calls = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            reasons.append("invalid_message"); continue
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or not function.get("name"):
                reasons.append("malformed_tool_call"); continue
            try:
                arguments = function.get("arguments")
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                if not isinstance(arguments, dict): raise ValueError
            except (ValueError, TypeError):
                reasons.append("malformed_tool_arguments")
            if function["name"] not in names:
                reasons.append("tool_not_in_harness_schema")
            if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool":
                reasons.append("missing_tool_observation")
            elif messages[index + 1].get("tool_call_id") != call.get("id"):
                reasons.append("tool_call_id_mismatch")
            calls.append(function["name"])
    if len(calls) != len(audit) or [a.get("tool_name") for a in audit] != calls:
        reasons.append("tool_audit_mismatch")
    if any(a.get("invalid_call") or a.get("guard_decision") != "allow" for a in audit):
        reasons.append("invalid_or_denied_tool_call")
    if any(ERROR_OBSERVATION.search(str(a.get("observation") or "")) for a in audit):
        reasons.append("tool_execution_error")
    task_type = meta.get("task_type")
    if task_type == "clean":
        if not calls: reasons.append("benign_without_tool_behavior")
    elif task_type == "query":
        if verdict.get("official_refusal") is not True or not REFUSAL.search(str(row.get("response") or "")):
            reasons.append("harmful_without_explicit_refusal")
        if calls and float(verdict.get("official_mean_score", 1)) != 0:
            reasons.append("harmful_tool_effect")
    else:
        reasons.append("unknown_task_type")
    return sorted(set(reasons))


def resolved_judge_model(verdict: dict[str, Any], legacy_map: dict[str, str]) -> tuple[str, str]:
    recorded = str(verdict.get("judge_model") or "").strip()
    if recorded:
        return recorded, "recorded"
    backend = str(verdict.get("judge_backend") or "")
    inferred = str(legacy_map.get(backend) or "").strip()
    return inferred, "legacy_backend_default" if inferred else "missing"


def convert(row: dict[str, Any], judge_model: str, judge_model_provenance: str) -> dict[str, Any]:
    meta = row["metadata"]
    redactions = 0
    def clean(value: Any) -> Any:
        nonlocal redactions
        if isinstance(value, str):
            result, n = sanitize_text(value)
            redactions += n
            return result
        if isinstance(value, list): return [clean(item) for item in value]
        if isinstance(value, dict): return {key: clean(item) for key, item in value.items()}
        return value
    messages = clean(row["messages"])
    tools = clean(meta["available_tools"])
    system = str(messages[0]["content"])
    end = system.index("</harness_context>") + len("</harness_context>")
    # The VeRL role-segment template does not inject its `tools` column. Put
    # each case's real native tool contract in the system context it consumes.
    messages[0]["content"] = system + "\n\n<available_tools>\n" + json.dumps(
        tools, ensure_ascii=False, sort_keys=True) + "\n</available_tools>"
    harness = str(meta["harness_id"])
    return {"messages": messages, "tools": tools, "metadata": {
        "record_id": str(row["id"]), "benchmark": "agentharm", "harness_name": harness,
        "harness_context": system[system.index("<harness_context>"):end],
        "source_kind": "agentharm_native_verified", "synthetic": False,
        "task_id": meta["task_id"], "family": meta["family_key"],
        "task_type": meta["task_type"], "teacher_model": meta.get("teacher_id"),
        "judge_backend": meta["verifier_details"]["judge_backend"],
        "judge_model": judge_model, "judge_model_provenance": judge_model_provenance,
        "source_path": str(meta.get("benchmark_version")), "redaction_count": redactions,
        "tool_audit_count": len(meta["tool_history"]),
        "split": split_for_family(str(meta["family_key"])),
    }}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=EXP / "outputs")
    parser.add_argument("--output-dir", type=Path, default=EXP / "data/sft/agentharm_native_verified_v1")
    parser.add_argument("--allowed-judges", default="codex_cli,openai_compatible", help="Comma-separated approved judge backends")
    parser.add_argument("--judge-models", default="gpt-5.6-sol,ali/qwen3.8-flash", help="Comma-separated approved judge model IDs")
    parser.add_argument("--legacy-judge-model-map", default="codex_cli=gpt-5.6-sol,openai_compatible=ali/qwen3.8-flash",
                        help="Explicit backend-to-model mapping for older records lacking judge_model")
    parser.add_argument("--min-benign-tools-per-harness", type=int, default=20)
    parser.add_argument("--min-harmful-per-harness", type=int, default=20)
    parser.add_argument("--min-harnesses", type=int, default=3)
    args = parser.parse_args()
    allowed = set(args.allowed_judges.split(","))
    allowed_models = {value.strip() for value in args.judge_models.split(",") if value.strip()}
    legacy_judge_map = dict(item.split("=", 1) for item in args.legacy_judge_model_map.split(",") if "=" in item)
    paths = sorted(args.input_dir.glob("agentharm*/test_public*.jsonl"))
    paths = [p for p in paths if ".errors." not in p.name]
    if not paths:
        raise SystemExit("No AgentHarm native collection JSONL files found")
    # Capture one coherent byte snapshot while collectors may still append.
    snapshots = {path: path.read_bytes() for path in paths}
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit: list[dict[str, Any]] = []
    accepted_keys: set[tuple[str, str]] = set()
    for path in paths:
        for row in read_rows(path, snapshots[path]):
            meta = row.get("metadata") or {}
            verdict = meta.get("verifier_details") or {}
            judge_model, judge_model_provenance = resolved_judge_model(verdict, legacy_judge_map)
            reasons = quality_reasons(row, allowed)
            if judge_model not in allowed_models:
                reasons.append("unverified_judge_model")
            key = (str(meta.get("harness_id")), str(meta.get("task_id")))
            if key in accepted_keys: reasons.append("duplicate_harness_task")
            event = {"id": row.get("id"), "harness": key[0], "task_id": key[1],
                     "source_file": str(path), "reasons": reasons, "accepted": not reasons}
            audit.append(event)
            if reasons: continue
            accepted_keys.add(key)
            converted = convert(row, judge_model, judge_model_provenance)
            by_split[converted["metadata"]["split"]].append(converted)
    # Admit only matched benign/harmful variants within each harness and split.
    # This teaches request-intent discrimination under the same interface.
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for split, rows in by_split.items():
        for row in rows:
            groups[(split, row["metadata"]["harness_name"], row["metadata"]["task_type"])].append(row)
    selected_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_ids: set[str] = set()
    pair_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        for harness in sorted({h for s, h, _ in groups if s == split}):
            benign = {str(r["metadata"]["family"]): r for r in groups[(split, harness, "clean")]}
            harmful = {str(r["metadata"]["family"]): r for r in groups[(split, harness, "query")]}
            paired = sorted(benign.keys() & harmful.keys())
            if split == "train": pair_counts[harness] = len(paired)
            for family in paired:
                for row in (benign[family], harmful[family]):
                    selected_by_split[split].append(row)
                    selected_ids.add(str(row["metadata"]["record_id"]))
    for event in audit:
        if event["accepted"] and str(event["id"]) not in selected_ids:
            event["accepted"] = False
            event["reasons"] = ["unmatched_benign_harmful_pair"]
    by_split = selected_by_split
    for split in ("train", "validation", "test"):
        by_split[split].sort(key=lambda r: str(r["metadata"]["record_id"]))
    raw_counts = Counter({(h, t): len(rows) for (s, h, t), rows in groups.items() if s == "train"})
    counts = Counter((r["metadata"]["harness_name"], r["metadata"]["task_type"])
                     for r in by_split["train"])
    harnesses = {h for h, _ in counts}
    ready = len(harnesses) >= args.min_harnesses and len(by_split["validation"]) >= 8 and all(
        pair_counts[h] >= max(args.min_benign_tools_per_harness, args.min_harmful_per_harness)
        for h in harnesses)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        (args.output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in by_split[split]), encoding="utf-8")
    (args.output_dir / "audit.jsonl").write_text(
        "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audit), encoding="utf-8")
    report = {"source_files": [str(p) for p in paths],
              "source_sha256": {str(p): hashlib.sha256(snapshots[p]).hexdigest() for p in paths},
              "allowed_judges": sorted(allowed), "allowed_judge_models": sorted(allowed_models),
              "legacy_judge_model_map": legacy_judge_map,
              "judge_model_provenance": "Recorded metadata is preferred; missing legacy values use the explicitly configured backend map.",
              "eligible_before_pairing": sum(len(v) for v in groups.values()),
              "accepted": sum(len(v) for v in by_split.values()),
              "by_split": {s: len(by_split[s]) for s in ("train", "validation", "test")},
              "train_by_harness_type": {f"{h}/{t}": n for (h, t), n in sorted(counts.items())},
              "eligible_train_by_harness_type": {f"{h}/{t}": n for (h, t), n in sorted(raw_counts.items())},
              "matched_train_pairs_by_harness": pair_counts,
              "rejected_by_reason": dict(Counter(reason for a in audit for reason in a["reasons"])),
              "training_ready": ready,
              "minimums": {"harnesses": args.min_harnesses, "benign_tool_traces_per_harness": args.min_benign_tools_per_harness,
                           "harmful_refusals_per_harness": args.min_harmful_per_harness},
              "evaluation_warning": "test_public is used for training; do not report the official AgentHarm test_public aggregate as held-out evaluation"}
    (args.output_dir / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
