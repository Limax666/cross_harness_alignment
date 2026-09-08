#!/usr/bin/env python3
"""Strict rejection sampling and LLaMA-Factory export for Skill-use SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


IM_END_RE = re.compile(r"<\|im_end\|>\s*$")
EMPTY_THINK_RE = re.compile(r"^\s*<think>\s*</think>\s*", re.DOTALL)
OBSERVATION_RE = re.compile(r"</observation>", re.IGNORECASE)
QUERY_MARKERS = {
    "query",
    "query_target",
    "pure",
    "pure_attack",
    "pure_malicious",
    "malicious_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-size", type=int, default=5000)
    parser.add_argument("--min-size", type=int, default=3000)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--split-seed", default="safety-skillrl-sft-v1")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-response-length", type=int, default=8192)
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=0,
        help="Maximum trajectories per task; zero keeps all distinct trajectories.",
    )
    parser.add_argument(
        "--max-per-family",
        type=int,
        default=0,
        help="Maximum trajectories per family; zero disables family truncation.",
    )
    parser.add_argument("--require-runtime-skill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-legacy-context-reconstruction", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at {path}:{line_no}")
                row.setdefault("_source_file", str(path))
                rows.append(row)
    return rows


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def normalized(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def task_type(row: dict[str, Any]) -> str:
    meta = metadata(row)
    markers = {
        normalized(meta.get(key))
        for key in (
            "scenario",
            "task_type",
            "harm_task_type",
            "safety_eval_family",
            "task_category",
            "group_position",
        )
    }
    task_id = normalized(meta.get("task_id"))
    if (
        bool(meta.get("is_malicious_query_task"))
        or bool(markers & QUERY_MARKERS)
        or task_id.startswith("querytarget")
        or "query_malicious" in task_id
    ):
        return "query"
    if normalized(meta.get("scenario")) in {"attacked", "environment_injection", "injection"}:
        return "injection"
    return "clean"


def reward_values(row: dict[str, Any]) -> tuple[float | None, float | None]:
    meta = metadata(row)
    breakdown = meta.get("reward_breakdown")
    metrics = meta.get("env_metrics")
    breakdown = breakdown if isinstance(breakdown, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    utility = breakdown.get("utility_reward", metrics.get("utility_reward"))
    safety = breakdown.get("safety_reward", metrics.get("safety_reward"))
    return (
        float(utility) if isinstance(utility, (int, float, bool)) else None,
        float(safety) if isinstance(safety, (int, float, bool)) else None,
    )


def skill_condition(row: dict[str, Any]) -> str:
    meta = metadata(row)
    return normalized(meta.get("skillrl_sft_skill_condition") or "correct")


def echoed_rendered_skill_id(row: dict[str, Any]) -> bool:
    meta = metadata(row)
    rendered_ids = meta.get("skillrl_rendered_skill_ids")
    if not isinstance(rendered_ids, list):
        return False
    response = str(row.get("response") or "")
    return any(
        isinstance(skill_id, str) and skill_id.strip() and skill_id.strip() in response
        for skill_id in rendered_ids
    )


def rejection_reason(row: dict[str, Any], args: argparse.Namespace) -> str | None:
    meta = metadata(row)
    metrics = meta.get("env_metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    utility, safety = reward_values(row)
    kind = task_type(row)
    condition = skill_condition(row)
    checks = (
        (condition != "correct", "non_correct_skill_condition"),
        (normalized(row.get("status")) != "completed", "not_completed"),
        (bool(row.get("remove_sample")), "remove_sample"),
        (bool(metrics.get("parser_problem")), "parser_problem"),
        (int(metrics.get("invalid_tool_call_count") or 0) != 0, "invalid_tool_call"),
        (bool(metrics.get("hit_max_turns")), "hit_max_turns"),
        (metrics.get("explicit_finish") is not True, "no_explicit_finish"),
        (int(metrics.get("model_turn_count") or 0) > args.max_turns, "too_many_turns"),
        (int(row.get("response_length") or 0) > args.max_response_length, "response_too_long"),
        (bool(metrics.get("risk_success")), "risk_success"),
        (safety != 1.0, "safety_not_one"),
        (
            kind in {"injection", "clean"} and utility != 1.0,
            "utility_not_one",
        ),
        (
            kind in {"injection", "clean"} and metrics.get("final_success") is not True,
            "final_failure",
        ),
        (
            kind in {"injection", "clean"} and not meta.get("tool_history"),
            "missing_tool_history",
        ),
        (
            kind == "query" and bool(meta.get("tool_history")),
            "query_contains_tool_call",
        ),
        (echoed_rendered_skill_id(row), "rendered_skill_id_echo"),
    )
    for failed, reason in checks:
        if failed:
            return reason

    structure_reason = trajectory_structure_reason(row)
    if structure_reason:
        return structure_reason

    if args.require_runtime_skill:
        bank_sha256 = str(meta.get("skillrl_bank_sha256") or "").strip().lower()
        skill_text = str(meta.get("skillrl_injected_skill_text") or "").strip()
        rendered_ids = meta.get("skillrl_rendered_skill_ids")
        skill_checks = (
            (meta.get("skillrl_enabled") is not True, "skill_not_enabled"),
            (not skill_text, "empty_skill_text"),
            (
                not isinstance(rendered_ids, list)
                or not rendered_ids
                or not all(isinstance(item, str) and item.strip() for item in rendered_ids),
                "invalid_rendered_skill_ids",
            ),
            (bool(meta.get("skillrl_retrieval_error")), "skill_retrieval_error"),
            (
                len(bank_sha256) != 64
                or any(character not in "0123456789abcdef" for character in bank_sha256),
                "invalid_skill_bank_sha256",
            ),
            (not str(meta.get("skillrl_compiler_version") or "").strip(), "missing_compiler_version"),
            (not str(meta.get("skillrl_retrieval_version") or "").strip(), "missing_retrieval_version"),
            (not str(meta.get("skillrl_renderer_version") or "").strip(), "missing_renderer_version"),
            (normalized(meta.get("skillrl_sft_model_family")) not in {"qwen35", "qwen3"}, "invalid_model_family"),
        )
        for failed, reason in skill_checks:
            if failed:
                return reason

    if not args.allow_legacy_context_reconstruction:
        if not str(meta.get("skillrl_system_instruction") or "").strip():
            return "missing_exact_system_instruction"
        if not str(meta.get("skillrl_user_content") or "").strip():
            return "missing_exact_user_content"
        system = str(meta["skillrl_system_instruction"])
        user = str(meta["skillrl_user_content"])
        skill_text = str(meta.get("skillrl_injected_skill_text") or "").strip()
        injection_target = normalized(meta.get("skillrl_injection_target"))
        if injection_target == "user" and skill_text not in user:
            return "skill_missing_from_exact_user_content"
        if injection_target == "system" and skill_text not in system:
            return "skill_missing_from_exact_system_instruction"
        if injection_target not in {"user", "system"}:
            return "invalid_skill_injection_target"
    return None


def clean_assistant_content(value: Any) -> str:
    text = str(value or "").strip()
    text = EMPTY_THINK_RE.sub("", text)
    return IM_END_RE.sub("", text).strip()


def final_assistant_text(row: dict[str, Any]) -> str:
    response = str(row.get("response") or "")
    if not response:
        conversations = row.get("conversations")
        if isinstance(conversations, list):
            response = next(
                (
                    str(message.get("value") or "")
                    for message in reversed(conversations)
                    if isinstance(message, dict) and message.get("from") == "gpt"
                ),
                "",
            )
    if OBSERVATION_RE.search(response):
        response = OBSERVATION_RE.split(response)[-1]
    response = clean_assistant_content(response)
    if "<tool_call>" in response:
        return ""
    return response


def available_tool_names(meta: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    tools = meta.get("available_tools")
    if not isinstance(tools, list):
        return result
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            name = str(function.get("name") or "").strip()
            if name:
                result.add(name)
    return result


def trajectory_structure_reason(row: dict[str, Any]) -> str | None:
    meta = metadata(row)
    history = meta.get("tool_history")
    history = history if isinstance(history, list) else []
    known_tools = available_tool_names(meta)
    if history and not known_tools:
        return "missing_available_tools"
    for turn in history:
        if not isinstance(turn, dict):
            return "invalid_tool_history_entry"
        if bool(turn.get("invalid_call")):
            return "invalid_tool_call_flag"
        tool_name = str(turn.get("tool_name") or "").strip()
        if not tool_name:
            return "missing_tool_name"
        if tool_name not in known_tools:
            return "tool_not_in_available_schema"
        if not isinstance(turn.get("arguments"), dict):
            return "non_object_tool_arguments"
        if turn.get("observation") is None:
            return "missing_tool_observation"
    if not final_assistant_text(row):
        return "missing_final_assistant_response"
    return None


def system_and_user(row: dict[str, Any], allow_legacy: bool) -> tuple[str, str]:
    meta = metadata(row)
    system = str(meta.get("skillrl_system_instruction") or "").strip()
    user = str(meta.get("skillrl_user_content") or "").strip()
    if system and user:
        return system, user
    if not allow_legacy:
        raise ValueError(f"Missing exact rollout context for {row.get('id')}")

    system = (
        "You are solving a task in an external environment. Do not output your hidden reasoning or long planning text. "
        "Use tools when needed, calling at most one tool per turn. For references to memos, emails, files, pages, notes, "
        "records, or documents, inspect available tools before asking for details or saying you cannot access them. "
        "If a tool is needed, emit the tool call immediately. When you are done, provide a concise final answer."
    )
    query = str(meta.get("user_query") or "").strip()
    skill = str(meta.get("skillrl_injected_skill_text") or "").strip()
    target = normalized(meta.get("skillrl_injection_target"))
    if target == "system" and skill:
        system = f"{system}\n\n{skill}"
        user = query
    elif skill:
        user = f"{skill}\n\nUser task:\n{query}"
    else:
        user = query
    return system, user


def build_messages(row: dict[str, Any], allow_legacy: bool) -> list[dict[str, Any]]:
    meta = metadata(row)
    system, user = system_and_user(row, allow_legacy)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    history = meta.get("tool_history")
    history = history if isinstance(history, list) else []
    for index, turn in enumerate(history):
        if not isinstance(turn, dict):
            continue
        tool_name = str(turn.get("tool_name") or "").strip()
        if not tool_name:
            raise ValueError(f"Parse-failure turn survived filtering in {row.get('id')}")
        messages.append(
            {
                "role": "assistant",
                "content": clean_assistant_content(turn.get("assistant_content")),
                "tool_calls": [
                    {
                        "id": str(turn.get("tool_call_id") or f"call_{index}"),
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": turn.get("arguments") or {},
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": tool_name,
                "content": str(turn.get("observation") or ""),
            }
        )
    final_text = final_assistant_text(row)
    if not final_text:
        raise ValueError(f"No clean final assistant response in {row.get('id')}")
    messages.append({"role": "assistant", "content": final_text})
    return messages


def family_key(row: dict[str, Any]) -> str:
    meta = metadata(row)
    return str(meta.get("family_key") or meta.get("task_id") or row.get("id"))


def split_for(row: dict[str, Any], seed: str, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}\0{family_key(row)}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if value < validation_fraction else "train"


def candidate_rank(row: dict[str, Any]) -> tuple[int, int, int, str]:
    meta = metadata(row)
    metrics = meta.get("env_metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    return (
        int(metrics.get("model_turn_count") or 0),
        int(row.get("response_length") or 0),
        -int(row.get("step") or 0),
        str(row.get("id") or ""),
    )


def category(row: dict[str, Any]) -> str:
    return task_type(row)


def deterministic_order(row: dict[str, Any], seed: str) -> str:
    return hashlib.sha256(f"{seed}\0{row.get('id')}".encode()).hexdigest()


def provenance_values(rows: list[dict[str, Any]], key: str) -> list[str]:
    return sorted(
        {
            str(metadata(row).get(key) or "").strip()
            for row in rows
            if str(metadata(row).get(key) or "").strip()
        }
    )


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = metadata(row)
        by_task[(str(meta.get("task_id") or row.get("id")), skill_condition(row))].append(row)
    task_selected: list[dict[str, Any]] = []
    for candidates in by_task.values():
        ordered = sorted(candidates, key=candidate_rank)
        task_selected.extend(ordered[: args.max_per_task] if args.max_per_task > 0 else ordered)

    family_counts: Counter[str] = Counter()
    family_selected: list[dict[str, Any]] = []
    for row in sorted(task_selected, key=lambda item: deterministic_order(item, args.split_seed)):
        key = family_key(row)
        if args.max_per_family > 0 and family_counts[key] >= args.max_per_family:
            continue
        family_counts[key] += 1
        family_selected.append(row)

    target = args.target_size if args.target_size > 0 else len(family_selected)
    quota_weights = {"injection": 40, "query": 25, "clean": 25}
    total_weight = sum(quota_weights.values())
    quotas = {name: round(target * weight / total_weight) for name, weight in quota_weights.items()}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in family_selected:
        buckets[category(row)].append(row)
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for name, bucket in buckets.items():
        ordered = sorted(bucket, key=lambda item: deterministic_order(item, args.split_seed))
        take = quotas.get(name, 0)
        selected.extend(ordered[:take])
        leftovers.extend(ordered[take:])
    if len(selected) < target:
        selected.extend(
            sorted(leftovers, key=lambda item: deterministic_order(item, args.split_seed))[
                : target - len(selected)
            ]
        )
    return sorted(selected, key=lambda item: str(item.get("id") or ""))


def converted_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    meta = metadata(row)
    utility, safety = reward_values(row)
    messages = build_messages(row, args.allow_legacy_context_reconstruction)
    tools = meta.get("available_tools")
    if not isinstance(tools, list):
        tools = []
    # HuggingFace Datasets infers one Arrow schema for the complete JSONL.
    # Tool schemas and arbitrary argument objects are intentionally
    # heterogeneous, so keep both as canonical JSON strings. LLaMA-Factory's
    # OpenAI converter parses function.arguments back to a dict and accepts
    # the tools column as a JSON string.
    arrow_stable_messages: list[dict[str, Any]] = []
    for message in messages:
        stable_message = dict(message)
        if stable_message["role"] == "assistant":
            stable_calls = []
            for tool_call in stable_message.get("tool_calls") or []:
                stable_call = dict(tool_call)
                function = dict(stable_call["function"])
                arguments = function.get("arguments", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(
                        arguments,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                function["arguments"] = arguments
                stable_call["function"] = function
                stable_calls.append(stable_call)
            # Do not leave this key absent. Arrow would otherwise materialize
            # it as null on final assistant messages and LLaMA-Factory calls
            # len(message["tool_calls"]).
            stable_message["tool_calls"] = stable_calls
        arrow_stable_messages.append(stable_message)
    tools_json = json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    rollout_file = str(row.get("rollout_file") or "").strip()
    source_run = Path(rollout_file).parent.parent.name if rollout_file else "unknown_run"
    return {
        "id": f"{source_run}:{row.get('id')}",
        "messages": arrow_stable_messages,
        "tools": tools_json,
        "meta": {
            "schema": "safety_skillrl.skill_use_sft.v1",
            "source_file": row.get("_source_file"),
            "source_rollout_file": row.get("rollout_file"),
            "source_step": row.get("step"),
            "task_id": meta.get("task_id"),
            "family_key": family_key(row),
            "task_type": task_type(row),
            "category": category(row),
            "domain": meta.get("domain"),
            "difficulty_tier": meta.get("difficulty_tier"),
            "injection_after_turn": meta.get("injection_after_turn"),
            "skill_condition": skill_condition(row),
            "model_family": normalized(meta.get("skillrl_sft_model_family")),
            "skill_bank_path": meta.get("skillrl_bank_path"),
            "skill_bank_sha256": meta.get("skillrl_bank_sha256"),
            "runtime_compiler_version": meta.get("skillrl_compiler_version"),
            "retrieval_version": meta.get("skillrl_retrieval_version"),
            "renderer_version": meta.get("skillrl_renderer_version"),
            "data_source": meta.get("skillrl_sft_data_source") or "policy_rollout",
            "teacher_checkpoint": meta.get("skillrl_teacher_checkpoint") or "",
            "teacher_config_sha256": meta.get("skillrl_teacher_config_sha256") or "",
            "teacher_id": meta.get("teacher_id") or "",
            "teacher_provider": meta.get("teacher_provider") or "",
            "benchmark_id": meta.get("benchmark_id") or "",
            "benchmark_version": meta.get("benchmark_version") or "",
            "semantic_task_family_id": meta.get("semantic_task_family_id") or family_key(row),
            "harness_id": meta.get("harness_id") or "",
            "harness_family": meta.get("harness_family") or "",
            "harness_version": meta.get("harness_version") or "",
            "harness_sha256": meta.get("harness_sha256") or "",
            "safety_module": meta.get("safety_module") or "",
            "safety_module_sha256": meta.get("safety_module_sha256") or "",
            "retrieved_skill_ids": meta.get("skillrl_retrieved_skill_ids") or [],
            "rendered_skill_ids": meta.get("skillrl_rendered_skill_ids") or [],
            "rendered_skill_text": meta.get("skillrl_injected_skill_text") or "",
            "injection_target": normalized(meta.get("skillrl_injection_target")),
            "utility_reward": utility,
            "safety_reward": safety,
            "invalid_tool_call_count": (meta.get("env_metrics") or {}).get("invalid_tool_call_count"),
            "verifier_pass": True,
            "split": split_for(row, args.split_seed, args.validation_fraction),
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between zero and one")
    source_rows = read_jsonl(args.input)
    rejection_stats: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    for row in source_rows:
        reason = rejection_reason(row, args)
        if reason:
            rejection_stats[reason] += 1
        else:
            accepted.append(row)

    provenance = {
        "model_family": provenance_values(accepted, "skillrl_sft_model_family"),
        "skill_bank_sha256": provenance_values(accepted, "skillrl_bank_sha256"),
        "compiler_version": provenance_values(accepted, "skillrl_compiler_version"),
        "retrieval_version": provenance_values(accepted, "skillrl_retrieval_version"),
        "renderer_version": provenance_values(accepted, "skillrl_renderer_version"),
        "injection_target": provenance_values(accepted, "skillrl_injection_target"),
    }
    if args.require_runtime_skill:
        for key, values in provenance.items():
            if len(values) > 1:
                raise RuntimeError(f"Mixed {key} values in one SFT build: {values}")

    unique_rows: list[dict[str, Any]] = []
    converted_by_object_id: dict[int, dict[str, Any]] = {}
    conversion_rejections: Counter[str] = Counter()
    seen: set[str] = set()
    for row in accepted:
        try:
            item = converted_row(row, args)
        except (TypeError, ValueError) as exc:
            conversion_rejections[type(exc).__name__ + ":" + str(exc).split(" in ", 1)[0]] += 1
            continue
        digest = hashlib.sha256(
            json.dumps(
                {"messages": item["messages"], "tools": item["tools"]},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if digest in seen:
            conversion_rejections["exact_duplicate"] += 1
            continue
        seen.add(digest)
        unique_rows.append(row)
        converted_by_object_id[id(row)] = item

    selected = select_rows(unique_rows, args)
    converted = [converted_by_object_id[id(row)] for row in selected]

    train = [row for row in converted if row["meta"]["split"] == "train"]
    validation = [row for row in converted if row["meta"]["split"] == "validation"]
    train_families = {row["meta"]["family_key"] for row in train}
    validation_families = {row["meta"]["family_key"] for row in validation}
    if train_families & validation_families:
        raise RuntimeError("Family leakage between train and validation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "skill_use_sft_train.jsonl"
    validation_path = args.output_dir / "skill_use_sft_validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    dataset_info = {
        "safety_skill_use_sft_train": {
            "file_name": train_path.name,
            "formatting": "openai",
            "columns": {"messages": "messages", "tools": "tools"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "observation_tag": "tool",
                "function_tag": "function_call",
                "system_tag": "system",
            },
        },
        "safety_skill_use_sft_validation": {
            "file_name": validation_path.name,
            "formatting": "openai",
            "columns": {"messages": "messages", "tools": "tools"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "observation_tag": "tool",
                "function_tag": "function_call",
                "system_tag": "system",
            },
        },
    }
    (args.output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema": "safety_skillrl.skill_use_sft_build_report.v1",
        "inputs": [str(path) for path in args.input],
        "requirements": {
            "strict_verifier": True,
            "skill_condition": "correct_only",
            "require_runtime_skill": args.require_runtime_skill,
            "allow_legacy_context_reconstruction": args.allow_legacy_context_reconstruction,
            "max_turns": args.max_turns,
            "max_response_length": args.max_response_length,
            "max_per_task": args.max_per_task,
            "max_per_family": args.max_per_family,
            "target_size": args.target_size,
            "min_size": args.min_size,
        },
        "counts": {
            "source": len(source_rows),
            "accepted_before_quota": len(accepted),
            "unique_before_quota": len(unique_rows),
            "selected_before_conversion": len(selected),
            "converted": len(converted),
            "train": len(train),
            "validation": len(validation),
            "rejections": dict(sorted(rejection_stats.items())),
            "conversion_rejections": dict(sorted(conversion_rejections.items())),
        },
        "distribution": {
            "category": dict(sorted(Counter(row["meta"]["category"] for row in converted).items())),
            "task_type": dict(sorted(Counter(row["meta"]["task_type"] for row in converted).items())),
            "skill_condition": dict(
                sorted(Counter(row["meta"]["skill_condition"] for row in converted).items())
            ),
            "mean_tool_turns": (
                mean(
                    sum(1 for message in row["messages"] if message["role"] == "tool")
                    for row in converted
                )
                if converted
                else 0.0
            ),
        },
        "provenance": provenance,
        "source_rollout_dirs": sorted(
            {
                str(Path(str(row.get("rollout_file"))).parent)
                for row in source_rows
                if str(row.get("rollout_file") or "").strip()
            }
        ),
        "split": {
            "unit": "family_key",
            "seed": args.split_seed,
            "validation_fraction": args.validation_fraction,
            "family_overlap": len(train_families & validation_families),
        },
        "outputs": {
            "train": str(train_path),
            "validation": str(validation_path),
            "dataset_info": str(args.output_dir / "dataset_info.json"),
        },
    }
    (args.output_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    required_min_size = min(args.target_size, args.min_size) if args.target_size > 0 else args.min_size
    build_problem = ""
    if len(converted) < required_min_size:
        build_problem = f"only {len(converted)} examples survived; required at least {required_min_size}"
    elif not train:
        build_problem = "training split is empty"
    elif not validation:
        build_problem = "validation split is empty"
    if build_problem and not args.allow_empty:
        print(
            f"SFT build gate failed: {build_problem}.",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
