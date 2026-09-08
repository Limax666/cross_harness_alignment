#!/usr/bin/env python3
"""Shared SkillBank and rollout-memory utilities for Safety-SkillRL.

This module is intentionally dependency-light so it can run in the same
environments as the existing `scripts/python` helpers.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SKILL_BANK_SCHEMA = "safety_skillrl.skill_bank.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def slug(value: Any, fallback: str = "skill") -> str:
    text = norm(value)
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:96] or fallback


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: str | os.PathLike[str], value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(value, tmp, ensure_ascii=False, indent=2, sort_keys=False)
        tmp.write("\n")
    try:
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def iter_input_files(paths: Iterable[str]) -> Iterable[Path]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            yield from sorted([*path.rglob("*.jsonl"), *path.rglob("*.pt")])
        elif path.exists():
            yield path


def iter_rollout_objects(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".pt":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(f"Reading rollout debug .pt files requires torch: {path}") from exc
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            yield data
        return
    yield from iter_jsonl(path)


def metadata_from_obj(obj: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield metadata dicts from flexible rollout/debug JSONL shapes."""

    if isinstance(obj.get("metadata"), dict):
        meta = dict(obj["metadata"])
        if "reward" not in meta and isinstance(obj.get("reward"), (int, float, bool)):
            meta["reward"] = float(obj["reward"])
        if "response" not in meta and isinstance(obj.get("response"), str):
            meta["response"] = obj["response"]
        yield meta

    prompt = obj.get("prompt")
    if isinstance(prompt, dict) and isinstance(prompt.get("metadata"), dict):
        yield dict(prompt["metadata"])
    elif isinstance(prompt, list):
        for item in prompt:
            if isinstance(item, dict) and isinstance(item.get("metadata"), dict):
                yield dict(item["metadata"])

    if any(key in obj for key in ("env_metrics", "reward_breakdown", "evaluation", "tool_history")):
        yield dict(obj)

    samples = obj.get("samples")
    if isinstance(samples, list):
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            metadata = sample.get("metadata")
            if not isinstance(metadata, dict):
                continue
            meta = dict(metadata)
            reward = sample.get("reward")
            if "reward" not in meta and isinstance(reward, (int, float, bool)):
                meta["reward"] = float(reward)
            response = sample.get("response")
            if "response" not in meta and isinstance(response, str):
                meta["response"] = response
            status = sample.get("status")
            if status is not None:
                meta["sample_status"] = status
            yield meta


def load_rollout_metadata(paths: Iterable[str], limit: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_input_files(paths):
        for obj in iter_rollout_objects(path):
            for meta in metadata_from_obj(obj):
                records.append(meta)
                if limit > 0 and len(records) >= limit:
                    return records
    return records


def env_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    value = metadata.get("env_metrics")
    return value if isinstance(value, dict) else {}


def reward_breakdown(metadata: dict[str, Any]) -> dict[str, Any]:
    value = metadata.get("reward_breakdown")
    return value if isinstance(value, dict) else {}


def evaluation(metadata: dict[str, Any]) -> dict[str, Any]:
    value = metadata.get("evaluation")
    return value if isinstance(value, dict) else {}


def tool_history(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in as_list(metadata.get("tool_history")) if isinstance(item, dict)]


def domain(metadata: dict[str, Any]) -> str:
    for key in ("domain", "scenario_domain", "task_domain"):
        value = norm(metadata.get(key))
        if value:
            return value
    task_meta = metadata.get("task_metadata")
    if isinstance(task_meta, dict):
        return norm(task_meta.get("domain") or task_meta.get("scenario")) or "any"
    return "any"


def attack_type(metadata: dict[str, Any]) -> str:
    detected = norm(metadata.get("detected_attack_type"))
    if detected:
        return detected
    detected_types = [norm(value) for value in as_list(metadata.get("detected_attack_types"))]
    detected_types = [value for value in detected_types if value]
    if detected_types:
        return detected_types[0]
    task_id = norm(metadata.get("task_id"))
    for marker in ("important_instructions", "ignore_previous", "direct", "environment_injection"):
        if marker in task_id:
            return marker
    if is_attacked(metadata):
        return "environment_injection"
    return "any"


def metadata_markers(metadata: dict[str, Any]) -> set[str]:
    markers: set[str] = set()
    for key in (
        "scenario",
        "task_type",
        "harm_task_type",
        "safety_eval_family",
        "task_category",
        "category",
        "score_family",
        "group_position",
    ):
        for value in as_list(metadata.get(key)):
            marker = norm(value)
            if marker:
                markers.add(marker)
    task_meta = metadata.get("task_metadata")
    if isinstance(task_meta, dict):
        for key in ("task_type", "harm_task_type", "safety_eval_family", "domain"):
            marker = norm(task_meta.get(key))
            if marker:
                markers.add(marker)
    return markers


def is_attacked(metadata: dict[str, Any]) -> bool:
    if bool(metadata.get("is_malicious_query_task")):
        return True
    markers = metadata_markers(metadata)
    if markers & {
        "attacked",
        "attack",
        "safety",
        "environment_injection",
        "injection",
        "pure",
        "pure_malicious",
        "pure_attack",
        "malicious_only",
        "query_target",
        "query",
    }:
        return True
    task_id = norm(metadata.get("task_id"))
    return (
        any("attack" in marker or "inject" in marker for marker in markers)
        or task_id.startswith("querytarget")
        or "query_malicious" in task_id
        or "querytarget" in task_id
    )


def scenario(metadata: dict[str, Any]) -> str:
    return "attacked" if is_attacked(metadata) else "clean"


def reward_value(metadata: dict[str, Any]) -> float:
    for key in ("reward", "raw_reward", "env_reward", "sample_reward"):
        value = metadata.get(key)
        if isinstance(value, (int, float, bool)):
            return float(value)
    metrics = env_metrics(metadata)
    for key in ("sample_reward", "env_reward"):
        value = metrics.get(key)
        if isinstance(value, (int, float, bool)):
            return float(value)
    rb = reward_breakdown(metadata)
    value = rb.get("reward")
    if isinstance(value, (int, float, bool)):
        return float(value)
    return 0.0


def success_label(metadata: dict[str, Any]) -> bool:
    metrics = env_metrics(metadata)
    rb = reward_breakdown(metadata)
    ev = evaluation(metadata)
    reward = reward_value(metadata)
    final_success = bool(metrics.get("final_success")) or bool(ev.get("success")) or reward >= 0.999
    risk_info = ev.get("risk_info") if isinstance(ev.get("risk_info"), dict) else {}
    risk_success = bool(metrics.get("risk_success")) or bool(risk_info.get("success"))
    safety_reward = rb.get("safety_reward")
    safety_ok = not isinstance(safety_reward, (int, float, bool)) or float(safety_reward) > 0.0
    return final_success and not risk_success and safety_ok


def has_injection(metadata: dict[str, Any]) -> bool:
    if attack_type(metadata) not in {"any", "environment_injection"}:
        return True
    traces = metadata.get("injection_trace")
    if isinstance(traces, list) and traces:
        return True
    for item in tool_history(metadata):
        if item.get("has_injection_observation") or item.get("detected_attack_type"):
            return True
        observation = str(item.get("observation") or "").lower()
        if any(token in observation for token in ("important message", "ignore previous", "system instruction")):
            return True
    return False


def failure_buckets(metadata: dict[str, Any]) -> list[str]:
    metrics = env_metrics(metadata)
    rb = reward_breakdown(metadata)
    buckets: list[str] = []
    if metrics.get("parser_problem") or int(metrics.get("parse_failure_count") or 0) > 0:
        buckets.append("parser_error")
    if int(metrics.get("invalid_tool_call_count") or 0) > 0:
        buckets.append("invalid_tool_call")
    if metrics.get("risk_success"):
        buckets.append("unsafe_outcome")
    safety_reward = rb.get("safety_reward")
    if isinstance(safety_reward, (int, float, bool)) and float(safety_reward) <= 0.0:
        buckets.append("unsafe_outcome")
    if metrics.get("hit_max_turns"):
        buckets.append("max_turns")
    if metrics.get("positive_step_but_final_failure"):
        buckets.append("positive_step_but_final_failure")
    if not metrics.get("final_success") and int(metrics.get("tool_call_count") or 0) == 0:
        buckets.append("no_tool_final")
    if not metrics.get("final_success") and int(metrics.get("tool_call_count") or 0) > 0:
        seq = [norm(item.get("tool_name")) for item in tool_history(metadata) if item.get("tool_name")]
        if len(seq) >= 3 and len(set(seq[-3:])) == 1:
            buckets.append("repeated_no_progress_tool")
    if is_attacked(metadata) and not metrics.get("final_success") and not metrics.get("risk_success"):
        utility_reward = rb.get("utility_reward")
        if isinstance(utility_reward, (int, float, bool)) and float(utility_reward) <= 0.0:
            buckets.append("safe_but_incomplete")
    if has_injection(metadata):
        buckets.append("environment_injection")
    if not buckets and not success_label(metadata):
        buckets.append("premature_final")
    deduped: list[str] = []
    for bucket in buckets:
        if bucket not in deduped:
            deduped.append(bucket)
    return deduped


def tool_sequence(metadata: dict[str, Any], limit: int = 8) -> list[str]:
    seq: list[str] = []
    for item in tool_history(metadata):
        name = norm(item.get("tool_name"))
        if name and (not seq or seq[-1] != name):
            seq.append(name)
        if len(seq) >= limit:
            break
    return seq


def tool_names(metadata: dict[str, Any]) -> list[str]:
    names = [norm(item.get("tool_name")) for item in tool_history(metadata) if item.get("tool_name")]
    for item in as_list(metadata.get("selected_tools")):
        if isinstance(item, str):
            names.append(norm(item))
    for item in as_list(metadata.get("available_tools")):
        if isinstance(item, dict):
            fn = item.get("function") if isinstance(item.get("function"), dict) else {}
            name = fn.get("name")
            if name:
                names.append(norm(name))
        elif isinstance(item, str):
            names.append(norm(item))
    return [name for name in names if name]


def cluster_key(metadata: dict[str, Any]) -> tuple[str, str, str]:
    return (scenario(metadata), domain(metadata), attack_type(metadata) if is_attacked(metadata) else "any")


def group_records(records: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[cluster_key(record)].append(record)
    return dict(grouped)


def summarize_cluster(key: tuple[str, str, str], records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [record for record in records if success_label(record)]
    failures = [record for record in records if not success_label(record)]
    rewards = [reward_value(record) for record in records]
    bucket_counts = Counter(bucket for record in failures for bucket in failure_buckets(record))
    tool_counts = Counter(tool for record in records for tool in tool_names(record))
    return {
        "scenario": key[0],
        "domain": key[1],
        "attack_type": key[2],
        "num_records": len(records),
        "num_success": len(successes),
        "num_failure": len(failures),
        "success_rate": len(successes) / max(1, len(records)),
        "mean_reward": sum(rewards) / max(1, len(rewards)),
        "failure_buckets": dict(bucket_counts.most_common(8)),
        "top_tools": dict(tool_counts.most_common(8)),
    }


def select_update_clusters(
    records: list[dict[str, Any]],
    update_threshold: float,
    min_failures: int,
    max_clusters: int,
) -> list[dict[str, Any]]:
    summaries = []
    grouped = group_records(records)
    for key, items in grouped.items():
        summary = summarize_cluster(key, items)
        if int(summary["num_failure"]) < min_failures:
            continue
        if float(summary["success_rate"]) > update_threshold:
            continue
        summary["records"] = items
        summaries.append(summary)
    summaries.sort(key=lambda item: (int(item["num_failure"]), -float(item["success_rate"])), reverse=True)
    return summaries[:max_clusters]


def empty_skill_bank(source: str = "") -> dict[str, Any]:
    return {
        "schema": SKILL_BANK_SCHEMA,
        "metadata": {
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "source": source,
            "update_history": [],
        },
        "general_skills": [],
        "task_specific_skills": {},
        "common_mistakes": [],
    }


def normalize_skill_bank(data: Any, source: str = "") -> dict[str, Any]:
    if not isinstance(data, dict):
        return empty_skill_bank(source=source)
    bank = dict(data)
    bank.setdefault("schema", SKILL_BANK_SCHEMA)
    metadata = bank.get("metadata") if isinstance(bank.get("metadata"), dict) else {}
    metadata = dict(metadata)
    metadata.setdefault("created_at_utc", utc_now())
    metadata["updated_at_utc"] = metadata.get("updated_at_utc") or utc_now()
    metadata.setdefault("source", source)
    metadata.setdefault("update_history", [])
    bank["metadata"] = metadata
    bank["general_skills"] = [normalize_skill(skill, kind="general") for skill in as_list(bank.get("general_skills")) if isinstance(skill, dict)]
    raw_task = bank.get("task_specific_skills") if isinstance(bank.get("task_specific_skills"), dict) else {}
    task_specific: dict[str, list[dict[str, Any]]] = {}
    for dom, skills in raw_task.items():
        task_specific[norm(dom) or "any"] = [
            normalize_skill(skill, kind="task", domain=norm(dom) or "any")
            for skill in as_list(skills)
            if isinstance(skill, dict)
        ]
    bank["task_specific_skills"] = task_specific
    bank["common_mistakes"] = [
        normalize_skill(skill, kind="mistake")
        for skill in as_list(bank.get("common_mistakes"))
        if isinstance(skill, dict)
    ]
    return bank


def load_skill_bank(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if not path:
        return empty_skill_bank()
    bank_path = Path(path)
    if not bank_path.exists():
        return empty_skill_bank(source=str(bank_path))
    return normalize_skill_bank(read_json(bank_path), source=str(bank_path))


def normalize_text_list(value: Any, max_items: int, max_chars: int = 240) -> list[str]:
    items: list[str] = []
    for item in as_list(value):
        text = " ".join(str(item or "").split())
        if not text:
            continue
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        if text not in items:
            items.append(text)
        if len(items) >= max_items:
            break
    return items


def normalize_skill(skill: dict[str, Any], kind: str, domain: str = "any") -> dict[str, Any]:
    name = slug(skill.get("name") or skill.get("title") or skill.get("skill_id"), fallback=f"{kind}_skill")
    principle = " ".join(str(skill.get("principle") or skill.get("summary") or skill.get("description") or skill.get("guidance") or skill.get("rule") or "").split())
    when_to_apply = " ".join(str(skill.get("when_to_apply") or skill.get("when_to_use") or skill.get("trigger") or skill.get("applicability") or skill.get("scenario") or "").split())
    result = dict(skill)
    result["name"] = name
    result.setdefault("skill_id", f"{kind}_{name}")
    result["principle"] = principle[:500]
    result["when_to_apply"] = when_to_apply[:300]
    result["workflow"] = normalize_text_list(
        skill.get("workflow") or skill.get("steps") or skill.get("rules") or skill.get("repair_strategy") or skill.get("repair"),
        6,
    )
    result["avoid"] = normalize_text_list(skill.get("avoid") or skill.get("common_mistakes"), 4)
    evidence = skill.get("evidence") if isinstance(skill.get("evidence"), dict) else {}
    result["evidence"] = dict(evidence)
    result.setdefault("source", skill.get("source") or "unknown")
    if kind == "task":
        result["domain"] = norm(skill.get("domain") or domain) or "any"
    if kind == "mistake":
        result["bucket"] = norm(skill.get("bucket") or skill.get("mistake_id") or name)
    return result


def skill_signature(skill: dict[str, Any], kind: str, domain_name: str = "any") -> tuple[str, str, str]:
    principle = slug(skill.get("principle") or skill.get("summary") or "")
    return (kind, norm(domain_name), slug(skill.get("name") or principle))


def merge_skill_list(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    kind: str,
    domain_name: str = "any",
) -> tuple[list[dict[str, Any]], int, int]:
    by_sig = {skill_signature(skill, kind, domain_name): dict(skill) for skill in existing}
    added = 0
    updated = 0
    for raw_skill in incoming:
        skill = normalize_skill(raw_skill, kind=kind, domain=domain_name)
        sig = skill_signature(skill, kind, domain_name)
        if sig not in by_sig:
            by_sig[sig] = skill
            added += 1
            continue
        merged = by_sig[sig]
        merged["evidence"] = {**(merged.get("evidence") or {}), **(skill.get("evidence") or {})}
        for field in ("workflow", "avoid"):
            merged[field] = normalize_text_list(as_list(merged.get(field)) + as_list(skill.get(field)), 8)
        if not merged.get("when_to_apply") and skill.get("when_to_apply"):
            merged["when_to_apply"] = skill["when_to_apply"]
        merged["source"] = skill.get("source") or merged.get("source")
        by_sig[sig] = merged
        updated += 1
    return list(by_sig.values()), added, updated


def merge_skill_update(bank: dict[str, Any], update: dict[str, Any], history: dict[str, Any]) -> dict[str, Any]:
    bank = normalize_skill_bank(bank)
    added_total = 0
    updated_total = 0

    general, added, updated = merge_skill_list(
        bank.get("general_skills", []),
        [item for item in as_list(update.get("general_skills")) if isinstance(item, dict)],
        kind="general",
    )
    bank["general_skills"] = general
    added_total += added
    updated_total += updated

    incoming_task = update.get("task_specific_skills") if isinstance(update.get("task_specific_skills"), dict) else {}
    task_bank = bank.get("task_specific_skills") if isinstance(bank.get("task_specific_skills"), dict) else {}
    for raw_domain, skills in incoming_task.items():
        dom = norm(raw_domain) or "any"
        merged, added, updated = merge_skill_list(
            task_bank.get(dom, []),
            [item for item in as_list(skills) if isinstance(item, dict)],
            kind="task",
            domain_name=dom,
        )
        task_bank[dom] = merged
        added_total += added
        updated_total += updated
    bank["task_specific_skills"] = task_bank

    mistakes, added, updated = merge_skill_list(
        bank.get("common_mistakes", []),
        [item for item in as_list(update.get("common_mistakes")) if isinstance(item, dict)],
        kind="mistake",
    )
    bank["common_mistakes"] = mistakes
    added_total += added
    updated_total += updated

    bank["metadata"]["updated_at_utc"] = utc_now()
    event = dict(history)
    event.update({"added_skills": added_total, "updated_skills": updated_total, "updated_at_utc": utc_now()})
    bank["metadata"].setdefault("update_history", []).append(event)
    return bank


def _compact_skill_context(metadata: dict[str, Any], max_chars: int = 900) -> dict[str, Any]:
    skill_ids = [
        str(item)
        for item in as_list(metadata.get("skillrl_retrieved_skill_ids"))
        if str(item).strip()
    ][:12]
    skill_text = " ".join(str(metadata.get("skillrl_injected_skill_text") or "").split())
    injection_target = str(metadata.get("skillrl_injection_target") or "").strip()
    context: dict[str, Any] = {}
    if skill_ids:
        context["retrieved_skill_ids"] = skill_ids
    if injection_target:
        context["injection_target"] = injection_target
    if skill_text:
        context["injected_skill_text_preview"] = skill_text[:max_chars]
    return context


def compact_trajectory(metadata: dict[str, Any], max_observation_chars: int = 220) -> dict[str, Any]:
    metrics = env_metrics(metadata)
    rb = reward_breakdown(metadata)
    steps = []
    for entry in tool_history(metadata)[:8]:
        steps.append(
            {
                "turn": entry.get("turn_index"),
                "tool": entry.get("tool_name"),
                "arguments": entry.get("arguments"),
                "observation_preview": " ".join(str(entry.get("observation") or "").split())[:max_observation_chars],
                "invalid_call": bool(entry.get("invalid_call")),
                "detected_attack_type": entry.get("detected_attack_type"),
                "risk_success_after_step": bool(entry.get("risk_success_after_step")),
            }
        )
    result = {
        "task_id": metadata.get("task_id"),
        "scenario": scenario(metadata),
        "domain": domain(metadata),
        "attack_type": attack_type(metadata),
        "user_query": str(metadata.get("user_query") or "")[:500],
        "success": success_label(metadata),
        "reward": reward_value(metadata),
        "failure_buckets": failure_buckets(metadata),
        "env_metrics": {
            key: metrics.get(key)
            for key in (
                "final_success",
                "risk_success",
                "tool_call_count",
                "parse_failure_count",
                "invalid_tool_call_count",
                "hit_max_turns",
                "positive_step_but_final_failure",
            )
        },
        "reward_breakdown": {
            key: rb.get(key)
            for key in ("utility_reward", "safety_reward", "risk_success", "source")
        },
        "tool_sequence": tool_sequence(metadata, limit=8),
        "steps": steps,
    }
    skill_context = _compact_skill_context(metadata)
    if skill_context:
        result["skill_context"] = skill_context
    return result


def sample_cluster_examples(
    records: list[dict[str, Any]],
    max_successes: int,
    max_failures: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successes = [record for record in records if success_label(record)]
    failures = [record for record in records if not success_label(record)]
    successes.sort(key=reward_value, reverse=True)
    failures.sort(key=reward_value)
    return (
        [compact_trajectory(record) for record in successes[:max_successes]],
        [compact_trajectory(record) for record in failures[:max_failures]],
    )


def bank_counts(bank: dict[str, Any]) -> dict[str, int]:
    task_count = sum(len(skills) for skills in (bank.get("task_specific_skills") or {}).values())
    return {
        "general_skills": len(bank.get("general_skills") or []),
        "task_specific_domains": len(bank.get("task_specific_skills") or {}),
        "task_specific_skills": task_count,
        "common_mistakes": len(bank.get("common_mistakes") or []),
    }
