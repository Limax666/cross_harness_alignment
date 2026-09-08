#!/usr/bin/env python3
"""Runtime retrieval/rendering for evolved and compiled Safety-SkillRL SkillBanks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from bank_utils import as_list, attack_type, domain, failure_buckets, is_attacked, load_skill_bank, norm, slug


_SCOPE_ANY = {"any", "all", "global", "*"}
_QUERY_MARKERS = {"pure", "pure_malicious", "pure_attack", "malicious_only", "query_target", "query"}
_ATTACK_MARKERS = {"attacked", "attack", "safety", "environment_injection", "injection"}
_RUNTIME_SCHEMA = "safety_skillrl.runtime.v1"
_RETRIEVAL_VERSION = "runtime_skill_retrieval.v1"
_RENDERER_VERSION = "runtime_skill_renderer.v1"


def _text_score(text: str, query: str) -> float:
    text_terms = set(norm(text).split("_"))
    query_terms = set(norm(query).split("_"))
    if not text_terms or not query_terms:
        return 0.0
    return len(text_terms & query_terms) / max(1, len(query_terms))


def _skill_text(skill: dict[str, Any]) -> str:
    parts = [
        str(skill.get("name") or ""),
        str(skill.get("principle") or ""),
        str(skill.get("when_to_apply") or ""),
        str(skill.get("trigger") or ""),
        " ".join(str(item) for item in as_list(skill.get("workflow"))),
        " ".join(str(item) for item in as_list(skill.get("action"))),
        " ".join(str(item) for item in as_list(skill.get("avoid"))),
    ]
    return " ".join(part for part in parts if part)


def _is_dynamic_skill(skill: dict[str, Any]) -> bool:
    if isinstance(skill.get("is_dynamic"), bool):
        return bool(skill["is_dynamic"])
    skill_id = str(skill.get("skill_id") or skill.get("id") or "")
    source = str(skill.get("source") or "")
    return skill_id.startswith("dyn_") or source.startswith(("strong_model", "offline_template"))


def _is_accepted_dynamic_skill(skill: dict[str, Any]) -> bool:
    if not _is_dynamic_skill(skill):
        return False
    return bool(skill.get("accepted_at_iteration") or skill.get("accepted_operation_id") or skill.get("accepted"))


def _is_runtime_bank(bank: dict[str, Any]) -> bool:
    return bank.get("schema") == _RUNTIME_SCHEMA and isinstance(bank.get("skills"), list)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _metadata_markers(metadata: dict[str, Any]) -> set[str]:
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
        for key in ("task_type", "harm_task_type", "safety_eval_family", "domain", "scenario"):
            marker = norm(task_meta.get(key))
            if marker:
                markers.add(marker)
    return markers


def _task_kind(metadata: dict[str, Any]) -> str:
    markers = _metadata_markers(metadata)
    task_id = norm(metadata.get("task_id"))
    if (
        bool(metadata.get("is_malicious_query_task"))
        or markers & _QUERY_MARKERS
        or task_id.startswith("querytarget")
        or "query_malicious" in task_id
        or "querytarget" in task_id
    ):
        return "query"
    if markers & _ATTACK_MARKERS or "attack" in task_id or "inject" in task_id:
        return "attacked"
    return "clean"


def _scope(skill: dict[str, Any]) -> dict[str, Any]:
    value = skill.get("scope")
    return value if isinstance(value, dict) else {}


def _scope_values(skill: dict[str, Any], key: str) -> set[str]:
    values = {norm(item) for item in as_list(_scope(skill).get(key))}
    return {item for item in values if item and item not in _SCOPE_ANY}


def _has_scope(skill: dict[str, Any]) -> bool:
    return any(_scope_values(skill, key) for key in ("domains", "scenarios", "attack_types", "failure_buckets"))


def _domain_aliases(metadata: dict[str, Any]) -> set[str]:
    current = domain(metadata)
    aliases = {current} if current and current != "any" else set()
    task_meta = metadata.get("task_metadata")
    if isinstance(task_meta, dict):
        for key in ("domain", "scenario"):
            value = norm(task_meta.get(key))
            if value and value != "any":
                aliases.add(value)
    return aliases


def _scenario_aliases(metadata: dict[str, Any]) -> set[str]:
    kind = _task_kind(metadata)
    if kind == "query":
        aliases = {"query", "pure", "pure_malicious", "pure_attack", "malicious_only", "query_target", "harmful_query"}
    elif kind == "attacked":
        aliases = {"attacked", "attack", "safety", "environment_injection", "injection", "indirect_prompt_injection"}
    else:
        aliases = {"clean", "benign", "utility", "normal"}
    marker_scenario = norm(metadata.get("scenario"))
    if marker_scenario and marker_scenario in aliases:
        aliases.add(marker_scenario)
    return aliases


def _attack_aliases(metadata: dict[str, Any]) -> set[str]:
    kind = _task_kind(metadata)
    if kind == "query":
        return {"query", "harmful_query", "pure", "pure_malicious", "pure_attack", "malicious_only", "query_target"}
    if kind == "clean":
        return {"clean", "none", "no_attack", "benign"}
    current = attack_type(metadata)
    aliases = {"attacked", "attack", "environment_injection", "injection", "indirect_prompt_injection"}
    if current and current != "any":
        aliases.add(current)
    return aliases


def _matches(scope_values: set[str], aliases: set[str]) -> bool:
    return not scope_values or bool(scope_values & aliases)


def _available_tool_names(metadata: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for item in as_list(metadata.get("available_tools")):
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if isinstance(function, dict):
            value = norm(function.get("name"))
        else:
            value = norm(item.get("name"))
        if value:
            names.add(value)
    return names


def _scope_allows_skill(skill: dict[str, Any], metadata: dict[str, Any], kind: str, skill_domain: str = "any") -> bool:
    """Constrain scoped skills before scoring so narrow dynamic skills cannot go global."""

    current_domain = domain(metadata)
    normalized_skill_domain = norm(skill_domain)
    if kind == "task" and normalized_skill_domain not in {"", "any"} and normalized_skill_domain != current_domain:
        return False

    dynamic = _is_dynamic_skill(skill)
    scoped = _has_scope(skill)
    if dynamic and _env_flag("SLIME_SAFETY_SKILLRL_STRICT_DYNAMIC_SCOPE", "1") and not scoped:
        return False
    if not scoped:
        return True

    domain_scope = _scope_values(skill, "domains")
    scenario_scope = _scope_values(skill, "scenarios")
    attack_scope = _scope_values(skill, "attack_types")

    if not _matches(domain_scope, _domain_aliases(metadata)):
        return False
    if not _matches(scenario_scope, _scenario_aliases(metadata)):
        return False
    if not _matches(attack_scope, _attack_aliases(metadata)):
        return False

    if kind == "mistake" and (domain_scope or scenario_scope or attack_scope):
        matched_key_scope = (
            bool(domain_scope & _domain_aliases(metadata))
            or bool(scenario_scope & _scenario_aliases(metadata))
            or bool(attack_scope & _attack_aliases(metadata))
        )
        if not matched_key_scope:
            return False
    return True


def _runtime_scope_allows_skill(skill: dict[str, Any], metadata: dict[str, Any]) -> bool:
    kind = norm(skill.get("kind"))
    skill_domain = norm(skill.get("domain")) or "any"
    if not _scope_allows_skill(skill, metadata, kind, skill_domain):
        return False

    task_types = _scope_values(skill, "task_types")
    if task_types and not task_types & _metadata_markers(metadata):
        return False
    tools = _scope_values(skill, "tools")
    if tools and not tools & _available_tool_names(metadata):
        return False
    return True


def _score_skill(skill: dict[str, Any], metadata: dict[str, Any], kind: str, skill_domain: str = "any") -> float:
    if not _scope_allows_skill(skill, metadata, kind, skill_domain):
        return 0.0

    score = 0.0
    query = " ".join(
        str(metadata.get(key) or "")
        for key in ("user_query", "task_id", "scenario", "task_type", "safety_eval_family")
    )
    query = f"{query} {domain(metadata)} {attack_type(metadata)}"
    score += _text_score(_skill_text(skill), query)
    if kind == "general":
        score += 0.6
        if is_attacked(metadata):
            score += 0.4
    elif kind == "task":
        if norm(skill_domain) == domain(metadata):
            score += 1.2
        elif norm(skill_domain) == "any":
            score += 0.4
    elif kind == "mistake":
        buckets = set(failure_buckets(metadata))
        bucket = norm(skill.get("bucket"))
        if bucket and bucket in buckets:
            score += 1.2
        score += 0.3
    scoped_buckets = _scope_values(skill, "failure_buckets")
    current_buckets = set(failure_buckets(metadata))
    if scoped_buckets and current_buckets and scoped_buckets & current_buckets:
        score += 0.4
    if _is_dynamic_skill(skill):
        score += float(os.getenv("SLIME_SAFETY_SKILLRL_DYNAMIC_BONUS", "0.5"))
    if _is_accepted_dynamic_skill(skill):
        score += float(os.getenv("SLIME_SAFETY_SKILLRL_ACCEPTED_DYNAMIC_BONUS", "0.0"))
    return score


def _runtime_score_skill(skill: dict[str, Any], metadata: dict[str, Any]) -> float:
    if not _runtime_scope_allows_skill(skill, metadata):
        return 0.0
    kind = norm(skill.get("kind"))
    skill_domain = norm(skill.get("domain")) or "any"
    query = " ".join(
        str(metadata.get(key) or "")
        for key in ("user_query", "task_id", "scenario", "task_type", "safety_eval_family")
    )
    score = _text_score(_skill_text(skill), f"{query} {domain(metadata)} {attack_type(metadata)}")
    if kind == "general":
        score += 0.6
        if is_attacked(metadata):
            score += 0.4
    elif kind == "task":
        score += 1.2 if skill_domain == domain(metadata) else 0.4
    elif kind == "mistake":
        buckets = _scope_values(skill, "failure_buckets")
        current_buckets = set(failure_buckets(metadata))
        score += 1.2 if buckets and current_buckets and buckets & current_buckets else 0.3
    try:
        score += max(0.0, float(skill.get("priority", 1.0))) * 0.1
    except (TypeError, ValueError):
        score += 0.1
    return score


def _top(scored: list[tuple[float, dict[str, Any]]], k: int) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)
    return [skill for score, skill in scored if score > 0][:k]


def _top_dynamic(scored: list[tuple[float, dict[str, Any]]], k: int) -> list[dict[str, Any]]:
    return _top([(score, skill) for score, skill in scored if _is_dynamic_skill(skill)], k)


def _runtime_top(scored: list[tuple[float, dict[str, Any]]], k: int) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    scored.sort(key=lambda item: (-item[0], str(item[1].get("id") or "")))
    return [skill for score, skill in scored if score > 0][:k]


def retrieve_runtime_skills(
    metadata: dict[str, Any],
    bank: dict[str, Any],
    *,
    general_topk: int = 1,
    task_topk: int = 2,
    mistake_topk: int = 1,
    total_topk: int = 4,
    only_attacked: bool = False,
) -> list[dict[str, Any]]:
    if not _is_runtime_bank(bank):
        raise ValueError(f"Expected {_RUNTIME_SCHEMA} SkillBank")
    if only_attacked and not is_attacked(metadata):
        return []
    by_kind: dict[str, list[tuple[float, dict[str, Any]]]] = {
        "general": [],
        "task": [],
        "mistake": [],
    }
    for skill in bank["skills"]:
        if not isinstance(skill, dict):
            continue
        kind = norm(skill.get("kind"))
        if kind not in by_kind or not bool(skill.get("accepted", True)):
            continue
        by_kind[kind].append((_runtime_score_skill(skill, metadata), skill))

    selected = [
        *_runtime_top(by_kind["general"], general_topk),
        *_runtime_top(by_kind["task"], task_topk),
        *_runtime_top(by_kind["mistake"], mistake_topk),
    ]
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for skill in selected:
        skill_id = str(skill.get("id") or "")
        if not skill_id or skill_id in seen:
            continue
        seen.add(skill_id)
        deduped.append(skill)
    return deduped[: max(0, total_topk)]


def retrieve_skills(
    metadata: dict[str, Any],
    bank: dict[str, Any],
    *,
    general_topk: int = 1,
    task_topk: int = 1,
    mistake_topk: int = 1,
    dynamic_task_topk: int = 0,
    dynamic_mistake_topk: int = 0,
    only_attacked: bool = False,
) -> list[dict[str, Any]]:
    if only_attacked and not is_attacked(metadata):
        return []
    selected: list[dict[str, Any]] = []
    selected.extend(
        _top(
            [(_score_skill(skill, metadata, "general"), skill) for skill in as_list(bank.get("general_skills")) if isinstance(skill, dict)],
            general_topk,
        )
    )
    task = bank.get("task_specific_skills") if isinstance(bank.get("task_specific_skills"), dict) else {}
    task_candidates: list[tuple[float, dict[str, Any]]] = []
    for raw_domain, skills in task.items():
        for skill in as_list(skills):
            if isinstance(skill, dict):
                task_candidates.append((_score_skill(skill, metadata, "task", str(raw_domain)), skill))
    selected.extend(_top(task_candidates, task_topk))
    selected.extend(_top_dynamic(task_candidates, dynamic_task_topk))
    mistake_candidates = [
        (_score_skill(skill, metadata, "mistake"), skill)
        for skill in as_list(bank.get("common_mistakes"))
        if isinstance(skill, dict)
    ]
    selected.extend(
        _top(mistake_candidates, mistake_topk)
    )
    selected.extend(_top_dynamic(mistake_candidates, dynamic_mistake_topk))

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for skill in selected:
        key = str(skill.get("skill_id") or skill.get("name") or slug(skill.get("principle")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(skill)
    return deduped


def render_skills(
    skills: list[dict[str, Any]],
    max_chars: int = 900,
    max_rule_chars: int = 180,
    dynamic_rule_chars: int | None = None,
) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    dynamic_rule_chars = dynamic_rule_chars if dynamic_rule_chars is not None else max_rule_chars
    for skill in skills:
        items = [skill.get("principle"), skill.get("when_to_apply")]
        items.extend(as_list(skill.get("workflow"))[:2])
        items.extend(as_list(skill.get("avoid"))[:1])
        for item in items:
            text = " ".join(str(item or "").split())
            if not text or text in seen:
                continue
            rule_chars = dynamic_rule_chars if _is_dynamic_skill(skill) else max_rule_chars
            if len(text) > rule_chars:
                text = text[: rule_chars - 3].rstrip() + "..."
            candidate = f"- {text}"
            if sum(len(line) + 1 for line in lines) + len(candidate) > max_chars:
                break
            lines.append(candidate)
            seen.add(text)
    if not lines:
        return ""
    return "Relevant safety skills:\n" + "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


def render_runtime_skills(
    skills: list[dict[str, Any]],
    *,
    max_total_chars: int = 900,
    max_skill_chars: int = 220,
    max_trigger_chars: int = 80,
    max_action_chars: int = 110,
    max_avoid_chars: int = 70,
) -> tuple[str, list[str], list[str]]:
    blocks: list[str] = []
    rendered_ids: list[str] = []
    dropped_ids: list[str] = []
    for index, skill in enumerate(skills, start=1):
        skill_id = str(skill.get("id") or "")
        trigger = _truncate(str(skill.get("trigger") or ""), max_trigger_chars)
        actions = [_truncate(str(item), max_action_chars) for item in as_list(skill.get("action")) if str(item).strip()]
        action = "; ".join(actions[:2])
        avoid = _truncate(str(skill.get("avoid") or ""), max_avoid_chars)
        header = f"[S{index}|{norm(skill.get('kind'))}|{norm(skill.get('domain')) or 'any'}|{skill_id}]"
        lines = [header]
        if trigger:
            lines.append(f"WHEN: {trigger}")
        if action:
            lines.append(f"DO: {action}")
        if avoid:
            lines.append(f"AVOID: {avoid}")
        block = "\n".join(lines)
        if len(block) > max_skill_chars:
            fallback = _truncate(" ".join(part for part in (trigger, action) if part), max_skill_chars - len(header) - 2)
            block = f"{header}\n{fallback}".rstrip()
        candidate = "\n\n".join([*blocks, block])
        if len(candidate) > max_total_chars:
            dropped_ids.append(skill_id)
            continue
        blocks.append(block)
        rendered_ids.append(skill_id)
    if not blocks:
        return "", [], dropped_ids
    return "Relevant safety skills:\n" + "\n\n".join(blocks), rendered_ids, dropped_ids


def retrieve_and_render(metadata: dict[str, Any], bank_path: str | os.PathLike[str] | None = None) -> tuple[str, list[str]]:
    path = str(bank_path or os.getenv("SLIME_SAFETY_SKILLRL_BANK_PATH") or "").strip()
    if not path:
        return "", []
    bank = load_skill_bank(path)
    if _is_runtime_bank(bank):
        skills = retrieve_runtime_skills(
            metadata,
            bank,
            general_topk=int(os.getenv("SLIME_SAFETY_SKILLRL_GENERAL_TOPK", "1")),
            task_topk=int(os.getenv("SLIME_SAFETY_SKILLRL_TASK_TOPK", "2")),
            mistake_topk=int(os.getenv("SLIME_SAFETY_SKILLRL_MISTAKE_TOPK", "1")),
            total_topk=int(os.getenv("SLIME_SAFETY_SKILLRL_TOTAL_TOPK", "4")),
            only_attacked=os.getenv("SLIME_SAFETY_SKILLRL_ONLY_ATTACKED", "0").lower()
            in {"1", "true", "yes", "on"},
        )
        text, rendered_ids, _ = render_runtime_skills(
            skills,
            max_total_chars=int(os.getenv("SLIME_SAFETY_SKILLRL_MAX_CHARS", "900")),
            max_skill_chars=int(os.getenv("SLIME_SAFETY_SKILLRL_MAX_SKILL_CHARS", "220")),
            max_trigger_chars=int(os.getenv("SLIME_SAFETY_SKILLRL_MAX_TRIGGER_CHARS", "80")),
            max_action_chars=int(os.getenv("SLIME_SAFETY_SKILLRL_MAX_ACTION_CHARS", "110")),
            max_avoid_chars=int(os.getenv("SLIME_SAFETY_SKILLRL_MAX_AVOID_CHARS", "70")),
        )
        return text, rendered_ids
    skills = retrieve_skills(
        metadata,
        bank,
        general_topk=int(os.getenv("SLIME_SAFETY_SKILLRL_GENERAL_TOPK", "1")),
        task_topk=int(os.getenv("SLIME_SAFETY_SKILLRL_TASK_TOPK", "1")),
        mistake_topk=int(os.getenv("SLIME_SAFETY_SKILLRL_MISTAKE_TOPK", "1")),
        dynamic_task_topk=int(os.getenv("SLIME_SAFETY_SKILLRL_DYNAMIC_TASK_TOPK", "0")),
        dynamic_mistake_topk=int(os.getenv("SLIME_SAFETY_SKILLRL_DYNAMIC_MISTAKE_TOPK", "0")),
        only_attacked=os.getenv("SLIME_SAFETY_SKILLRL_ONLY_ATTACKED", "0").lower() in {"1", "true", "yes", "on"},
    )
    text = render_skills(
        skills,
        max_chars=int(os.getenv("SLIME_SAFETY_SKILLRL_MAX_CHARS", "900")),
        dynamic_rule_chars=int(os.getenv("SLIME_SAFETY_SKILLRL_DYNAMIC_RULE_CHARS", "180")),
    )
    ids = [str(skill.get("skill_id") or skill.get("name") or "") for skill in skills]
    return text, ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--metadata-json", required=True, help="Metadata JSON string or path to a JSON file.")
    args = parser.parse_args()
    raw = args.metadata_json
    if Path(raw).exists():
        metadata = json.loads(Path(raw).read_text(encoding="utf-8"))
    else:
        metadata = json.loads(raw)
    text, ids = retrieve_and_render(metadata, args.bank)
    print(json.dumps({"skill_ids": ids, "text": text}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
