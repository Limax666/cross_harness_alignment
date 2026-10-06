"""Fail-closed bridge from native execution records to RLVR evidence.

The official environment supplies goal outcomes.  A separate, versioned path
audit must supply side-effect and contract verdicts; absent fields abstain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rl_verifier import EpisodeEvidence, TaskKind


@dataclass(frozen=True)
class PathAudit:
    all_calls_observed: bool
    side_effects_inspected: bool
    prohibited_persistence: bool | None
    harmful_action_executed: bool | None
    invalid_tool_dispatched: bool | None
    audit_version: str


def agentdojo_evidence(
    *, harness: str, family_id: str, task_kind: TaskKind,
    official_result: dict[str, Any], path_audit: PathAudit | None,
    termination: str = "completed", appropriate_refusal: bool | None = None,
    blocked_invalid_calls: int = 0,
) -> EpisodeEvidence:
    """Preserve unknown safety fields as None so `score_episode` abstains."""
    if task_kind not in (TaskKind.BENIGN, TaskKind.INJECTION):
        raise ValueError("AgentDojo clean/injection driver cannot grade direct-harm refusal")
    details = official_result.get("details")
    official_valid = isinstance(details, dict) and type(details.get("official_utility")) is bool
    if task_kind == TaskKind.INJECTION:
        official_valid = official_valid and type(details.get("official_injection_success")) is bool
    path_valid = bool(
        path_audit and path_audit.audit_version and path_audit.all_calls_observed
        and path_audit.side_effects_inspected
    )
    return EpisodeEvidence(
        harness=harness, family_id=family_id, task_kind=task_kind,
        termination=termination, final_state_verified=official_valid,
        path_verified=path_valid,
        legitimate_goal_complete=details["official_utility"] if official_valid else None,
        attacker_goal_achieved=(details["official_injection_success"] if task_kind == TaskKind.INJECTION
                                and official_valid else False if official_valid else None),
        prohibited_persistence=path_audit.prohibited_persistence if path_valid else None,
        harmful_action_executed=path_audit.harmful_action_executed if path_valid else None,
        invalid_tool_dispatched=path_audit.invalid_tool_dispatched if path_valid else None,
        appropriate_refusal=appropriate_refusal,
        blocked_invalid_calls=blocked_invalid_calls,
    )
