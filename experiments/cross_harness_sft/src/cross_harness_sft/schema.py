from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Case:
    benchmark_id: str
    task_id: str
    family_id: str
    task_type: str
    domain: str
    user_query: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    initial_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Case":
        return cls(
            benchmark_id=str(row["benchmark_id"]),
            task_id=str(row["task_id"]),
            family_id=str(row.get("semantic_task_family_id") or row["task_id"]),
            task_type=str(row.get("task_type") or "clean"),
            domain=str(row.get("domain") or "any"),
            user_query=str(row["user_query"]),
            tools=list(row.get("tools") or []),
            initial_state=dict(row.get("initial_state") or {}),
            metadata=dict(row.get("metadata") or {}),
        )


@dataclass
class ToolResult:
    observation: str
    result: Any = None
    invalid_call: bool = False
    side_effect: bool = False
    risk_event: bool = False


@dataclass
class UnifiedVerdict:
    utility_reward: float
    safety_reward: float
    final_success: bool
    risk_success: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantTurn:
    content: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    raw: dict[str, Any]
    request_id: str = ""
    provider: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0

