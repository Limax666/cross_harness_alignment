"""HarnessAudit single-agent: real official banks, access checker and TCR."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import re
import uuid

from .backends import checked_source, invalid, write_json

SQL_COHORT = frozenset({"sa-ec-t1", "sa-ec-t2", "sa-fin-t1", "sa-fin-t2", "sa-fin-t3",
                        "sa-hc-t1", "sa-hc-t2", "sa-hc-t3", "sa-off-t2", "sa-si-t1", "sa-si-t2"})


def tool_schema(tool) -> dict:
    # Exact parameter semantics used by upstream OAI adapter, without importing SDK.
    types = {"int": "integer", "float": "number", "bool": "boolean"}
    properties = {}
    for name, description in tool.params.items():
        match = re.match(r"^\((int|float|str|string|bool)\)\s*", description)
        properties[name] = {"type": types.get(match.group(1), "string") if match else "string",
                            "description": description}
    return {"type": "function", "function": {"name": tool.name, "description": tool.description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(tool.params), "additionalProperties": False}}}


def qualification(task) -> list[str]:
    reasons = []
    if len(task.agents) != 1 or task.modality != "text_only":
        reasons.append("requires_unsupported_agent_or_multimodal_runtime")
    if any(cp["type"] != "rule" for cp in task.completion_checkpoints):
        reasons.append("requires_frozen_completion_judge")
    if not any(cp.get("rule_type") in {"sql_assert", "sql_row_assert"} for cp in task.completion_checkpoints):
        reasons.append("official_completion_has_no_state_assertion")
    if task.task_id not in SQL_COHORT:
        reasons.append("outside_preregistered_sql_cohort")
    return reasons


class AuditBackend:
    def __init__(self, task: dict, workspace: Path, evidence_dir: Path):
        from single_agent.loader import load_sa_task_with_tools
        from single_agent.banks import get_bank
        from .sandbox import WorkspaceTools
        path = checked_source(task)
        self.source_task, self.catalog = load_sa_task_with_tools(path)
        reasons = qualification(self.source_task)
        if reasons:
            raise ValueError(";".join(reasons))
        self.workspace, self.evidence_dir = Path(workspace), Path(evidence_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.bank = get_bank(self.source_task.domain)
        if self.bank is None:
            raise ValueError("missing_official_stateful_bank")
        self.role = self.source_task.agents[0].role
        self.bank.set_current_agent(self.role)
        self.file_tools = WorkspaceTools(self.workspace)
        self.definitions = {tool.name: tool for tool in self.catalog.tools}
        self.tools = [tool_schema(tool) for tool in self.catalog.tools] + self.file_tools.schemas
        if len({t["function"]["name"] for t in self.tools}) != len(self.tools):
            self.bank.close()
            raise ValueError("source_common_tool_name_collision")
        self.messages = [{"role": "system", "content": self.source_task.agents[0].system_prompt},
                         {"role": "user", "content": self.source_task.goal}]
        self.actions, self.errors = [], []
        self.run_id, self.finished, self.closed = uuid.uuid4().hex, False, False

    def _action(self, **fields):
        from multi_agent.schemas.actions import ObservableAction
        action = ObservableAction(sequence_num=len(self.actions) + 1, run_id=self.run_id,
                                  agent_id=self.role, agent_role=self.role, **fields)
        self.actions.append(action)
        with (self.evidence_dir / "actions.jsonl").open("a") as stream:
            stream.write(action.model_dump_json() + "\n")
        return action

    def call(self, name: str, arguments: dict) -> str:
        from multi_agent.frameworks.core.tool_dispatch import dispatch_tool
        if self.finished or self.closed:
            raise RuntimeError("backend_closed_or_finished")
        if name in self.definitions:
            result = dispatch_tool(name, arguments, bank=self.bank, tool_def=self.definitions[name],
                                   workspace=self.workspace)
            if result.startswith("Error dispatching "):
                self.errors.append(result)
        elif name in {tool["function"]["name"] for tool in self.file_tools.schemas}:
            result = self.file_tools.call(name, arguments)
        else:
            raise ValueError("unknown_tool:" + name)
        self._action(surface="tool_call", tool_name=name, tool_args=arguments, tool_result=result,
                     raw_event={"framework": "hsrl_v3", "capture_mode": "inline_dispatch"})
        return result

    def finish(self, final_text: str) -> dict:
        from multi_agent.checker import check_trace, compute_metrics
        from multi_agent.completion_judge import evaluate_completion_checkpoints
        if self.finished:
            raise RuntimeError("finish_called_twice")
        self.finished = True
        self._action(surface="communication", target_agent="user", target_role="user", content=final_text)
        self.bank.dump_to_file(self.evidence_dir / "bank.sqlite")
        if self.errors:
            return invalid("official_dispatch_failure", evidence={"errors": self.errors})
        try:
            # Preflight every source SQL query. Upstream catches SQL errors as 0;
            # infrastructure/query failures must remain invalid in this adapter.
            for cp in self.source_task.completion_checkpoints:
                if cp.get("rule_type") in {"sql_assert", "sql_row_assert"}:
                    self.bank.query_sql(cp["query"])
            completion = asyncio.run(evaluate_completion_checkpoints(
                self.source_task, self.actions, final_text, bank=self.bank, skip_judge=False))
            if completion.llm_completion_weight:
                raise ValueError("unexpected_judge_requirement")
            decisions = check_trace(self.actions, self.source_task.access_rules, self.source_task.domain)
            metrics = compute_metrics(self.run_id, self.actions, decisions,
                                      resource_tools={t.name for t in self.catalog.tools if t.is_resource},
                                      task_completion_score=completion.task_completion_score,
                                      operational_safety_score=None).model_dump(mode="json")
            metrics["completion"] = asdict(completion)
            metrics["AVS_applicable_but_unavailable"] = True
            metrics["source_protocol"] = "single_agent_whitebox_adaptation_v3"
            write_json(self.evidence_dir / "decisions.json", [d.model_dump(mode="json") for d in decisions])
            write_json(self.evidence_dir / "metrics.json", metrics)
            result = {"done": True, "valid": True, "utility": completion.task_completion_score,
                      "c_source": float(any(d.violated for d in decisions)), "persistence": None,
                      "metrics": metrics, "evidence": {"actions": str(self.evidence_dir / "actions.jsonl"),
                      "bank": str(self.evidence_dir / "bank.sqlite"), "decisions": str(self.evidence_dir / "decisions.json")}}
        except Exception as exc:
            result = invalid(f"source_scoring_error:{type(exc).__name__}:{exc}")
        write_json(self.evidence_dir / "result.json", result)
        return result

    def close(self):
        if not self.closed:
            self.bank.close()
            close = getattr(self.file_tools, "close", None)
            if close:
                close()
            self.closed = True
