"""One isolated, auditable AgentDojo episode for a current-policy rollout.

This module executes only model-attributed tool calls. It does not generate
tokens, claim an on-policy rollout, or turn an unscorable trace into a reward.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from rl_agentdojo_path_audit import NativePathRecorder, _ground_truth_tools
from rl_native_evidence import agentdojo_evidence
from rl_online_batch_gate import CELLS
from rl_tool_attempt_audit import attest_native_calls, parse_strict_tool_turn
from rl_hermes_mcp_contract import hermes_mcp_tools
from rl_mcp_benchmark_contract import mcp_benchmark_tools
from rl_verifier import TaskKind, score_episode
from rl_reward_v4 import score_agentdojo_v4
from rl_reward_verified_progress import score_verified_progress
from rl_progress_registry import DEFAULT as DEFAULT_PROGRESS_ORACLE_VERSION, resolve as resolve_progress_oracle


def _digest(environment: Any) -> str:
    data = environment.model_dump(mode="json")
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class NativeAgentDojoEpisode:
    driver: Any
    task_id: str
    seed: int
    hermes_mcp: bool = False
    mcp_benchmark: bool = False
    harness: str = "hermes"
    path_weight: float = 0.25
    reward_contract: str = "legacy"
    reward_variant: str = "progress_protocol"
    progress_oracle_version: str = DEFAULT_PROGRESS_ORACLE_VERSION
    allowed_cells: frozenset[str] = CELLS
    wire_transform: Any = None
    episode: Any = field(init=False)
    tools: list[dict] = field(init=False)
    recorder: NativePathRecorder = field(init=False)
    initial_state_sha256: str = field(init=False)
    raw_assistant_turns: list[str] = field(default_factory=list)
    last_answer: str = ""
    finished: bool = False
    protocol_rejected: bool = False
    wire_to_native: dict[str, str] = field(init=False)

    def __post_init__(self):
        if self.task_id not in self.allowed_cells or type(self.seed) is not int or self.seed < 0:
            raise ValueError("pilot requires one registered train task cell and a nonnegative seed")
        if self.hermes_mcp and self.mcp_benchmark:
            raise ValueError("choose exactly one MCP wire adapter")
        self.progress_oracle = None
        if self.reward_contract == "verified_progress_v1":
            self.progress_oracle = resolve_progress_oracle(self.progress_oracle_version)
        started = self.driver.start(self.task_id, self.seed)
        self.episode = started["episode"]
        if self.wire_transform is not None:
            if self.hermes_mcp or self.mcp_benchmark:
                raise ValueError("custom wire transform cannot be combined with MCP adapters")
            self.tools, self.wire_to_native = self.wire_transform(started["tools"])
            originals = {t["function"]["name"]: t["function"]["parameters"] for t in started["tools"]}
            if set(self.wire_to_native.values()) != set(originals) or len(self.tools) != len(originals):
                raise ValueError("custom wire mapping changed native tool coverage")
            for tool in self.tools:
                f = tool["function"]
                if f["parameters"] != originals[self.wire_to_native[f["name"]]]:
                    raise ValueError("custom wire transform changed parameter semantics")
        elif self.hermes_mcp:
            self.tools, self.wire_to_native = hermes_mcp_tools(started["tools"])
        elif self.mcp_benchmark:
            self.tools, self.wire_to_native = mcp_benchmark_tools(started["tools"])
        else:
            self.tools = started["tools"]
            self.wire_to_native = {tool["function"]["name"]: tool["function"]["name"] for tool in self.tools}
        self.initial_state_sha256 = _digest(self.episode.environment)
        qualified_families = frozenset(cell.rpartition(":")[0] for cell in self.allowed_cells)
        self.recorder = NativePathRecorder(self.driver, self.episode, tool_schemas=self.tools,
                                           qualified_families=qualified_families)

    def turn(self, raw_assistant: str, parsed_calls: list[dict], *, final_content: str | None = None) -> list[dict]:
        """Check all attempted calls against generated text before dispatch."""
        if self.finished or self.protocol_rejected or not isinstance(raw_assistant, str) or not raw_assistant:
            raise ValueError("episode ended or raw assistant generation missing")
        if final_content is not None and (not isinstance(final_content, str) or parsed_calls):
            raise ValueError("final answer must be text from a tool-free assistant turn")
        calls = []
        for call in parsed_calls:
            if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
                raise ValueError("parser must supply a separate tool name and JSON object arguments")
            calls.append(SimpleNamespace(name=call["name"], arguments=call["arguments"]))
        candidate = self.raw_assistant_turns + [raw_assistant]
        all_calls = [SimpleNamespace(name=r.name, arguments=r.arguments) for r in self.recorder.records] + calls
        if not attest_native_calls(candidate, all_calls, self.tools):
            raise ValueError("generated tool syntax does not match parsed calls; no dispatch")
        self.raw_assistant_turns.append(raw_assistant)
        if not calls:
            self.last_answer = final_content if final_content is not None else raw_assistant
        observations = []
        for call, parsed in zip(calls, parsed_calls):
            blocked_reason = parsed.get("blocked_reason")
            if blocked_reason is not None:
                observations.append(self.recorder.blocked_tool(call.name, call.arguments, blocked_reason))
                continue
            native_name = self.wire_to_native.get(call.name) if (self.hermes_mcp or self.mcp_benchmark or self.wire_transform is not None) else call.name
            if native_name is None:
                raise ValueError(f"unregistered MCP tool name: {call.name}")
            result = self.recorder.tool(call.name, call.arguments, native_name=native_name)
            observations.append(result)
        return observations

    def turn_from_raw(self, raw_assistant: str) -> list[dict]:
        """Use the strict SFT-compatible parser as the sole dispatch path."""
        final_content, calls = parse_strict_tool_turn(raw_assistant, self.tools)
        return self.turn(raw_assistant, calls, final_content=final_content if not calls else None)

    def reject_protocol_turn(self, raw_assistant: str) -> None:
        """End a malformed model turn without dispatch; retain exact raw evidence."""
        self._reject_turn(raw_assistant, "protocol_error")

    def reject_incomplete_turn(self, raw_assistant: str) -> None:
        """End an unfinished model turn returned by a completed generation."""
        self._reject_turn(raw_assistant, "model_timeout")

    def _reject_turn(self, raw_assistant: str, kind: str) -> None:
        if self.finished or self.protocol_rejected or not isinstance(raw_assistant, str) or not raw_assistant:
            raise ValueError("invalid protocol rejection")
        if kind == "protocol_error":
            try:
                parse_strict_tool_turn(raw_assistant, self.tools, allow_schema_errors=True)
            except ValueError:
                pass
            else:
                raise ValueError("valid tool syntax cannot be rejected as malformed")
        elif kind == "model_timeout":
            if any(raw_assistant.rstrip().endswith(tag) for tag in
                   ("</tool_call>", "<|im_end|>", "<|endoftext|>")):
                raise ValueError("complete model turn cannot be rejected as incomplete")
        else:
            raise ValueError("unsupported rejected-turn kind")
        before = self.episode.environment.model_dump(mode="json")
        trace_len = len(self.episode.trace)
        self.raw_assistant_turns.append(raw_assistant)
        after = self.episode.environment.model_dump(mode="json")
        if before != after or len(self.episode.trace) != trace_len:
            raise RuntimeError("protocol rejection changed native environment")
        self.recorder.rejected_turn = raw_assistant
        self.recorder.rejected_kind = kind
        self.recorder.rejected_before = before
        self.recorder.rejected_after = after
        self.protocol_rejected = True

    def finish(self, *, termination: str = "completed") -> dict:
        if self.finished or not self.raw_assistant_turns:
            raise ValueError("episode already ended or no assistant generation")
        if termination not in {"completed", "model_timeout", "protocol_error", "empty_final_answer", "infrastructure_error"}:
            raise ValueError("unsupported termination status")
        if self.protocol_rejected != (termination in {"protocol_error", "model_timeout"} and
                                      self.recorder.rejected_turn is not None):
            raise ValueError("protocol termination must match rejected raw turn")
        if self.recorder.rejected_turn is not None and termination != self.recorder.rejected_kind:
            raise ValueError("rejected-turn termination mismatch")
        if termination == "completed" and not self.last_answer:
            raise ValueError("completed episode has no final assistant answer")
        self.finished = True
        self.recorder.raw_assistant_turns = list(self.raw_assistant_turns)
        official = self.driver.finish(self.episode, self.last_answer)
        audit = self.recorder.audit(self.task_id)
        family_id = self.task_id.rpartition(":")[0]
        kind = TaskKind.BENIGN if self.task_id.endswith(":clean") else TaskKind.INJECTION
        valid_dispatches = [r.native_name for r in self.recorder.records
                            if r.blocked_reason is None and r.native_name]
        ground_truth_tools = _ground_truth_tools(self.episode)
        reference_tool_count = len(ground_truth_tools or ())
        subgoal_fraction = (len(set(valid_dispatches) & ground_truth_tools) / len(ground_truth_tools)
                            if ground_truth_tools else 0.0)
        evidence = agentdojo_evidence(
            harness=self.harness, family_id=family_id, task_kind=kind,
            official_result=official, path_audit=audit, termination=termination,
            blocked_invalid_calls=sum(r.blocked_reason is not None for r in self.recorder.records),
            valid_tool_dispatches=len(valid_dispatches),
            verified_safe_subgoal_fraction=subgoal_fraction,
        )
        progress_evidence = None
        protocol_error_verified = False
        if self.reward_contract == "verified_progress_v1":
            if self.path_weight != 0 or os.environ.get("CHSPO_REWARD_VERSION") == "v4":
                raise ValueError("verified progress cannot be combined with legacy path rewards")
            def replay_read(record):
                from agentdojo.functions_runtime import FunctionsRuntime
                from cross_harness_sft.backends.agentdojo import _jsonable
                try:
                    cloned = type(self.episode.environment).model_validate(record.before)
                    result, error = FunctionsRuntime(self.episode.suite.tools).run_function(
                        cloned, record.native_name, record.arguments)
                    # Reproduce the entire recorded state transition. Most
                    # progress steps are state-preserving reads; a narrowly
                    # whitelisted browser-history append is checked by the
                    # task oracle only after this exact replay succeeds.
                    return (error is None and cloned.model_dump(mode="json") == record.after
                            and _jsonable(result) == record.tool_result)
                except Exception:
                    return False
            if evidence.final_state_verified and evidence.path_verified:
                progress_evidence = self.progress_oracle.verify_task_progress(
                    self.task_id, self.recorder.records, replay_read,
                    bool(evidence.legitimate_goal_complete))
            protocol_error_verified = termination == "protocol_error"
            for raw in self.raw_assistant_turns:
                try:
                    _, parsed = parse_strict_tool_turn(raw, self.tools, allow_schema_errors=True)
                except ValueError:
                    continue
                for call in parsed:
                    reason = call.get("blocked_reason")
                    if reason and any(r.name == call['name'] and r.arguments == call['arguments'] and
                                      r.blocked_reason == reason for r in self.recorder.records):
                        protocol_error_verified = True
            verdict = score_verified_progress(evidence,
                progress_evidence['fraction'] if progress_evidence is not None else None,
                protocol_error_verified, variant=self.reward_variant)
        elif os.environ.get("CHSPO_REWARD_VERSION") == "v4":
            verdict = score_agentdojo_v4(evidence, required_tool_count=reference_tool_count,
                                          path_weight=self.path_weight)
        else:
            verdict = score_episode(evidence, path_weight=self.path_weight, reward_contract=self.reward_contract)
        second = self.driver.start(self.task_id, self.seed + 1)["episode"]
        reset_verified = _digest(second.environment) == self.initial_state_sha256
        return {
            "task_id": self.task_id, "seed": self.seed,
            "initial_state_sha256": self.initial_state_sha256,
            "final_state_sha256": _digest(self.episode.environment),
            "reset_verified": reset_verified,
            "final_state_verified": evidence.final_state_verified,
            "path_verified": evidence.path_verified,
            "official_utility": official.get("utility_reward"),
            "official_attack_success": official.get("risk_success"),
            "audit_complete": evidence.final_state_verified and evidence.path_verified,
            "audit_version": audit.audit_version,
            "prohibited_persistence": audit.prohibited_persistence,
            "harmful_action_executed": audit.harmful_action_executed,
            "invalid_tool_dispatched": audit.invalid_tool_dispatched,
            "termination": termination,
            "reward": verdict.score, "reward_outcome": verdict.outcome,
            "reward_reason": verdict.reason, "reward_path_signal": verdict.path_signal,
            "reward_contract": self.reward_contract, "reward_variant": self.reward_variant,
            "progress_oracle_version": (self.progress_oracle.VERSION if self.progress_oracle is not None else None),
            "progress_evidence": progress_evidence,
            "protocol_error_verified": protocol_error_verified,
            "verified_safe_subgoal_fraction": subgoal_fraction,
            "reference_tool_count": reference_tool_count,
            "valid_tool_dispatches": len(valid_dispatches),
            "raw_assistant_turns": list(self.raw_assistant_turns),
            "tool_calls": len(self.recorder.records),
            "blocked_tool_calls": sum(r.blocked_reason is not None for r in self.recorder.records),
            "audit_trace": ({
                "abstention_reason": self.recorder.abstention_reason,
                "calls": [{"wire": r.name, "native": r.native_name,
                           "changed_state": r.before != r.after,
                           "invalid": r.invalid_call, "blocked": r.blocked_reason is not None}
                          for r in self.recorder.records],
                "native_trace": [str(getattr(t, "function", "")) for t in self.episode.trace],
                "rejected_kind": self.recorder.rejected_kind,
            } if not evidence.path_verified else None),
        }
