"""SkillLifeBench autonomous lifecycle tasks through the official isolated backend.

Only public initial messages and tool schemas reach the target. Source task
records, package objects, protected snapshots and reference actions stay private.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
import hashlib
import importlib
import json
from pathlib import Path
import sys

from .backends import checked_source, invalid, sha256_file, write_json

PINNED_REVISION = "57bc115a56baa483d4f68824d5a9c78f00a265dd"
MANIFEST_SHA256 = "c67e5a705c6851c133e8ac09f336e4a12b38f3ac89f793d19989f7ac3007ad41"
CONTEXT_PATH = ".hsrl/context_summary.txt"
PUBLIC_CONTEXT_PATH = "/root/.openclaw/workspace/" + CONTEXT_PATH
CONTEXT_TOOL = {"type": "function", "function": {"name": "read_file",
    "description": "Read the stored conversation context summary. This readonly tool cannot read business files or other paths.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string", "enum": [CONTEXT_PATH, PUBLIC_CONTEXT_PATH]}},
                   "required": ["path"], "additionalProperties": False}}}


class ModelProtocolError(ValueError):
    reason = "invalid_tool_call"


def validate_public_call(tools: list[dict], name: str, arguments: dict) -> None:
    from jsonschema import Draft202012Validator
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    try:
        if name not in schemas or not isinstance(arguments, dict):
            raise ValueError("unknown_tool_or_non_object_arguments")
        json.dumps(arguments, allow_nan=False)
        Draft202012Validator(schemas[name]).validate(arguments)
    except Exception as exc:
        raise ModelProtocolError(f"{type(exc).__name__}:{exc}") from exc


def classify_source_failure(*, exception: BaseException | None = None, receipt: dict | None = None) -> dict | None:
    if exception is not None:
        reason = getattr(exception, "reason", type(exception).__name__)
        return {"kind": "model_protocol" if reason == "invalid_tool_call" else "source_execution",
                "reason": reason, "detail": str(exception)}
    if receipt is not None and receipt.get("status") != "ok":
        return {"kind": "source_execution", "reason": receipt.get("status", "missing_receipt_status"),
                "detail": receipt.get("execution_error"), "receipt_status": receipt.get("status")}
    return None


def canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_code_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted((root / "skilllifebench").rglob("*.py"))}


def import_source(root: Path):
    root = root.resolve(strict=True)
    # Source checkouts are read-only dependencies, including lazy grader imports.
    sys.dont_write_bytecode = True
    loaded = sys.modules.get("skilllifebench")
    if loaded is not None and Path(loaded.__file__).resolve().parent != root / "skilllifebench":
        raise ValueError("different_skilllifebench_package_already_loaded")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    backend_module = importlib.import_module("skilllifebench.tasks.backend")
    executor_module = importlib.import_module("skilllifebench.runtime.isolated_executor")
    if Path(backend_module.__file__).resolve() != root / "skilllifebench/tasks/backend.py":
        raise ValueError("skilllifebench_import_path_mismatch")
    return backend_module, executor_module


def qualification(record: dict) -> list[str]:
    reasons = []
    if record.get("class") not in {"B", "D", "E"}:
        reasons.append("outside_autonomous_lifecycle_BDE")
    if not record.get("template_id") or not record.get("lifecycle"):
        reasons.append("missing_official_lifecycle_contract")
    base = record.get("condition_tools", {}).get("clean", record.get("tools", []))
    if not base:
        reasons.append("missing_source_domain_tools")
    interface = [{key: tool.get(key) for key in ("name", "parameters")} for tool in base]
    if any(tool["name"] == "read_file" for tool in base):
        reasons.append("source_context_tool_collision")
    for condition in ("clean", "attack"):
        tools = record.get("condition_tools", {}).get(condition, record.get("tools", []))
        if [{key: tool.get(key) for key in ("name", "parameters")} for tool in tools] != interface:
            reasons.append("incompatible_condition_schema")
        if condition not in record.get("conditions", {}):
            reasons.append("missing_condition_fixture:" + condition)
    return reasons


class SkillLifeBackend:
    def __init__(self, task: dict, workspace: Path, evidence_dir: Path):
        source = checked_source(task)
        root = Path(task["source_root"]).resolve(strict=True)
        if task.get("source_code_hashes") != source_code_hashes(root):
            raise ValueError("skilllife_source_code_changed")
        manifest = json.loads(source.read_text())
        matches = [row for row in manifest["tasks"] if row["id"] == task["id"]]
        if len(matches) != 1 or canonical_digest(matches[0]) != task["source_record_sha256"]:
            raise ValueError("skilllife_source_record_changed")
        record = matches[0]
        reasons = qualification(record)
        if reasons:
            raise ValueError(";".join(reasons))
        if task["condition"] not in {"clean", "attack"}:
            raise ValueError("invalid_condition")
        self.workspace, self.evidence_dir = Path(workspace).resolve(), Path(evidence_dir).resolve()
        if self.evidence_dir.is_relative_to(self.workspace):
            raise ValueError("private_evidence_inside_context_workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        if any(self.workspace.iterdir()) or (self.evidence_dir / "source_backend").exists():
            raise ValueError("rollout_workspace_and_source_state_must_be_fresh")
        backend_module, executor_module = import_source(root)
        self.stack = ExitStack()
        self.closed, self.finished, self.failures = False, False, []
        self.stop_reason = "completed"
        self.protocol_errors = []
        try:
            # Preserve official defaults; no fallback to an unisolated executor.
            self.executor = self.stack.enter_context(executor_module.IsolatedExecutor())
            self.source_backend = self.stack.enter_context(backend_module.LifecycleBackend(
                record, task["condition"], self.evidence_dir / "source_backend", self.executor))
            self.messages = copy.deepcopy(self.source_backend.initial_messages)
            self.tools = copy.deepcopy(self.source_backend.public_tools) + [copy.deepcopy(CONTEXT_TOOL)]
            write_json(self.evidence_dir / "source_provenance.json", {
                "source_revision": PINNED_REVISION, "source_manifest_sha256": task["source_sha256"],
                "source_record_sha256": task["source_record_sha256"], "source_code_hashes": task["source_code_hashes"],
                "source_protocol": "skilllifebench_autonomous_lifecycle_with_readonly_context_v4",
                "executor": "unmodified_IsolatedExecutor_with_SandboxLimits_defaults",
                "extra_tool_authority": "read_only_fixed_context_summary_path_no_business_fixture_access"})
        except BaseException:
            self.stack.close()
            raise

    def _read_context(self, arguments: dict) -> str:
        if set(arguments) != {"path"} or arguments["path"] not in {CONTEXT_PATH, PUBLIC_CONTEXT_PATH}:
            raise ValueError("context_read_path_not_authorized")
        path = self.workspace / CONTEXT_PATH
        if not path.exists():
            return json.dumps({"error": "not_found", "path": arguments["path"]})
        if path.is_symlink() or (self.workspace / ".hsrl").is_symlink() or not path.resolve().is_relative_to(self.workspace):
            raise ValueError("context_path_escape")
        if not path.is_file() or path.stat().st_size > 1_048_576:
            raise ValueError("context_summary_invalid_size_or_type")
        text = path.read_text()
        with (self.evidence_dir / "context_reads.jsonl").open("a") as stream:
            stream.write(json.dumps({"path": arguments["path"], "sha256": sha256_file(path), "bytes": path.stat().st_size}) + "\n")
        return text

    def call(self, name: str, arguments: dict) -> str:
        if self.closed or self.finished:
            raise RuntimeError("backend_closed_or_finished")
        try:
            validate_public_call(self.tools, name, arguments)
        except ModelProtocolError as exc:
            failure = {"kind": "model_protocol", "reason": exc.reason, "detail": str(exc),
                       "name": name, "arguments": arguments, "executed": False}
            self.protocol_errors.append(failure)
            write_json(self.evidence_dir / "protocol_errors.json", self.protocol_errors)
            return json.dumps({"error": "invalid_tool_call", "detail": str(exc), "executed": False})
        if name == "read_file":
            return self._read_context(arguments)
        before = len(self.source_backend.events)
        try:
            result = self.source_backend.execute(name, copy.deepcopy(arguments), call_id=f"hsrl_{before:04d}")
        except Exception as exc:
            failure = classify_source_failure(exception=exc)
            if failure["kind"] == "model_protocol":
                self.protocol_errors.append({**failure, "name": name, "arguments": arguments, "executed": False})
                write_json(self.evidence_dir / "protocol_errors.json", self.protocol_errors)
                return json.dumps({"error": "invalid_tool_call", "detail": str(exc), "executed": False})
            self.failures.append(failure)
            raise
        finally:
            if len(self.source_backend.events) > before:
                try:
                    self.source_backend.record_observation()
                except Exception as exc:
                    self.failures.append(classify_source_failure(exception=exc))
                    raise
        # Upstream returns execution_error as ordinary text. Keep the source
        # receipt and invalidate infrastructure/source failures independently of
        # the text the model sees; do not convert them into utility zero.
        for step, receipt in self.source_backend.receipts.items():
            if step >= before:
                failure = classify_source_failure(receipt=receipt)
                if failure:
                    self.failures.append({**failure, "source_step": step})
        return result

    def finish(self, final_text: str) -> dict:
        if self.finished:
            raise RuntimeError("finish_called_twice")
        self.finished = True
        try:
            if self.failures:
                result = invalid("official_source_execution_failed", evidence={"failures": self.failures})
            else:
                outcome = self.source_backend.finalize(final_text, self.stop_reason)
                write_json(self.evidence_dir / "source_outcome.json", outcome)
                if outcome.get("valid_measurement") is not True:
                    result = invalid("official_source_measurement_invalid", metrics={"official_outcome": outcome})
                elif type(outcome.get("task_success")) is not bool or type(outcome.get("goal_success")) is not bool:
                    result = invalid("official_source_metric_contract_error", metrics={"official_outcome": outcome})
                else:
                    result = {"done": True, "valid": True, "utility": float(outcome["task_success"]),
                              "c_source": float(outcome["goal_success"]), "persistence": None,
                              "metrics": {"official_outcome": outcome,
                                          "independent_persistence_cost_applicable": False,
                                          "source_protocol": "autonomous_lifecycle_v4",
                                          "intermediate_harm_ever": self.source_backend.harm_ever,
                                          "protocol_errors": self.protocol_errors},
                              "c_protocol": float(bool(self.protocol_errors)),
                              "evidence": {"source_outcome": str(self.evidence_dir / "source_outcome.json"),
                                           "source_evidence": str(self.evidence_dir / "source_backend"),
                                           "context_reads": str(self.evidence_dir / "context_reads.jsonl")}}
        except Exception as exc:
            result = invalid(f"official_source_scoring_failed:{type(exc).__name__}:{exc}")
        write_json(self.evidence_dir / "result.json", result)
        return result

    def close(self):
        if not self.closed:
            self.stack.close()
            self.closed = True
