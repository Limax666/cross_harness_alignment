from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import Case
from .utils import sha256_text


@dataclass
class SafetyContext:
    text: str
    metadata: dict[str, Any]


def static_prompt(path: Path) -> SafetyContext:
    text = path.read_text(encoding="utf-8").strip()
    return SafetyContext(text, {"safety_module": "static_prompt", "safety_module_sha256": sha256_text(text)})


def skillbank_context(repo_root: Path, bank: Path, case: Case) -> SafetyContext:
    script = repo_root / "core" / "skillrl" / "runtime_retrieval.py"
    metadata = {
        "scenario": "attacked" if case.task_type == "injection" else case.task_type,
        "domain": case.domain,
        "task_type": case.task_type,
        "attack_type": case.metadata.get("attack_type", ""),
    }
    completed = subprocess.run(
        [sys.executable, str(script), "--bank", str(bank), "--metadata-json", json.dumps(metadata)],
        text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("SkillBank retrieval failed: %s" % completed.stderr[-2000:])
    value = json.loads(completed.stdout)
    bank_hash = sha256_text(bank.read_text(encoding="utf-8"))
    return SafetyContext(str(value.get("text") or ""), {
        "safety_module": "safeevolve_skillbank",
        "safety_module_sha256": bank_hash,
        "skillrl_bank_path": str(bank),
        "skillrl_bank_sha256": bank_hash,
        "skillrl_compiler_version": "runtime_skill_compiler.v1",
        "skillrl_retrieval_version": "runtime_skill_retrieval.v1",
        "skillrl_renderer_version": "runtime_skill_renderer.v1",
        "skillrl_retrieved_skill_ids": list(value.get("skill_ids") or []),
        "skillrl_rendered_skill_ids": list(value.get("skill_ids") or []),
        "skillrl_injected_skill_text": str(value.get("text") or ""),
    })


def build_safety_context(module: str, repo_root: Path, case: Case, config: dict[str, Any]) -> SafetyContext:
    if module in {"none", "pretool_guard"}:
        return SafetyContext("", {"safety_module": module, "safety_module_sha256": sha256_text(module)})
    if module == "static_prompt":
        return static_prompt(repo_root / str(config["static_prompt_path"]))
    if module == "safeevolve_skillbank":
        return skillbank_context(repo_root, repo_root / str(config["skillbank_path"]), case)
    raise ValueError("Unknown safety module: %s" % module)


def guard_tool_call(module: str, case: Case, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if module != "pretool_guard":
        return {"decision": "allow", "reason_code": "guard_disabled"}
    forbidden_tools = set(case.metadata.get("forbidden_tools") or [])
    forbidden_values = {str(value).lower() for value in (case.metadata.get("forbidden_argument_values") or [])}
    if name in forbidden_tools:
        return {"decision": "block", "reason_code": "forbidden_tool"}
    serialized = json.dumps(arguments, ensure_ascii=False).lower()
    if any(value and value in serialized for value in forbidden_values):
        return {"decision": "block", "reason_code": "forbidden_argument_value"}
    return {"decision": "allow", "reason_code": "deterministic_guard_pass"}

