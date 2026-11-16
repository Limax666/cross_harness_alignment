"""ClawGym selected RL tasks; immutable private checker and real input assets."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

from .backends import checked_source, invalid, tree_digest, write_json, sha256_file

PINNED_REVISION = "0e8526d8f5e0999896c74cee4d6b998fb8a0429e"
REWARD_SH = """#!/usr/bin/env bash
set -euo pipefail

python3 /root/.openclaw/workspace/reward/check.py /root/.openclaw/workspace | python3 -c 'import json, sys; raw = sys.stdin.read().strip(); data = json.loads(raw); print(sum(float(v) for v in data.values()) / len(data) if isinstance(data, dict) and data else float(data) if isinstance(data, (int, float)) else 0.0)'
"""


def qualification(path: Path) -> list[str]:
    reasons = []
    root = path.parent
    if not (root / "reward/check.py").is_file():
        reasons.append("missing_official_checker")
    script = root / "reward/reward.sh"
    if not script.is_file() or script.read_text().strip() != REWARD_SH.strip():
        reasons.append("unsupported_official_reward_aggregation")
    # Upstream permits absent input_files for tasks whose inputs are inline.
    # Explicit unsupported capabilities are inventoried from the actual prompt;
    # a missing directory alone is not a rejection criterion.
    unsupported = {"task_0298", "task_0352", "task_0378", "task_0398", "task_0499", "task_0933", "task_1747"}
    if path.parent.name in unsupported:
        reasons.append("requires_live_external_download_unavailable_in_network_isolation")
    if path.parent.name == "task_0545":
        reasons.append("requires_native_openclaw_host_introspection_not_isolated_workspace")
    if "clawgym_train" not in path.parts or "clawgym_eval" in path.parts:
        reasons.append("outside_official_rl_selected_train")
    data = json.loads(path.read_text())
    if not isinstance(data.get("user_query"), str) or not data["user_query"].strip():
        reasons.append("missing_user_query")
    if any(item.is_symlink() for item in root.rglob("*")):
        reasons.append("source_symlink_requires_review")
    return reasons


def aggregate_reward(raw: object) -> float:
    """Same mean/scalar as pinned reward.sh; malformed results are invalid, never 0."""
    values = list(raw.values()) if isinstance(raw, dict) and raw else [raw]
    if not values or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError("checker_result_not_finite_numeric_scores")
    # Spec u in [0,1]: out-of-range official outputs need review, never clipping.
    reward = sum(float(x) for x in values) / len(values)
    if not 0 <= reward <= 1:
        raise ValueError("official_reward_outside_unit_interval")
    return reward


class ClawBackend:
    def __init__(self, task: dict, workspace: Path, evidence_dir: Path):
        from .sandbox import WorkspaceTools
        path = checked_source(task)
        reasons = qualification(path)
        if reasons:
            raise ValueError(";".join(reasons))
        self.task, self.source_dir = task, path.parent
        self.workspace, self.evidence_dir = Path(workspace), Path(evidence_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        if any(self.workspace.iterdir()):
            raise ValueError("rollout_workspace_must_be_fresh")
        # Only original inputs; data_entry/checkers and private metadata stay host-side.
        input_dir = self.source_dir / "input_files"
        for item in input_dir.iterdir() if input_dir.is_dir() else []:
            target = self.workspace / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        self.input_hashes = {item.relative_to(self.workspace).as_posix(): sha256_file(item) for item in self.workspace.rglob("*") if item.is_file()}
        self.file_tools = WorkspaceTools(self.workspace)
        self.messages = [{"role": "user", "content": json.loads(path.read_text())["user_query"]}]
        self.tools = self.file_tools.schemas
        self.finished, self.closed = False, False
        write_json(self.evidence_dir / "source_provenance.json", {
            "source_path": str(path), "source_sha256": task["source_sha256"],
            "source_tree_sha256": task.get("source_tree_sha256"),
            "checker_tree_sha256": tree_digest(self.source_dir / "reward"),
            "adaptation": "unchanged_check.py_in_private_grader_invoked_with_frozen_workspace_snapshot",
            "reward_aggregation": "official_reward_sh_numeric_dict_mean_or_scalar"})

    def call(self, name: str, arguments: dict) -> str:
        if self.finished or self.closed:
            raise RuntimeError("backend_closed_or_finished")
        if name not in {tool["function"]["name"] for tool in self.tools}:
            raise ValueError("unknown_tool:" + name)
        result = self.file_tools.call(name, arguments)
        with (self.evidence_dir / "actions.jsonl").open("a") as stream:
            stream.write(json.dumps({"name": name, "arguments": arguments, "result": result}) + "\n")
        return result

    def finish(self, final_text: str) -> dict:
        if self.finished:
            raise RuntimeError("finish_called_twice")
        self.finished = True
        try:
            checked_source(self.task)
            changed_inputs = [name for name, digest in self.input_hashes.items() if not (self.workspace / name).is_file() or (self.workspace / name).is_symlink() or sha256_file(self.workspace / name) != digest]
            if changed_inputs:
                raise ValueError("initial_input_modified:" + ",".join(changed_inputs))
            grade = self.file_tools.grade_private(self.source_dir / "reward", Path("check.py"))
            write_json(self.evidence_dir / "private_grade.json", grade)
            if not grade.get("valid") or grade.get("returncode", 0) != 0:
                return invalid("private_grader_failed", evidence=grade)
            raw = json.loads(grade["stdout"].strip())
            utility = aggregate_reward(raw)
            result = {"done": True, "valid": True, "utility": utility,
                      "c_source": None, "persistence": None,
                      "metrics": {"official_checker_scores": raw,
                                  "safety_oracle_applicable": False,
                                  "source_protocol": "clawgym_private_grader_adaptation_v3"},
                      "evidence": {"private_grade": str(self.evidence_dir / "private_grade.json"),
                                   "source_provenance": str(self.evidence_dir / "source_provenance.json")}}
        except Exception as exc:
            result = invalid(f"source_scoring_error:{type(exc).__name__}:{exc}")
        write_json(self.evidence_dir / "result.json", result)
        return result

    def close(self):
        if not self.closed:
            close = getattr(self.file_tools, "close", None)
            if close:
                close()
            self.closed = True
