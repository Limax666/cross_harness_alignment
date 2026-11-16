"""ClawGym selected RL tasks; immutable private checker and real input assets."""
from __future__ import annotations

import ast
import json
import re
import math
from pathlib import Path
import shutil

from .backends import checked_source, invalid, tree_digest, write_json, sha256_file

PINNED_REVISION = "0e8526d8f5e0999896c74cee4d6b998fb8a0429e"
HF_REVISION = "bd9043cff88b913922a165d9c022a4796aee3f3a"
HF_SHA256 = "0455f13c4f7a4839d39c3c1da9704c4959046ed529040d44f46f8f9bd07f7774"
REWARD_SH = """#!/usr/bin/env bash
set -euo pipefail

python3 /root/.openclaw/workspace/reward/check.py /root/.openclaw/workspace | python3 -c 'import json, sys; raw = sys.stdin.read().strip(); data = json.loads(raw); print(sum(float(v) for v in data.values()) / len(data) if isinstance(data, dict) and data else float(data) if isinstance(data, (int, float)) else 0.0)'
"""


REVIEWED_WRITABLE_OUTPUTS = {
    "task_0102": {"pipeline/config.yaml": "Updated production configuration is an explicitly requested output; checker verifies fixed production/deploy/version values. Event gold uses immutable HTML and immutable src/event_parser.py."},
    "task_0486": {"input/indexing_config.yaml": "Updated indexing configuration is an explicitly requested output; checker uses immutable catalog HTML for records/coverage and hardcoded original baseline values for change deltas."},
}


def _checker_capability_scan(checker: Path) -> dict:
    tree = ast.parse(checker.read_text())
    imports, network_calls, host_calls = [], [], []
    lines = checker.read_text().splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name in {"requests.get", "requests.post", "requests.request", "urllib.request.urlopen", "urlopen", "httpx.get", "httpx.post"}:
            network_calls.append({"line": node.lineno, "text": lines[node.lineno - 1], "call": name})
        if name in {"subprocess.run", "subprocess.check_output", "subprocess.call", "os.system"} and node.args:
            try:
                command = ast.literal_eval(node.args[0])
            except (ValueError, TypeError):
                continue
            text = " ".join(map(str, command)) if isinstance(command, (list, tuple)) else str(command)
            if re.search(r"(?:^|[ /])(curl|wget|ping|traceroute|whois|lspci|lsusb)(?: |$)", text):
                host_calls.append({"line": node.lineno, "text": lines[node.lineno - 1], "command": text})
    return {"checker_sha256": sha256_file(checker), "imports": sorted(set(imports)),
            "required_network_calls": network_calls, "required_host_commands": host_calls}


TOOL_IMAGE_THIRD_PARTY_MODULES = frozenset({"yaml", "numpy", "pandas", "openpyxl", "requests", "bs4", "lxml", "cryptography", "jinja2", "pytest"})


def dependency_contract(source_root: Path, query: str) -> dict:
    import sys
    assets = source_root / "input_files"
    python_files = sorted(assets.rglob("*.py"))
    local_modules = {p.stem for p in python_files} | {part for p in python_files for part in p.relative_to(assets).parts[:-1]}
    required, observed, pending = [], [], []
    explicit_execution = bool(re.search(r"(?:run|execute).{0,90}(?:test suite|tests|script|python|\.py)|actually run", query, re.I))
    for path in [source_root / "reward/check.py", *python_files]:
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeError):
            continue
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            modules = [alias.name.split(".")[0] for alias in node.names] if isinstance(node, ast.Import) else [str(node.module or "").split(".")[0]]
            if isinstance(node, ast.ImportFrom) and node.level:
                continue
            ancestor, optional = parents.get(node), False
            while ancestor is not None:
                if isinstance(ancestor, ast.Try):
                    optional = True
                ancestor = parents.get(ancestor)
            for module in modules:
                if not module or module in sys.stdlib_module_names or module in local_modules:
                    continue
                entry = {"module": module, "source": path.relative_to(source_root).as_posix(), "line": node.lineno,
                         "import": ast.unparse(node), "optional_guard": optional,
                         "declared_image_support": module in TOOL_IMAGE_THIRD_PARTY_MODULES}
                observed.append(entry)
                if path == source_root / "reward/check.py":
                    required.append(entry)
                elif explicit_execution and not optional:
                    pending.append(entry)
    unavailable = sorted({row["module"] for row in required if not row["declared_image_support"]})
    unresolved = sorted({row["module"] for row in pending if not row["declared_image_support"]})
    return {"declared_tool_image_modules": sorted(TOOL_IMAGE_THIRD_PARTY_MODULES),
            "observed_third_party_imports": observed, "unavailable_required_checker_modules": unavailable,
            "unresolved_requested_execution_modules": unresolved,
            "policy": "unavailable_checker_imports_including_unverified_optional_fallbacks_and_requested_execution_unknowns_are_pending_not_utility_zero"}


def source_contract(path: Path) -> dict:
    """Freeze capability and input-role admission from source text, never results.

    URL strings and urllib.parse imports are not network requirements. Positive
    external actions are matched as clauses, with explicit offline/no-network
    contracts and bundled input inventories retained as review evidence.
    """
    path = Path(path)
    query = json.loads(path.read_text())["user_query"]
    source_root = path.parent
    assets = {p.relative_to(source_root / "input_files").as_posix(): sha256_file(p)
              for p in sorted((source_root / "input_files").rglob("*")) if p.is_file()}
    clauses = [part.strip() for part in re.split(r"\n|(?<=[.!?;])\s+", query) if part.strip()]
    negative_network = re.compile(r"(?:do not|don't|never) (?:use|access|make|perform|fetch|download|call|contact|rely on).{0,45}(?:internet|network|external|web)|\bno (?:live )?(?:internet|network|external calls)|without (?:any )?network|locally without network|fully offline", re.I)
    offline_clauses = [part for part in clauses if negative_network.search(part)]
    network_patterns = [
        r"(?:internet|web retrieval|web search).{0,35}(?:required|essential|mandatory|requirement)",
        r"(?:essential|required|mandatory|live|actual|real).{0,30}(?:internet|web search|search engine)",
        r"(?:perform|use|using|run|conduct|via|with).{0,45}(?:web search|search engine)",
        r"\b(?:download|fetch|retrieve)\b.{0,230}(?:official|https?://|api|remote|internet|webpage|website|online|geonames)",
        r"(?:search|query).{0,100}(?:wikidata|official.{0,30}api)",
        r"(?:perform|run|conduct|collect|retrieve).{0,60}(?:DNS lookup|WHOIS|RDAP|reverse DNS|TLS certificate|internet speed|speed test)",
    ]
    evidence = []
    for part in clauses:
        if negative_network.search(part) or re.search(r"would.{0,60}(?:fetch|online)|if (?:it|the site) were online|simulated fetch|raw fetch result dumps", part, re.I):
            continue
        if any(re.search(pattern, part, re.I) for pattern in network_patterns):
            # Clear local-only contracts resolve references to HTTP code/design
            # as offline work; conflicting imperatives are deferred, not run.
            loopback = bool(re.search(r"https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?", part, re.I))
            bundled_service = any(re.search(r"(?:mock|server|fixture|api)", asset, re.I) for asset in assets)
            if loopback and bundled_service:
                continue
            evidence.append({"capability": "declared_local_service_unavailable" if loopback else "external_network", "source": "user_query", "text": part})
    checker = _checker_capability_scan(source_root / "reward/check.py")
    dependencies = dependency_contract(source_root, query)
    for call in checker["required_network_calls"]:
        evidence.append({"capability": "external_network", "source": "official_checker_ast", **call})
    host_patterns = [r"(?:run|perform).{0,80}(?:audit|inspection).{0,80}(?:our|my|this|current).{0,30}(?:host|workstation|system)",
                     r"\b(?:run|execute|collect|enumerate).{0,35}(?:lspci|lsusb|hardware enumeration|host discovery|latency test|path trace|network reachability)",
                     r"(?:audit|inventory|inspection).{0,50}(?:PCI|USB devices|hardware|OpenClaw host)"]
    snapshot_contract = bool(re.search(r"(?:provided|captured|exported).{0,50}(?:snapshot|command output|system report|memory.analysis)|do not.{0,30}(?:probe|scan|contact|network)", query, re.I))
    for part in clauses:
        if not snapshot_contract and any(re.search(pattern, part, re.I) for pattern in host_patterns):
            evidence.append({"capability": "native_host_or_external_probe", "source": "user_query", "text": part})
    for call in checker["required_host_commands"]:
        evidence.append({"capability": "native_host_or_external_probe", "source": "official_checker_ast", **call})
    requirements = sorted({row["capability"] for row in evidence})
    if offline_clauses and requirements:
        # Do not mislabel a task as network-required when offline supplied files
        # are explicitly authoritative. Freeze the conflict for review instead.
        requirements = ["capability_contract_conflict_pending"]
    writable_evidence, pending_writes = {}, {}
    lines = query.splitlines()
    for asset in assets:
        for index, line in enumerate(lines):
            if asset not in line:
                continue
            window = " ".join(lines[max(0, index - 1):index + 2])
            escaped = re.escape(asset)
            direct = bool(re.search(r"(?:modify|update|edit|refactor|fix|patch|overwrite|replace)\b.{0,100}" + escaped, window, re.I))
            in_place = bool(re.search(r"in.place|file to modify|modified:|modified in place|updated in place|you will modify", window, re.I))
            prohibited = bool(re.search(r"do not (?:modify|edit|change)|don't (?:modify|edit|change)|inputs? \(do not modify", window, re.I))
            different_output = bool(re.search(escaped + r".{0,160}(?:to|into|as|save|produce).{0,70}\boutputs?/", window, re.I))
            if (direct or in_place) and not prohibited and (in_place or not different_output):
                if asset in REVIEWED_WRITABLE_OUTPUTS.get(source_root.name, {}):
                    writable_evidence[asset] = {"source_text": window, "reviewed_gold_contract": REVIEWED_WRITABLE_OUTPUTS[source_root.name][asset]}
                else:
                    pending_writes[asset] = window
    # Directory-wide refactoring can authorize existing source changes without
    # repeating each filename. Such unresolved scopes stay pending as well.
    for asset in assets:
        if asset in writable_evidence or asset in pending_writes:
            continue
        for parent in Path(asset).parents:
            scope = parent.as_posix()
            if scope == ".":
                continue
            match = re.search(r"(?:modify|refactor|fix|patch|overwrite)\b.{0,100}" + re.escape(scope) + r"/", query, re.I)
            if match and not re.search(r"do not|don't|new file|new module|outputs?/", match.group(0), re.I):
                pending_writes[asset] = match.group(0)
                break
    # Reviewed source clauses must still exist; edits to task text cannot grant
    # a writable role through a task-id allowlist alone.
    writable = sorted(writable_evidence)
    return {"version": "clawgym_source_capability_and_input_roles_v4_fix1", "source_query_sha256": hashlib_sha(query),
            "initial_input_hashes": assets, "immutable_inputs": sorted(set(assets) - set(writable)),
            "writable_inputs": writable, "writable_input_evidence": writable_evidence,
            "pending_writable_inputs": pending_writes, "capability_requirements": requirements,
            "capability_evidence": evidence, "offline_contract_evidence": offline_clauses,
            "provided_fixture_paths": sorted(assets), "checker_capability_scan": checker, "dependency_contract": dependencies,
            "reference_policy": "unchanged_source_checker_with_hash_guarded_immutable_inputs_and_explicit_writable_outputs"}


def hashlib_sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def input_integrity_violations(workspace: Path, contract: dict) -> list[str]:
    violations = []
    for name in contract["immutable_inputs"]:
        path = workspace / name
        if path.is_symlink() or not path.resolve().is_relative_to(workspace.resolve()) or not path.is_file() or sha256_file(path) != contract["initial_input_hashes"][name]:
            violations.append(name)
    return violations


def qualification(path: Path, *, contract: dict | None = None, source_identity: dict | None = None) -> list[str]:
    reasons = []
    root = path.parent
    if not (root / "reward/check.py").is_file():
        reasons.append("missing_official_checker")
    script = root / "reward/reward.sh"
    if not script.is_file() or script.read_text().strip() != REWARD_SH.strip():
        reasons.append("unsupported_official_reward_aggregation")
    if (root / "reward/check.py").is_file():
        contract = contract if contract is not None else source_contract(path)
        for capability in contract["capability_requirements"]:
            reasons.append("requires_unsupported_" + capability)
        if contract["pending_writable_inputs"]:
            reasons.append("writable_input_gold_contract_pending_review")
        deps = contract["dependency_contract"]
        if deps["unavailable_required_checker_modules"] or deps["unresolved_requested_execution_modules"]:
            reasons.append("required_third_party_dependency_contract_pending")
    hf_selected = bool(source_identity and source_identity.get("format") == "hf_selected2k"
        and source_identity.get("hf_revision") == HF_REVISION and source_identity.get("hf_sha256") == HF_SHA256
        and source_identity.get("selection_revision") == PINNED_REVISION
        and source_identity.get("official_selection_id") == path.parent.name
        and source_identity.get("hf_record_canonical_sha256") and source_identity.get("hf_record_sha256"))
    if ("clawgym_train" not in path.parts and not hf_selected) or "clawgym_eval" in path.parts:
        reasons.append("outside_official_rl_selected_train")
    data = json.loads(path.read_text())
    if not isinstance(data.get("user_query"), str) or not data["user_query"].strip():
        reasons.append("missing_user_query")
    if any(item.is_symlink() for item in root.rglob("*")):
        reasons.append("source_symlink_requires_review")
    return reasons


def aggregate_reward(raw: object) -> float:
    """Same mean/scalar as pinned reward.sh; malformed results are invalid, never 0."""
    if isinstance(raw, dict) and raw:
        try:
            values = [float(value) for value in raw.values()]
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("checker_result_not_finite_numeric_scores") from exc
    elif isinstance(raw, (int, float)):
        values = [float(raw)]
    else:
        raise ValueError("checker_result_not_finite_numeric_scores")
    if not values or any(not math.isfinite(value) for value in values):
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
        reasons = qualification(path, source_identity=task.get("source_identity"))
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
        self.input_contract = source_contract(path)
        if task.get("input_contract") != self.input_contract:
            raise ValueError("source_input_contract_changed")
        self.input_hashes = self.input_contract["initial_input_hashes"]
        self.file_tools = WorkspaceTools(self.workspace)
        self.messages = [{"role": "user", "content": json.loads(path.read_text())["user_query"]}]
        self.tools = self.file_tools.schemas
        self.finished, self.closed = False, False
        write_json(self.evidence_dir / "source_provenance.json", {
            "source_path": str(path), "source_sha256": task["source_sha256"],
            "source_identity": task.get("source_identity", {"format": "pinned_git_selected2k", "revision": PINNED_REVISION}),
            "source_tree_sha256": task.get("source_tree_sha256"),
            "checker_tree_sha256": tree_digest(self.source_dir / "reward"),
            "adaptation": "unchanged_check.py_in_private_grader_invoked_with_frozen_workspace_snapshot",
            "reward_aggregation": "official_reward_sh_numeric_dict_mean_or_scalar",
            "input_contract": self.input_contract})

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
            changed_inputs = input_integrity_violations(self.workspace, self.input_contract)
            if changed_inputs:
                raise ValueError("immutable_grading_reference_modified:" + ",".join(changed_inputs))
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
                                  "source_protocol": "clawgym_private_grader_adaptation_v4",
                                  "source_format": self.task.get("source_identity", {}).get("format", "pinned_git_selected2k")},
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
