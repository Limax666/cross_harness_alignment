"""Focused real-source metadata and data-only checker checks; no episodes/APIs."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import pytest

from hsrl_v3.audit_backend import SQL_COHORT, qualification as audit_qualification, tool_schema
from hsrl_v3.backends import checked_source, open_backend
from hsrl_v3.claw_backend import aggregate_reward, qualification as claw_qualification
from hsrl_v3.pool import prepare_pool, prepare_reference_pool_v3
from hsrl_v3.safe_backend import IMPLEMENTED_UTILITY, accept_utility_checker

PROJECT = Path(os.environ.get("HSRL_SOURCE_PROJECT_ROOT", "/data/home/liumingxiao/cross_harness_alignment"))
EXP = PROJECT / "experiments/cross_harness_sft"
CLAW = EXP / "vendor/clawgym_agents_hsrl_v3_0e8526d"
SAFE = EXP / "vendor/harnesssafe/runs/active"


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    return prepare_reference_pool_v3(PROJECT, CLAW, tmp_path_factory.mktemp("actual_source_pool_v3_reference"))


def test_real_asset_inventory_and_no_source_replacement(manifest):
    assert manifest["source_counts"] == {"clawgym": 2000, "harnesssafe": 328, "harnessaudit": 34}
    assert manifest["source_weights"] == {"clawgym": .75, "harnesssafe": .125, "harnessaudit": .125}
    identities = [(row["source"], row["id"], row["condition"]) for row in manifest["tasks"]]
    assert len(identities) == len(set(identities))
    assert all("clawgym_eval" not in row["source_path"] for row in manifest["tasks"])
    assert all(row["exclusion_reasons"] for row in manifest["exclusions"])


def test_split_precedes_condition_expansion_and_no_family_crossing(manifest):
    splits, pairs = defaultdict(set), defaultdict(set)
    for row in manifest["tasks"]:
        splits[row["source"], row["family"]].add(row["split"])
        pairs[row["source"], row["id"]].add((row["condition"], row["split"]))
        assert row["id"] != row["family"]
    assert all(len(value) == 1 for value in splits.values())
    for (source, _), conditions in pairs.items():
        if source == "harnesssafe" and len(conditions) > 1:
            assert {condition for condition, _ in conditions} == {"clean", "attack"}
            assert len({split for _, split in conditions}) == 1
    for source in ("harnesssafe", "harnessaudit"):
        assert {row["split"] for row in manifest["tasks"] if row["source"] == source and row["qualified"]} == {"train", "dev", "test"}


def test_every_real_source_hash_is_frozen(manifest):
    # Full trees were fingerprinted during preparation; sample one actual tree
    # per source through the same backend admission gate (no tool execution).
    for source in manifest["source_counts"]:
        row = next(row for row in manifest["tasks"] if row["source"] == source and row["qualified"])
        assert checked_source(row).is_file()
        broken = {**row, "source_sha256": "0" * 64}
        with pytest.raises(ValueError, match="source_hash_changed"):
            checked_source(broken)


def test_unqualified_official_source_is_not_silently_scored(manifest, tmp_path):
    row = next(row for row in manifest["exclusions"] if row["source"] == "harnesssafe")
    with pytest.raises(ValueError, match="task_not_qualified"):
        with open_backend(row, tmp_path / "workspace", tmp_path / "evidence"):
            pytest.fail("unqualified backend entered")


@pytest.mark.parametrize("case", sorted(IMPLEMENTED_UTILITY))
def test_independent_safe_utility_real_input_positive_and_negatives(case):
    path = next(SAFE.glob("**/" + case + "/case_meta.json"))
    evidence = accept_utility_checker(path.parent)
    assert evidence["kind"] == "data_only_checker_acceptance_not_episode"
    assert evidence["valid"], evidence
    assert all(evidence["checks"].values())
    assert evidence["official_clean_skill_sha256"]


def test_audit_qualification_rejects_judge_and_tool_only_completion():
    from single_agent.loader import load_sa_task_with_tools
    admitted, reasons = set(), Counter()
    for path in (PROJECT / "HarnessAudit/single_agent/tasks").rglob("*.yaml"):
        task, catalog = load_sa_task_with_tools(path)
        failures = audit_qualification(task)
        if not failures:
            admitted.add(task.task_id)
            assert all(cp["type"] == "rule" for cp in task.completion_checkpoints)
        reasons.update(failures)
        for tool in catalog.tools:
            schema = tool_schema(tool)["function"]
            assert schema["name"] == tool.name
            assert schema["parameters"]["required"] == list(tool.params)
            assert schema["parameters"]["additionalProperties"] is False
    assert admitted == SQL_COHORT
    assert reasons["requires_frozen_completion_judge"] == 16


def test_claw_unchanged_reward_aggregation_and_strict_failure():
    source = CLAW / "RL/data/clawgym_train/task_0000/data_entry.json"
    assert claw_qualification(source) == []
    assert aggregate_reward({"a": 1, "b": .5, "c": 0}) == .5
    assert aggregate_reward(.5) == .5
    for malformed in ({}, {"score": float("nan")}, {"score": "error"}, [], True, 2):
        with pytest.raises(ValueError):
            aggregate_reward(malformed)


@pytest.fixture(scope="module")
def manifest_v4(tmp_path_factory):
    return prepare_pool(PROJECT, CLAW, tmp_path_factory.mktemp("actual_source_pool_v4"))


def test_v4_actual_lifecycle_inventory_and_fixed_mixture(manifest_v4):
    manifest = manifest_v4
    assert manifest["schema_version"] == 4
    assert manifest["source_counts"] == {"clawgym": 2000, "skilllifebench": 60}
    assert manifest["condition_counts"]["skilllifebench"] == 120
    assert manifest["source_weights"] == {"clawgym": .75, "skilllifebench": .25}
    assert manifest["source_condition_weights"]["skilllifebench"] == {"clean": .5, "attack": .5}
    assert manifest["reference_sources"]["harnesssafe"]["weight"] == 0
    assert manifest["reference_sources"]["harnessaudit"]["weight"] == 0
    assert {row["source"] for row in manifest["train"]} == {"clawgym", "skilllifebench"}
    assert manifest["runtime_admission_verified"] is False


def test_v4_exact_frozen_lifecycle_template_split(manifest_v4):
    split_path = Path(__file__).resolve().parents[1] / "docs/research/skilllife_lifecycle_split_20261116.json"
    frozen = {row["id"]: row for row in json.loads(split_path.read_text())["tasks"]}
    by_family = defaultdict(set)
    conditions = defaultdict(set)
    counts = Counter()
    for row in manifest_v4["tasks"]:
        if row["source"] != "skilllifebench":
            continue
        assert row["split"] == frozen[row["id"]]["split"]
        assert row["template_id"] == frozen[row["id"]]["template_id"]
        assert row["source_record_sha256"]
        assert row["source_code_hashes"]
        by_family[row["family"]].add(row["split"])
        conditions[row["id"]].add(row["condition"])
        counts[row["split"]] += 1
    assert len(by_family) == 12
    assert all(len(splits) == 1 for splits in by_family.values())
    assert all(value == {"clean", "attack"} for value in conditions.values())
    assert counts == {"train": 60, "dev": 30, "test": 30}


def test_v4_claw_optional_inputs_and_real_capability_exclusions(manifest_v4):
    by_id = {row["id"]: row for row in manifest_v4["tasks"] if row["source"] == "clawgym"}
    for task_id in ("task_0176", "task_0177", "task_0678", "task_1275", "task_1909"):
        assert not (Path(by_id[task_id]["source_path"]).parent / "input_files").exists()
        assert by_id[task_id]["qualified"]
    assert not by_id["task_0398"]["qualified"]
    assert "live_external_download" in by_id["task_0398"]["exclusion_reasons"][0]
    assert "Internet essential" in by_id["task_0398"]["exclusion_evidence"]["user_query"]
    assert not by_id["task_0545"]["qualified"]
    assert manifest_v4["qualified_instance_counts"]["clawgym"] == 1992


def test_v4_official_source_interfaces_and_context_authority():
    import inspect
    from hsrl_v3.skilllife_backend import CONTEXT_TOOL, import_source, qualification
    root = EXP / "vendor/skilllifebench_security_research_20261116"
    backend, executor = import_source(root)
    assert list(inspect.signature(backend.LifecycleBackend).parameters) == ["task", "condition", "output_dir", "executor"]
    assert list(inspect.signature(backend.LifecycleBackend.execute).parameters) == ["self", "name", "arguments", "call_id"]
    assert list(inspect.signature(backend.LifecycleBackend.finalize).parameters) == ["self", "final_output", "stop_reason"]
    assert executor.SandboxLimits().wall_seconds == 5.0
    assert executor.SandboxLimits().cpu_seconds == 2
    assert CONTEXT_TOOL["function"]["name"] == "read_file"
    assert set(CONTEXT_TOOL["function"]["parameters"]["properties"]["path"]["enum"]) == {
        ".hsrl/context_summary.txt", "/root/.openclaw/workspace/.hsrl/context_summary.txt"}
    # Source schemas are examined as metadata, without entering an executor or
    # running any reference actions, qualification replay, model or source tool.
    tasks = json.loads((root / "data/lifecycle/protocol/task-manifest.json").read_text())["tasks"]
    assert all(qualification(task) == [] for task in tasks)
