import json
from pathlib import Path

import pytest

from flashrsi_b0_qualification import validate_family_split, qualify_calibration_pair


def test_split_gate_rejects_train_family_and_cross_split_duplicate(tmp_path):
    manifest = tmp_path / "families.jsonl"
    rows = [
        {"benchmark_namespace": "agentdojo", "source_family_id": "agentdojo:workspace:user_task_3",
         "split": "validation", "source_prompt_sha256": ["a" * 64]},
        {"benchmark_namespace": "agentdojo", "source_family_id": "agentdojo:workspace:user_task_6",
         "split": "train", "source_prompt_sha256": ["b" * 64]},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert validate_family_split(manifest, "workspace:user_task_3")["split"] == "validation"
    with pytest.raises(ValueError):
        validate_family_split(manifest, "workspace:user_task_6")
    rows[1]["source_prompt_sha256"] = ["a" * 64]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        validate_family_split(manifest, "workspace:user_task_3")


@pytest.mark.parametrize("family", ["workspace:user_task_3", "workspace:user_task_17"])
def test_real_validation_pair_qualifies_with_correct_facts_and_counterexamples(family):
    from cross_harness_sft.backends.agentdojo import AgentDojoDriver
    root = Path(__file__).resolve().parents[1]
    manifest = root.parents[1] / "hf_transfer/extracted/data/sft/teacher_release_v7/family_split_manifest.jsonl"
    if not manifest.is_file():
        pytest.skip("private handoff manifest is unavailable")
    driver = AgentDojoDriver(str(root / "configs/agentdojo_worker.yaml"))
    result = qualify_calibration_pair(driver, family, manifest)
    assert result["split"] == "validation"
    assert result["same_user_task"] and result["same_tool_schemas"]
    assert result["required_facts_visible_both"] and result["reset_verified_both"]
    assert result["attack_oracle_negative"] is False and result["attack_oracle_positive"] is True
    assert result["utility_oracle_positive_both"]
    assert result["historical_development_exposure"] == "unknown"


def test_qualification_rejects_unknown_family(tmp_path):
    with pytest.raises(ValueError):
        qualify_calibration_pair(None, "workspace:user_task_6", tmp_path / "missing")
