import copy

import pytest

from flashrsi_b0_plan import build_calibration_plan, validate_calibration_plan


def qualification(family, vector):
    return {"family": family, "split": "validation", "attack_vector": vector,
            "attack_payload_sha256": "a7347a07f90a014947fc4767972ee0867ebc4a6430e7d72e8e4b2a4fecf80b8e",
            "manifest_sha256": "a" * 64, "tool_schema_sha256": "b" * 64,
            "same_user_task": True, "same_tool_schemas": True, "required_facts_visible_both": True,
            "reset_verified_both": True, "attack_oracle_negative": False,
            "attack_oracle_positive": True, "utility_oracle_positive_both": True,
            "historical_development_exposure": "unknown", "status": "calibration_candidate_only_not_formal_admission"}


def test_frozen_plan_contains_every_arm_and_seed():
    q = [qualification("workspace:user_task_3", "calendar_dinner_with_clients_injection"),
         qualification("workspace:user_task_17", "email_hiking_injection")]
    plan = build_calibration_plan(q, snapshot_id="c" * 64, policy_sha256="d" * 64,
                                  code_hashes={"scripts/example.py": "e" * 64})
    validate_calibration_plan(plan)
    assert len(plan["samples"]) == 8
    assert {(x["family"], x["arm"], x["seed"]) for x in plan["samples"]} == {
        (family, arm, seed) for family in ("workspace:user_task_3", "workspace:user_task_17")
        for arm in ("clean", "attack") for seed in (11, 29)}


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_seed", "wrong_split", "wrong_revision", "unqualified"])
def test_frozen_plan_rejects_tampering(change):
    q = [qualification("workspace:user_task_3", "calendar_dinner_with_clients_injection"),
         qualification("workspace:user_task_17", "email_hiking_injection")]
    p = build_calibration_plan(q, snapshot_id="c" * 64, policy_sha256="d" * 64,
                               code_hashes={"scripts/example.py": "e" * 64})
    p = copy.deepcopy(p)
    if change == "missing": p["samples"].pop()
    if change == "duplicate": p["samples"].append(copy.deepcopy(p["samples"][0]))
    if change == "wrong_seed": p["samples"][0]["seed"] = 999
    if change == "wrong_split": p["qualifications"][0]["split"] = "train"
    if change == "wrong_revision": p["model_revision"] = "other"
    if change == "unqualified": p["qualifications"][0]["required_facts_visible_both"] = False
    with pytest.raises(ValueError):
        validate_calibration_plan(p)
