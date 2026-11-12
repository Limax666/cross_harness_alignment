"""Freeze a small validation calibration, not a formal baseline estimate."""
import copy
import re

MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
FAMILIES = {
    "workspace:user_task_3": "calendar_dinner_with_clients_injection",
    "workspace:user_task_17": "email_hiking_injection",
}
PAYLOAD_HASH = "a7347a07f90a014947fc4767972ee0867ebc4a6430e7d72e0867ebc4a2fecf80b8e"
# Corrected below to the 64-character digest from the qualified fixtures.
PAYLOAD_HASH = "a7347a07f90a014947fc4767972ee0867ebc4a6430e7d72e8e4b2a4fecf80b8e"


def build_calibration_plan(qualifications, *, snapshot_id, policy_sha256, code_hashes):
    plan = {
        "status": "validation_calibration_only_not_formal_baseline",
        "model": "Qwen/Qwen3.5-4B", "model_revision": MODEL_REVISION,
        "snapshot_id": snapshot_id, "policy_sha256": policy_sha256,
        "code_hashes": copy.deepcopy(code_hashes),
        "generation": {"temperature": 0, "max_model_len": 8192, "max_tokens": 512,
                       "max_turns": 6, "enable_thinking": False},
        "qualifications": copy.deepcopy(qualifications),
        "samples": [{"family": family, "arm": arm, "seed": seed}
                    for family in FAMILIES for arm in ("clean", "attack") for seed in (11, 29)],
    }
    validate_calibration_plan(plan)
    return plan


def validate_calibration_plan(plan):
    def require(condition):
        if not condition:
            raise ValueError("invalid calibration plan")

    digest = lambda value: isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    require(plan.get("status") == "validation_calibration_only_not_formal_baseline")
    require(plan.get("model") == "Qwen/Qwen3.5-4B" and plan.get("model_revision") == MODEL_REVISION)
    require(digest(plan.get("snapshot_id")) and digest(plan.get("policy_sha256")))
    require(isinstance(plan.get("code_hashes"), dict) and bool(plan["code_hashes"]))
    require(all(isinstance(k, str) and k and digest(v) for k, v in plan["code_hashes"].items()))
    require(plan.get("generation") == {"temperature": 0, "max_model_len": 8192, "max_tokens": 512,
                                      "max_turns": 6, "enable_thinking": False})
    quals = plan.get("qualifications")
    require(isinstance(quals, list) and len(quals) == len(FAMILIES))
    require({q.get("family") for q in quals} == set(FAMILIES))
    for q in quals:
        require(q.get("split") == "validation" and q.get("attack_vector") == FAMILIES[q["family"]])
        require(q.get("attack_payload_sha256") == PAYLOAD_HASH)
        require(digest(q.get("manifest_sha256")) and digest(q.get("tool_schema_sha256")))
        for key in ("same_user_task", "same_tool_schemas", "required_facts_visible_both", "reset_verified_both", "attack_oracle_positive", "utility_oracle_positive_both"):
            require(q.get(key) is True)
        require(q.get("attack_oracle_negative") is False)
        require(q.get("status") == "calibration_candidate_only_not_formal_admission")
        require(q.get("historical_development_exposure") == "unknown")
    require(len({q["manifest_sha256"] for q in quals}) == 1)
    require(len({q["tool_schema_sha256"] for q in quals}) == 1)
    samples = plan.get("samples")
    require(isinstance(samples, list) and len(samples) == 8)
    require(all(set(s) == {"family", "arm", "seed"} for s in samples))
    require({(s["family"], s["arm"], s["seed"]) for s in samples} ==
            {(f, a, s) for f in FAMILIES for a in ("clean", "attack") for s in (11, 29)})
