from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rl_online_batch_gate import CELLS, PilotAdmissionError
from rl_verl_update_guard import require_verl_update_batch


def batch_fixture():
    evidence = []
    for cell in sorted(CELLS):
        for seed in range(4):
            evidence.append({
                "rollout_id": f"{cell}/{seed}", "task_id": cell, "split": "train",
                "harness": "hermes", "on_policy_rollout": True, "policy_snapshot": "sft-step-267",
                "reset_verified": True, "harness_contract_verified": True,
                "audit_complete": True, "seed": seed, "initial_state_sha256": "a" * 64,
                "reward": 1 if seed == 0 else 0,
                "token_origins": ["assistant", "context"],
            })
    n = len(evidence)
    uids = [f"group:{item['task_id']}" for item in evidence]
    return SimpleNamespace(batch={
        "responses": [[10, 11, 0] for _ in range(n)],
        "response_mask": [[1, 0, 0] for _ in range(n)],
        "attention_mask": [[1, 1, 1, 1, 0] for _ in range(n)],
        "rollout_log_probs": [[-0.5, 0, 0] for _ in range(n)],
        "old_log_probs": [[-0.5, 0, 0] for _ in range(n)],
        "rm_scores": [[0, e["reward"], 0] for e in evidence],
    }, non_tensor_batch={"pilot_evidence": evidence, "uid": uids})


def test_verified_verl_batch_admitted():
    result = require_verl_update_batch(batch_fixture(), policy_snapshot="sft-step-267")
    assert result["rollouts"] == 4 * len(CELLS)
    assert result["variable_reward_groups"] == len(CELLS)


def test_reward_tampering_is_rejected_before_update():
    batch = batch_fixture()
    batch.batch["rm_scores"][0][1] = -1
    with pytest.raises(PilotAdmissionError, match="reward differs"):
        require_verl_update_batch(batch, policy_snapshot="sft-step-267")


def test_missing_sampled_logprobs_is_rejected_before_update():
    batch = batch_fixture()
    del batch.batch["rollout_log_probs"]
    with pytest.raises(PilotAdmissionError, match="missing VeRL token"):
        require_verl_update_batch(batch, policy_snapshot="sft-step-267")


def test_unscorable_row_is_rejected_before_update():
    batch = batch_fixture()
    batch.non_tensor_batch["pilot_evidence"][0]["audit_complete"] = False
    with pytest.raises(PilotAdmissionError, match="incomplete independent audit"):
        require_verl_update_batch(batch, policy_snapshot="sft-step-267")


def test_tool_token_mislabeled_as_assistant_is_rejected():
    batch = batch_fixture()
    batch.batch["response_mask"][0][1] = 1
    with pytest.raises(PilotAdmissionError, match="context token contributes"):
        require_verl_update_batch(batch, policy_snapshot="sft-step-267")
