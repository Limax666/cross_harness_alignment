"""Adversarial admission fixtures; these are not online RL trajectories."""
from copy import deepcopy

import pytest

from rl_online_batch_gate import CELLS, PilotAdmissionError, require_online_batch


def valid_fixture():
    return [dict(
        rollout_id=f"fixture-{cell}-{seed}", task_id=cell, split="train",
        harness="hermes", on_policy_rollout=True, policy_snapshot="fixture-policy",
        reset_verified=True, harness_contract_verified=True, audit_complete=True,
        reward=seed % 2, seed=seed, initial_state_sha256="a" * 64,
        response_ids=[11, 22, 33], response_mask=[1, 0, 1],
        token_origins=["assistant", "context", "assistant"],
        rollout_log_probs=[-1.0, 0.0, -2.0], old_log_probs=[-1.01, 0.0, -2.01],
    ) for cell in sorted(CELLS) for seed in range(4)]


def test_keeps_constant_groups_alongside_variable_group():
    rows = valid_fixture()
    for row in rows[4:]:
        row["reward"] = -1
    result = require_online_batch(rows, policy_snapshot="fixture-policy")
    assert result["rollouts"] == 4 * len(CELLS) and result["variable_reward_groups"] == 1


@pytest.mark.parametrize("mutation", [
    {"audit_complete": False}, {"on_policy_rollout": False},
    {"policy_snapshot": "stale"}, {"reset_verified": False},
    {"harness_contract_verified": False}, {"split": "validation"},
    {"response_mask": [1, 1, 1]}, {"old_log_probs": [-1.0]},
    {"rollout_log_probs": [float("nan"), 0.0, -1.0]},
    {"reward": None}, {"reward": 0.5}, {"initial_state_sha256": "b" * 64},
])
def test_rejects_untrusted_or_misaligned_evidence(mutation):
    rows = valid_fixture()
    rows[0].update(mutation)
    with pytest.raises(PilotAdmissionError):
        require_online_batch(rows, policy_snapshot="fixture-policy")


def test_zero_advantage_does_not_trigger_kl_only_update():
    rows = valid_fixture()
    for row in rows:
        row["reward"] = 0
    with pytest.raises(PilotAdmissionError, match="zero outcome advantage"):
        require_online_batch(rows, policy_snapshot="fixture-policy")


def test_no_selective_dropping_or_duplicate_rollouts():
    rows = valid_fixture()
    for bad in (rows[:-1], rows + [deepcopy(rows[0])], [rows[0]] + rows[:-1]):
        with pytest.raises(PilotAdmissionError):
            require_online_batch(bad, policy_snapshot="fixture-policy")


def test_no_duplicate_seeds_in_group():
    rows = valid_fixture()
    rows[1]["seed"] = rows[0]["seed"]
    with pytest.raises(PilotAdmissionError, match="duplicate seed"):
        require_online_batch(rows, policy_snapshot="fixture-policy")
