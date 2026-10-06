import pytest

from rl_guarded_verl_trainer import admitted_batch_counts


def test_counts_follow_admitted_sixteen_cell_batch():
    counts = admitted_batch_counts({"groups": 16, "rollouts": 64, "variable_reward_groups": 6}, 64)
    assert counts == {
        "sampled_groups": 16,
        "sampled_rollouts": 64,
        "scored_rollouts": 64,
        "unscorable_rollouts": 0,
        "zero_variance_group_rate": 0.625,
    }


def test_inconsistent_counts_are_rejected():
    with pytest.raises(ValueError, match="inconsistent"):
        admitted_batch_counts({"groups": 16, "rollouts": 64, "variable_reward_groups": 17}, 64)
    with pytest.raises(ValueError, match="inconsistent"):
        admitted_batch_counts({"groups": 16, "rollouts": 64, "variable_reward_groups": 6}, 16)
