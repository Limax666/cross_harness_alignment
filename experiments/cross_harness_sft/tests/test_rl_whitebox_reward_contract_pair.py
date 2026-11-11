import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rl_whitebox_reward_contract_pair import paired_reward_profile, validate_paired_profiles


def test_grpo_and_chspo_share_identical_verifier_reward_profile():
    config = {"reward_contract": "verified_progress_v1", "reward_variant": "outcome", "path_weight": 0.0}
    assert paired_reward_profile("GRPO-progress-v1", config) == paired_reward_profile("CHS-PO-progress-v1", config)


def test_paired_profile_rejects_legacy_or_undefined_reward():
    for config in ({"reward_contract": "legacy", "path_weight": 0.0},
                   {"reward_contract": "verified_progress_v1", "reward_variant": "unknown", "path_weight": 0.0},
                   {"reward_contract": "verified_progress_v1", "reward_variant": "outcome", "path_weight": 0.25}):
        with pytest.raises(ValueError):
            paired_reward_profile("GRPO-progress-v1", config)


def test_comparison_rejects_different_reward_variants_or_oracles():
    base = {"reward_contract": "verified_progress_v1", "reward_variant": "outcome", "path_weight": 0.0,
            "progress_oracle_version": "agentdojo-workspace-progress-v3"}
    validate_paired_profiles(base, dict(base))
    with pytest.raises(ValueError):
        validate_paired_profiles(base, {**base, "reward_variant": "progress"})
    with pytest.raises(ValueError):
        validate_paired_profiles(base, {**base, "progress_oracle_version": "agentdojo-workspace-progress-v4-candidate"})
