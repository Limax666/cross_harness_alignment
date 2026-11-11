"""Reward contract used by matched GRPO and CHS-PO experiments."""


def paired_reward_profile(algorithm, config):
    if algorithm not in ("GRPO-progress-v1", "CHS-PO-progress-v1"):
        raise ValueError("paired comparison requires the progress objectives")
    contract = config.get("reward_contract")
    variant = config.get("reward_variant")
    path_weight = config.get("path_weight")
    if contract != "verified_progress_v1" or variant not in ("outcome", "progress", "progress_protocol") or path_weight != 0:
        raise ValueError("paired comparison requires an explicit shared verified reward contract")
    from rl_progress_registry import DEFAULT, resolve
    oracle = config.get("progress_oracle_version", DEFAULT)
    resolve(oracle)
    return {"contract": contract, "variant": variant, "path_weight": path_weight,
            "progress_oracle_version": oracle}


def validate_paired_profiles(grpo_config, chspo_config):
    grpo = paired_reward_profile("GRPO-progress-v1", grpo_config)
    chspo = paired_reward_profile("CHS-PO-progress-v1", chspo_config)
    if grpo != chspo:
        raise ValueError("GRPO and CHS-PO must use identical reward profiles")
    return grpo
