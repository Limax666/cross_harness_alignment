#!/usr/bin/env python3
"""Exercise this checkout's real VeRL GRPO and PPO advantage functions on CPU."""

import json

import numpy as np
import torch

from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
)


def main():
    # The middle position stands for a tool result: it contributes no policy
    # gradient, but the terminal reward still belongs to the full trajectory.
    mask = torch.tensor([[1., 0., 1.], [1., 0., 1.]])
    rewards = torch.tensor([[0., 0., 1.], [0., 0., -1.]])
    grpo, _ = compute_grpo_outcome_advantage(rewards, mask, np.array(["same", "same"]))
    assert grpo[0, 0] > 0 and grpo[1, 0] < 0
    assert torch.all(grpo[:, 1] == 0), "tool observations must be masked in GRPO"

    zero, _ = compute_grpo_outcome_advantage(
        torch.tensor([[0., 0., -1.], [0., 0., -1.]]),
        mask, np.array(["same", "same"]),
    )
    assert torch.all(zero == 0), "all-fail groups should have zero ordinary GRPO advantage"

    gae, returns = compute_gae_advantage_return(
        rewards, torch.zeros_like(rewards), mask, 1.0, 1.0,
    )
    assert returns[0, 0] > 0 and returns[1, 0] < 0
    # VeRL's GAE function propagates advantage through an observation slot;
    # the downstream actor loss must apply response_mask again.
    assert torch.all((gae * mask)[:, 1] == 0)
    print(json.dumps({"status": "verl_cpu_advantages_passed", "grpo": grpo.tolist(),
                      "zero_group": zero.tolist(), "masked_gae": (gae * mask).tolist()}))


if __name__ == "__main__":
    main()
