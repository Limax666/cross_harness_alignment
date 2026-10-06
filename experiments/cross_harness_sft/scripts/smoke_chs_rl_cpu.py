#!/usr/bin/env python3
"""CPU-only end-to-end mechanics check; this is not a native-harness RL run.

The fixture actually resets directories, executes simple side-effecting actions,
verifies final files, constructs GRPO groups, and updates a tiny torch policy.
It is intentionally distinct from the Hermes/NanoBot/OpenClaw environments.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from rl_verifier import EpisodeEvidence, TaskKind, group_advantages, score_episode


HARNESS_NAMES = ("hermes", "nanobot", "openclaw")
KINDS = (TaskKind.BENIGN, TaskKind.INJECTION, TaskKind.DIRECT_HARM)


def run_fixture(harness: str, kind: TaskKind, action: int, root: Path) -> EpisodeEvidence:
    """0=refuse, 1=finish legitimate task, 2=write prohibited state."""
    episode = root / f"{harness}-{kind.value}"
    episode.mkdir()
    if action == 1 and kind != TaskKind.DIRECT_HARM:
        (episode / "legitimate_done.txt").write_text("done", encoding="utf-8")
    elif action == 2:
        (episode / "prohibited_persistence.txt").write_text("unsafe", encoding="utf-8")
    return EpisodeEvidence(
        harness=harness, family_id=f"fixture-{kind.value}", task_kind=kind,
        termination="completed", final_state_verified=True, path_verified=True,
        legitimate_goal_complete=(episode / "legitimate_done.txt").exists(),
        attacker_goal_achieved=(episode / "prohibited_persistence.txt").exists(),
        prohibited_persistence=(episode / "prohibited_persistence.txt").exists(),
        harmful_action_executed=(episode / "prohibited_persistence.txt").exists(),
        invalid_tool_dispatched=False, appropriate_refusal=(action == 0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--group-size", type=int, default=4)
    args = parser.parse_args()
    if args.group_size < 2:
        parser.error("--group-size must be at least 2")
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)
    logits = torch.nn.Parameter(torch.zeros(len(KINDS), 3, dtype=torch.float32))
    reference = logits.detach().clone()
    optimizer = torch.optim.SGD([logits], lr=0.05)
    records = []
    losses = []
    zero_groups = 0
    with tempfile.TemporaryDirectory(prefix="chs-rl-cpu-") as scratch:
        root = Path(scratch)
        for harness in HARNESS_NAMES:
            for kind_index, kind in enumerate(KINDS):
                probs = torch.softmax(logits[kind_index].detach(), dim=-1)
                actions = torch.multinomial(probs, args.group_size, replacement=True, generator=generator)
                rewards = []
                for rollout_index, action in enumerate(actions.tolist()):
                    episode_root = root / f"{harness}-{kind.value}-{rollout_index}"
                    episode_root.mkdir()
                    evidence = run_fixture(harness, kind, action, episode_root)
                    verdict = score_episode(evidence)
                    assert verdict.scorable
                    rewards.append(verdict.score)
                advantages = group_advantages(rewards)
                if all(value == 0.0 for value in advantages):
                    zero_groups += 1
                old_logprobs = torch.log_softmax(logits[kind_index].detach().clone(), dim=-1)[actions]
                new_logprobs = torch.log_softmax(logits[kind_index], dim=-1)[actions]
                ratio = (new_logprobs - old_logprobs).exp()
                advantage_tensor = torch.tensor(advantages)
                surrogate = torch.minimum(
                    ratio * advantage_tensor,
                    ratio.clamp(0.8, 1.2) * advantage_tensor,
                ).mean()
                ref_probs = torch.softmax(reference[kind_index], dim=-1)
                current_probs = torch.softmax(logits[kind_index], dim=-1)
                kl = (current_probs * (current_probs.log() - ref_probs.log())).sum()
                losses.append(-surrogate + 0.01 * kl)
                records.append({"harness": harness, "kind": kind.value, "rewards": rewards,
                                "advantages": advantages, "actions": actions.tolist()})
        optimizer.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        optimizer.step()
        assert torch.isfinite(logits).all()
        assert not torch.equal(logits.detach(), reference), "policy did not update"
        assert len(records) == len(HARNESS_NAMES) * len(KINDS)
    print(json.dumps({"status": "cpu_mechanics_passed", "fixture_groups": len(records),
                      "zero_advantage_groups": zero_groups, "policy_changed": True,
                      "native_harness_tested": False, "records": records}, ensure_ascii=False))


if __name__ == "__main__":
    main()
