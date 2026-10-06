"""Fail closed before the one-update AgentDojo online GRPO pilot.

Inputs are produced by the trusted rollout/audit adapter, never by model text.
This is an admission check, not a rollout implementation or a reward model.
"""
from __future__ import annotations

from collections import defaultdict
from math import isfinite

# Registered online-pilot cells. Expanded from the original four workspace
# cells (2026-10-05) to eight train families across all four AgentDojo
# suites, two families per suite, clean + injection_task_1 arms each: the
# ±1/0 outcome reward saturated on the four-cell pool (zero-variance group
# rate 0.25→1.0 within five updates, run r16), so every registered family
# must come from the frozen accepted-train split (enforced by
# build_rl_native_pilot_dataset.py).
_CELL_FAMILIES = (
    ("workspace", (5, 6)),
    ("banking", (0, 2)),
    ("travel", (11, 13)),
    ("slack", (0, 1)),
)
CELLS = frozenset(
    f"{suite}:user_task_{task}:{arm}"
    for suite, tasks in _CELL_FAMILIES
    for task in tasks
    for arm in ("clean", "injection_task_1")
)


class PilotAdmissionError(ValueError):
    pass


def require_online_batch(rows: list[dict], *, policy_snapshot: str) -> dict:
    """Check every rollout group; never select only favourable trajectories.

    Token origin is recorded when generation/context tokens are appended, not
    reconstructed by parsing assistant text. Old-policy probabilities may be
    recomputed by VeRL, so numerical equality with sampler probabilities is
    deliberately not required (their difference must be monitored separately).
    """
    def require(condition, message):
        if not condition:
            raise PilotAdmissionError(message)

    require(bool(policy_snapshot), "missing policy snapshot identity")
    require(len(rows) == 4 * len(CELLS), "expected one group of G=4 rollouts per registered cell")
    groups = defaultdict(list)
    ids = set()
    for row in rows:
        rid = row.get("rollout_id")
        require(isinstance(rid, str) and rid and rid not in ids, "missing/duplicate rollout ID")
        ids.add(rid)
        require(row.get("task_id") in CELLS, "unexpected task cell")
        require(row.get("split") == "train", "non-train family")
        require(row.get("harness") == "hermes", "unexpected harness")
        require(row.get("on_policy_rollout") is True, "teacher/replay data prohibited")
        require(row.get("policy_snapshot") == policy_snapshot, "mixed/stale policy snapshot")
        require(row.get("reset_verified") is True, "environment reset not verified")
        require(row.get("harness_contract_verified") is True, "harness contract not verified")
        require(row.get("audit_complete") is True, "incomplete independent audit")
        reward = row.get("reward")
        require(type(reward) in (float, int) and isfinite(reward) and reward in (-1, 0, 1),
                "invalid outcome-first reward")
        seed = row.get("seed")
        require(type(seed) is int and seed >= 0, "missing rollout seed")
        initial = row.get("initial_state_sha256")
        require(isinstance(initial, str) and len(initial) == 64, "missing initial-state digest")
        token_ids = row.get("response_ids", [])
        masks = row.get("response_mask", [])
        origins = row.get("token_origins", [])
        sampled = row.get("rollout_log_probs", [])
        old = row.get("old_log_probs", [])
        n = len(token_ids)
        require(n > 0 and all(len(x) == n for x in (masks, origins, sampled, old)),
                "token/mask/log-prob lengths differ")
        require(all(type(t) is int and t >= 0 for t in token_ids), "invalid token IDs")
        require(all(type(m) is int and m in (0, 1) for m in masks) and any(masks),
                "invalid/empty assistant mask")
        for mask, origin, sample_lp, old_lp in zip(masks, origins, sampled, old):
            require(origin in ("assistant", "context"), "unknown token origin")
            require(mask == int(origin == "assistant"), "context token contributes to policy loss")
            if mask:
                require(all(type(v) in (int, float) and isfinite(v) and v <= 1e-5
                            for v in (sample_lp, old_lp)), "invalid assistant log probability")
        groups[row["task_id"]].append(row)

    require(set(groups) == CELLS, "missing task cells")
    variable_groups = 0
    rewards_by_cell = {}
    for cell, group in sorted(groups.items()):
        require(len(group) == 4, "each cell must contain G=4")
        require(len({r["seed"] for r in group}) == 4, "duplicate seed within group")
        require(len({r["initial_state_sha256"] for r in group}) == 1,
                "independent resets differ within task cell")
        rewards = [float(r["reward"]) for r in group]
        rewards_by_cell[cell] = rewards
        variable_groups += len(set(rewards)) > 1
    require(variable_groups > 0, "all groups have zero outcome advantage; stop without update")
    return {"admitted": True, "rollouts": 4 * len(CELLS), "groups": len(CELLS),
            "variable_reward_groups": variable_groups,
            "rewards_by_cell": rewards_by_cell,
            "engineering_only": True, "policy_snapshot": policy_snapshot}
