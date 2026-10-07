"""Strict current-policy admission for frozen multi-harness train cells."""
from __future__ import annotations
from collections import defaultdict
from math import isfinite
from collections import Counter
from rl_multiharness_stratified_sampler import EXPECTED_STRATA, GROUPS_PER_STRATUM

class MultiharnessAdmissionError(ValueError):
    pass


def require_online_batch(rows: list[dict], *, policy_snapshot: str, cells: dict[str, dict]) -> dict:
    """Check every rollout group; never select only favourable trajectories.

    Token origin is recorded when generation/context tokens are appended, not
    reconstructed by parsing assistant text. Old-policy probabilities may be
    recomputed by VeRL, so numerical equality with sampler probabilities is
    deliberately not required (their difference must be monitored separately).
    """
    def require(condition, message):
        if not condition:
            raise MultiharnessAdmissionError(message)

    require(bool(policy_snapshot), "missing policy snapshot identity")
    selected_count = len(EXPECTED_STRATA) * GROUPS_PER_STRATUM
    require(len(rows) == 4 * selected_count, "expected G=4 rollouts for every sampled task")
    groups = defaultdict(list)
    ids = set()
    for row in rows:
        rid = row.get("rollout_id")
        require(isinstance(rid, str) and rid and rid not in ids, "missing/duplicate rollout ID")
        ids.add(rid)
        require(row.get("cell_id") in cells, "unexpected task cell")
        require(row.get("split") == "train", "non-train family")
        require(row.get("benchmark") == cells[row["cell_id"]]["benchmark"] and
                row.get("source_family_id") == cells[row["cell_id"]]["source_family_id"] and
                row.get("task_id") == cells[row["cell_id"]]["task_id"] and
                row.get("arm") == cells[row["cell_id"]]["arm"], "train cell provenance mismatch")
        require(row.get("harness") == cells[row["cell_id"]]["harness_name"], "unexpected harness")
        require(row.get("on_policy_rollout") is True, "teacher/replay data prohibited")
        require(row.get("policy_snapshot") == policy_snapshot, "mixed/stale policy snapshot")
        require(row.get("reset_verified") is True, "environment reset not verified")
        require(row.get("harness_contract_verified") is True, "harness contract not verified")
        require(row.get("audit_complete") is True, "incomplete independent audit")
        reward = row.get("reward")
        require(type(reward) in (float, int) and isfinite(reward) and -1.25 <= reward <= 1.25,
                "invalid outcome-first reward")
        # Reward must decompose into a discrete outcome plus the bounded,
        # independently audited path channel; anything else is not scorable.
        path_signal = row.get("reward_path_signal")
        require(type(path_signal) in (float, int) and isfinite(path_signal)
                and abs(path_signal) <= 0.25 + 1e-9, "missing or unbounded path signal")
        outcome_component = reward - path_signal
        require(any(abs(outcome_component - value) < 1e-6
                    for value in (-1.0, -0.5, 0.0, 1.0)),
                "reward does not decompose into outcome plus bounded path signal")
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
        groups[row["cell_id"]].append(row)

    require(len(groups) == selected_count, "missing or duplicate sampled task groups")
    strata = Counter((cells[cell]["benchmark"], cells[cell]["harness_name"], cells[cell]["arm"])
                     for cell in groups)
    require(set(strata) == EXPECTED_STRATA and all(count == GROUPS_PER_STRATUM for count in strata.values()),
            "online batch is not balanced across source/harness/arm strata")
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
    return {"admitted": True, "rollouts": len(rows), "groups": len(groups),
            "variable_reward_groups": variable_groups,
            "rewards_by_cell": rewards_by_cell,
            "engineering_only": False, "policy_snapshot": policy_snapshot}
