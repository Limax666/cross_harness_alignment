"""Apply the existing rollout admission gate to scored expanded GRPO segments."""
from __future__ import annotations

from rl_harness_segments import expand_rollouts
from rl_multiharness_batch_gate import require_online_batch


def admit_segmented_rollouts(rollouts, evidence, segments, old_log_probs, *, policy_snapshot, cells, assignments):
    """Reconstruct generated-token evidence once per original rollout, then admit.

    This GRPO-only helper requires the exact, ordered result of expand_rollouts.
    Input evidence must originate from the trusted environment, not model text.
    """
    rollouts, evidence, segments = list(rollouts), list(evidence), list(segments)
    expected = expand_rollouts(rollouts)
    if segments != expected:
        raise ValueError('segments differ from original recorded rollouts or advantages')
    if len(old_log_probs) != len(segments) or len(evidence) != len(rollouts):
        raise ValueError('rollout or segment count mismatch')
    by_id = {row.get('rollout_id'): dict(row) for row in evidence}
    if len(by_id) != len(evidence) or set(by_id) != {r.rollout_id for r in rollouts}:
        raise ValueError('missing or duplicate rollout evidence')
    group_to_cell, cell_to_group = {}, {}
    for r in rollouts:
        row = by_id[r.rollout_id]
        if row.get('reward') != r.reward:
            raise ValueError('recorded and verified rollout rewards differ')
        cell = row.get('cell_id')
        if (r.group_id in group_to_cell and group_to_cell[r.group_id] != cell or
                cell in cell_to_group and cell_to_group[cell] != r.group_id):
            raise ValueError('rollout group and task cell mismatch')
        group_to_cell[r.group_id], cell_to_group[cell] = cell, r.group_id
        row.update(response_ids=[], response_mask=[], token_origins=[], rollout_log_probs=[], old_log_probs=[])
    width = max(len(s.tokens) for s in segments)
    for segment, old in zip(segments, old_log_probs):
        old = old.tolist() if hasattr(old, 'tolist') else list(old)
        if len(old) != width:
            raise ValueError('old-policy log probability width mismatch')
        row = by_id[segment.rollout_id]
        row['response_ids'].extend(segment.tokens)
        row['response_mask'].extend([1] * len(segment.tokens))
        row['token_origins'].extend(['assistant'] * len(segment.tokens))
        row['rollout_log_probs'].extend(segment.logprobs)
        row['old_log_probs'].extend(old[:len(segment.tokens)])
    return require_online_batch([by_id[r.rollout_id] for r in rollouts], policy_snapshot=policy_snapshot,
                                cells=cells, assignments=assignments, allow_zero_reward_variance=True)
