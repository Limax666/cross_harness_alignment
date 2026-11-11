"""Explicit GRPO-only preparation path for segmented VeRL batches.

This module does not install itself in the live trainer or invoke an optimizer.
"""
from __future__ import annotations


def prepare_segmented_step(trainer, rollouts, evidence, *, cells, assignments,
                           policy_snapshot, algorithm, pad_token_id,
                           optimizer_segment_capacity, max_prompt_length,
                           max_response_length):
    """Expand original rollout groups, score real prefixes, and run admission.

    Only GRPO without reward KL or rollout correction is currently supported.
    The caller must enforce that configuration and independently verify policy
    identity, trusted evidence and distributed worker behavior. This function
    produces a scored/admitted batch, never performs an actor update.
    """
    if algorithm != 'GRPO':
        raise ValueError('only GRPO is supported on the experimental segment path')
    if not assignments or not policy_snapshot:
        raise ValueError('frozen assignment and policy snapshot required')
    from rl_harness_segments import expand_rollouts
    from rl_harness_segment_dataproto import make_dataproto
    from rl_harness_segment_actor_bridge import prepare_actor_batch
    from rl_harness_segment_admission import admit_segmented_rollouts

    rollouts, evidence = list(rollouts), list(evidence)
    segments = expand_rollouts(rollouts)
    batch = make_dataproto(segments, pad_token_id=pad_token_id,
                           max_prompt_length=max_prompt_length,
                           max_response_length=max_response_length)
    batch = prepare_actor_batch(batch, trainer._compute_old_log_prob,
                                trainer._compute_ref_log_prob,
                                loss_agg_mode='token-mean',
                                optimizer_segment_capacity=optimizer_segment_capacity)
    report = admit_segmented_rollouts(rollouts, evidence, segments,
                                      batch.batch['old_log_probs'],
                                      policy_snapshot=policy_snapshot,
                                      cells=cells, assignments=assignments)
    return batch, report
