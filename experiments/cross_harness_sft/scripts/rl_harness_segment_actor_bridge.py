"""Prepare segmented actor inputs using existing VeRL scorer methods.

This opt-in helper does not invoke an optimizer or replace trainer admission.
"""
from __future__ import annotations


def prepare_actor_batch(batch, compute_old, compute_ref, *, loss_agg_mode, optimizer_segment_capacity):
    """Score all segment rows on their own prefixes, without recomputing advantage.

    compute_old and compute_ref accept a VeRL DataProto and have the same
    return contracts as RayPPOTrainer._compute_old_log_prob and
    RayPPOTrainer._compute_ref_log_prob. The complete expanded set must fit
    one actor minibatch so splitting segments cannot change token weights.
    """
    if batch.meta_info.get('segment_advantages_precomputed') is not True:
        raise ValueError('expected rollout-level precomputed advantages')
    if loss_agg_mode != 'token-mean':
        raise ValueError('segments require token-mean reduction')
    shape = tuple(batch.batch['responses'].shape)
    if len(shape) != 2 or not all(shape):
        raise ValueError('invalid response shape')
    if type(optimizer_segment_capacity) is not int or optimizer_segment_capacity < shape[0]:
        raise ValueError('expanded segments exceed optimizer minibatch capacity')
    for name in ('response_mask', 'advantages'):
        if tuple(batch.batch[name].shape) != shape:
            raise ValueError(f'invalid {name} shape')
    if 'old_log_probs' in batch.batch or 'ref_log_prob' in batch.batch:
        raise ValueError('refusing to replace preexisting log probabilities')
    old, _mfu = compute_old(batch)
    ref = compute_ref(batch)
    for data, key in ((old, 'old_log_probs'), (ref, 'ref_log_prob')):
        if key not in data.batch or tuple(data.batch[key].shape) != shape:
            raise ValueError(f'invalid {key} shape')
        if not bool(data.batch[key].isfinite().all()):
            raise ValueError(f'nonfinite {key}')
    if 'entropys' in old.batch:
        old.batch.pop('entropys')
    batch = batch.union(old)
    batch = batch.union(ref)
    batch.meta_info['requires_old_logprob_recompute'] = False
    return batch
