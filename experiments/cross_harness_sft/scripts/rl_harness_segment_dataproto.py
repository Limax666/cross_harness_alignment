"""Opt-in segment-to-VeRL adapter. It is not connected to the live trainer."""
from __future__ import annotations

import math


def pack_segment_rows(segments, *, pad_token_id, max_prompt_length=None, max_response_length=None):
    """Build padded VeRL-shaped rows without ever truncating generation prefixes."""
    segments = list(segments)
    if not segments or type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError('segments and a valid padding token are required')
    rollout_metadata = {}
    for s in segments:
        metadata = (s.group_id, s.reward, s.advantage)
        if s.rollout_id in rollout_metadata and rollout_metadata[s.rollout_id] != metadata:
            raise ValueError("inconsistent original rollout metadata")
        rollout_metadata[s.rollout_id] = metadata
        if not s.prefix or not s.tokens or len(s.tokens) != len(s.logprobs):
            raise ValueError('invalid segment')
        if (any(type(x) is not int or x < 0 for x in s.prefix + s.tokens)
                or not all(math.isfinite(x) for x in s.logprobs)
                or not math.isfinite(s.advantage)):
            raise ValueError('invalid token or nonfinite segment data')
    prompt_width = max(len(s.prefix) for s in segments)
    response_width = max(len(s.tokens) for s in segments)
    if max_prompt_length is not None and prompt_width > max_prompt_length:
        raise ValueError('prefix exceeds prompt limit; refusing to truncate')
    if max_response_length is not None and response_width > max_response_length:
        raise ValueError('response exceeds limit; refusing to truncate')
    keys = ('prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids',
            'response_mask', 'rollout_log_probs', 'advantages', 'rollout_id', 'group_id')
    rows = {key: [] for key in keys}
    for s in segments:
        left, right = prompt_width - len(s.prefix), response_width - len(s.tokens)
        prompts = [pad_token_id] * left + list(s.prefix)
        responses = list(s.tokens) + [pad_token_id] * right
        mask = [0] * left + [1] * (len(s.prefix) + len(s.tokens)) + [0] * right
        position, count = [], 0
        for bit in mask:
            position.append(count if bit else 0)
            count += bit
        rows['prompts'].append(prompts)
        rows['responses'].append(responses)
        rows['input_ids'].append(prompts + responses)
        rows['attention_mask'].append(mask)
        rows['position_ids'].append(position)
        rows['response_mask'].append([1] * len(s.tokens) + [0] * right)
        rows['rollout_log_probs'].append(list(s.logprobs) + [0.] * right)
        rows['advantages'].append([s.advantage] * len(s.tokens) + [0.] * right)
        rows['rollout_id'].append(s.rollout_id)
        rows['group_id'].append(s.group_id)
    return rows


def make_dataproto(segments, *, pad_token_id, max_prompt_length=None, max_response_length=None):
    """Materialize a DataProto after rollout-level advantages have been computed.

    Actor old/reference log probabilities must still be recomputed from these
    exact padded input_ids before the result can be submitted to the trainer.
    """
    import torch
    from verl import DataProto

    rows = pack_segment_rows(segments, pad_token_id=pad_token_id,
                             max_prompt_length=max_prompt_length,
                             max_response_length=max_response_length)
    names = ('prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids', 'response_mask')
    tensors = {name: torch.tensor(rows[name], dtype=torch.long) for name in names}
    tensors.update({name: torch.tensor(rows[name], dtype=torch.float32)
                    for name in ('rollout_log_probs', 'advantages')})
    return DataProto.from_dict(tensors=tensors,
                               non_tensors={k: rows[k] for k in ('rollout_id', 'group_id')},
                               meta_info={'segment_advantages_precomputed': True,
                                          'requires_old_logprob_recompute': True})
