"""Actual tiny CPU HF batches, never a checkpoint/source/GPU/RL run."""
import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def implementation():
    assert importlib.util.find_spec('hsrl_batched_policy_v2') is not None, 'scale8 policy missing'
    import hsrl_batched_policy_v2
    return hsrl_batched_policy_v2


@pytest.fixture
def policy():
    module = implementation()
    from transformers import GPT2Config, GPT2LMHeadModel
    from test_hsrl_batched_policy_v1 import Tokenizer
    torch.manual_seed(17)
    core = GPT2LMHeadModel(GPT2Config(vocab_size=11, n_positions=32, n_embd=8,
        n_layer=1, n_head=1, bos_token_id=1, eos_token_id=10, pad_token_id=0,
        attn_pdrop=0., embd_pdrop=0., resid_pdrop=0.))

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core, self.calls = core, []

        def get_base_model(self):
            return self.core

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return self.core.generate(**kwargs)

    class CPUScalePolicy(module.BatchedPolicy):
        def score(self, prompt_ids, response_ids, *, reference=False):
            assert reference, 'generation old logprobs must never be recomputed'
            with torch.no_grad():
                ids = torch.tensor([prompt_ids + response_ids])
                logits = core(input_ids=ids).logits[0, len(prompt_ids)-1:-1].float()
                return logits.log_softmax(-1).gather(-1, torch.tensor(response_ids)[:, None]).flatten().tolist()

    result = CPUScalePolicy.__new__(CPUScalePolicy)
    result.torch, result.model, result.tokenizer = torch, Wrapped(), Tokenizer()
    result.eos_ids, result.version, result.batch_events = [10], 2, []
    return result


@pytest.mark.parametrize('size', [4, 8])
def test_actual_hf_four_and_eight_rows_preserve_raw_logps_and_exact_left_padding(policy, size):
    prompts = [[1] * (i + 1) + [2 + i % 6] for i in range(size)]
    output = policy.generate_batch([{'prompt_ids': prompt, 'max_new_tokens': 6} for prompt in prompts])
    assert len(output) == size
    assert len(policy.model.calls) == 1
    call = policy.model.calls[0]
    assert call['input_ids'].shape == (size, size + 1)
    for index, prompt in enumerate(prompts):
        assert call['input_ids'][index].tolist() == [0] * (size + 1 - len(prompt)) + prompt
        assert call['attention_mask'][index].tolist() == [0] * (size + 1 - len(prompt)) + [1] * len(prompt)
    assert (call['temperature'], call['top_p'], call['top_k'], call['repetition_penalty']) == (.7, .95, 0, 1.)
    assert call['disable_compile'] is True and call['logits_to_keep'] == 1
    assert call['output_scores'] is call['output_logits'] is False
    for row in output:
        assert row['old_logprobs'] == pytest.approx(row['reference_logprobs'], abs=1e-6)
        assert len(row['response_ids']) == len(row['old_logprobs'])
        assert all(math.isfinite(value) for value in row['old_logprobs'])
        assert row['policy_version'] == 2
        if row['stop_reason'] == 'eos':
            assert row['response_ids'][-1] == 10
            assert row['text'] == ','.join(str(token) for token in row['response_ids'][:-1])
        else:
            assert len(row['response_ids']) == 6 and row['length_limited']
    assert any(row['stop_reason'] == 'eos' for row in output)
    assert any(len(row['response_ids']) < 6 for row in output)
    assert len({len(row['response_ids']) for row in output}) > 1
    event = policy.batch_events[-1]
    assert event['batch_size'] == size and event['status'] == 'complete'
    assert event['peak_pending_logits_elements'] == size * 11
    assert event['sampled_tokens'] == [len(row['response_ids']) for row in output]


def test_ninth_row_rejected_without_generation(policy):
    with pytest.raises(ValueError):
        policy.generate_batch([{'prompt_ids': [1], 'max_new_tokens': 1}] * 9)
    assert not policy.model.calls


def test_v2_identity_names_new_policy_and_preserves_v1_provenance(monkeypatch):
    module = implementation()
    import hsrl_batched_policy_v1 as v1
    baseline = {'schema': 'hsrl-fast-runtime/v1', 'policy': v1.VERSION,
                'packages': {'torch': {'version': 'pinned'}}, 'generation_compile_disabled': True}
    monkeypatch.setattr(v1.BatchedPolicy, 'fast_runtime_identity', lambda self: dict(baseline))
    policy = module.BatchedPolicy.__new__(module.BatchedPolicy)
    identity = policy.fast_runtime_identity()
    assert identity['policy'] == module.VERSION and identity['policy'] != v1.VERSION
    assert identity['max_batch_size'] == 8
    assert identity['inherited_policy'] == v1.VERSION
    assert identity['policy_class'] == 'hsrl_batched_policy_v2.BatchedPolicy'
    assert identity['policy_source']['sha256']
    assert identity['packages'] == baseline['packages']
    assert baseline['policy'] == v1.VERSION
