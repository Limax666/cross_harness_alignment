"""CPU Torch correctness checks, never a source/model-checkpoint RL experiment."""
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def module():
    assert importlib.util.find_spec('hsrl_batched_policy_v1') is not None, 'batched policy missing'
    import hsrl_batched_policy_v1
    return hsrl_batched_policy_v1


class Transition(torch.nn.Module):
    def forward(self, input_ids):
        # A genuine tiny Torch forward; raw scores vary with the consumed token.
        logits = torch.arange(11, dtype=torch.float32)[None, None, :]
        return SimpleNamespace(logits=logits.expand(input_ids.shape[0], 1, -1).clone())


def test_capture_scores_raw_logits_and_masks_each_eos_without_retaining_history():
    implementation = module()
    model = Transition()
    prompt = torch.tensor([[0, 2], [3, 4]])
    with implementation.RawLogprobCapture(model, prompt, [9, 10], 0) as capture:
        output = model(input_ids=prompt)
        # Generation may mutate its logits in place after our forward hook.
        output.logits.fill_(123.)
        model(input_ids=torch.tensor([[9], [5]]))
        assert tuple(capture.pending.shape) == (2, 11)
        assert {name for name, value in vars(capture).items() if isinstance(value, torch.Tensor)} == {'prompt', 'pending'}
        assert all(isinstance(value, float) for row in capture.logprobs for value in row)
        model(input_ids=torch.tensor([[0], [6]]))
        responses, logp = capture.finish(torch.tensor([[0, 2, 9, 0, 0], [3, 4, 5, 6, 10]]))
        assert capture.pending is None
    assert responses == [[9], [5, 6, 10]]
    normalizer = math.log(sum(math.exp(i) for i in range(11)))
    assert logp[0] == pytest.approx([9 - normalizer])
    assert logp[1] == pytest.approx([i - normalizer for i in [5, 6, 10]])
    assert len(model._forward_hooks) == len(model._forward_pre_hooks) == 0


@pytest.mark.parametrize('problem', ['sequence', 'padding', 'prefix', 'extra_step', 'full_logits', 'nan'])
def test_capture_rejects_misalignment_nonfinite_or_full_sequence_logits(problem):
    implementation = module()
    model = Transition()
    prompt = torch.tensor([[1, 2], [3, 4]])
    with pytest.raises(RuntimeError):
        with implementation.RawLogprobCapture(model, prompt, [9], 0) as capture:
            if problem in {'full_logits', 'nan'}:
                bad = torch.zeros(2, 2 if problem == 'full_logits' else 1, 11)
                if problem == 'nan':
                    bad[0, 0, 0] = float('nan')
                capture.after_forward(model, (), {}, SimpleNamespace(logits=bad))
            else:
                model(input_ids=prompt)
                model(input_ids=torch.tensor([[9], [5]]))
                tail = [[9, 0], [5, 6]]
                if problem == 'sequence':
                    tail[1][0] = 7
                if problem == 'padding':
                    tail[0][1] = 4
                if problem == 'extra_step':
                    tail = [[9], [5]]
                result = torch.cat([prompt, torch.tensor(tail)], dim=1)
                if problem == 'prefix':
                    result[0, 0] = 7
                capture.finish(result)
    assert len(model._forward_hooks) == len(model._forward_pre_hooks) == 0


class Tokenizer:
    pad_token_id = 0

    def decode(self, ids, **kwargs):
        assert kwargs == {'skip_special_tokens': False}
        return ','.join(str(token) for token in ids)


@pytest.fixture
def policy():
    implementation = module()
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(17)
    core = GPT2LMHeadModel(GPT2Config(vocab_size=11, n_positions=32, n_embd=8,
        n_layer=1, n_head=1, bos_token_id=1, eos_token_id=10, pad_token_id=0,
        attn_pdrop=0., embd_pdrop=0., resid_pdrop=0.))

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = core
            self.calls = []

        def get_base_model(self):
            return self.core

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return self.core.generate(**kwargs)

    class CPUReferencePolicy(implementation.BatchedPolicy):
        def score(self, prompt_ids, response_ids, *, reference=False):
            assert reference, 'old logprobs must come from generation, never rescoring'
            with torch.no_grad():
                ids = torch.tensor([prompt_ids + response_ids])
                logits = core(input_ids=ids).logits[0, len(prompt_ids)-1:-1].float()
                targets = torch.tensor(response_ids)
                return logits.log_softmax(-1).gather(-1, targets[:, None]).flatten().tolist()

    result = CPUReferencePolicy.__new__(CPUReferencePolicy)
    result.torch, result.model = torch, Wrapped()
    result.tokenizer, result.eos_ids, result.version = Tokenizer(), [10], 3
    result.batch_events = []
    return result


def test_real_hf_batch_generation_left_padding_matches_independent_teacher_forcing(policy):
    output = policy.generate_batch([{'prompt_ids': [1, 2], 'max_new_tokens': 4},
                                    {'prompt_ids': [3, 4, 5, 6], 'max_new_tokens': 4}])
    assert len(policy.model.calls) == 1
    call = policy.model.calls[0]
    assert call['input_ids'].tolist() == [[0, 0, 1, 2], [3, 4, 5, 6]]
    assert call['attention_mask'].tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]
    assert call['logits_to_keep'] == 1
    assert call['output_scores'] is call['output_logits'] is False
    assert (call['temperature'], call['top_p'], call['top_k']) == (.7, .95, 0)
    for row in output:
        assert row['old_logprobs'] == pytest.approx(row['reference_logprobs'], abs=1e-6)
        assert len(row['response_ids']) == len(row['old_logprobs'])
        assert row['policy_version'] == 3
        assert all(math.isfinite(x) for x in row['old_logprobs'])
    event = policy.batch_events[-1]
    assert event['batch_size'] == 2 and event['forward_calls'] >= 1
    assert event['sampled_tokens'] == [len(row['response_ids']) for row in output]
    assert event['generation_seconds'] >= 0 and event['reference_seconds'] >= 0


@pytest.mark.parametrize('requests', [[], [{ 'prompt_ids': [1], 'max_new_tokens': 0}],
    [{'prompt_ids': [], 'max_new_tokens': 1}],
    [{'prompt_ids': [1], 'max_new_tokens': 1}, {'prompt_ids': [2], 'max_new_tokens': 2}],
    [{'prompt_ids': [1], 'max_new_tokens': 1}] * 3,
    [{'prompt_ids': [1.5], 'max_new_tokens': 1}],
    [{'prompt_ids': [True], 'max_new_tokens': 1}],
    [{'prompt_ids': [1], 'max_new_tokens': True}]])
def test_invalid_batch_never_calls_model_or_truncates(policy, requests):
    with pytest.raises(ValueError):
        policy.generate_batch(requests)
    assert not policy.model.calls


def test_single_request_preserves_serial_interface(policy):
    row = policy.generate([1, 2], 2)
    assert row['sampling'] == {'temperature': .7, 'top_p': .95, 'top_k': 0}
    assert row['logprob_contract'] == 'raw_model_log_softmax_before_temperature_top_p_v1'


def test_worker_thread_rejected_before_generation(policy):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(policy.generate, [1], 1)
        with pytest.raises(RuntimeError, match='main_thread'):
            future.result()
    assert not policy.model.calls


def test_actual_peft_generation_bypasses_outer_forward_but_capture_still_matches(policy):
    from peft import LoraConfig, TaskType, get_peft_model
    core = policy.model.core
    policy.model = get_peft_model(core, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=2, lora_alpha=4, lora_dropout=0., target_modules=['c_attn'], fan_in_fan_out=True))
    outer_calls = []
    handle = policy.model.register_forward_hook(lambda *args: outer_calls.append(1))
    try:
        result = policy.generate_batch([{'prompt_ids': [1, 2], 'max_new_tokens': 3},
                                        {'prompt_ids': [2, 3, 4], 'max_new_tokens': 3}])
    finally:
        handle.remove()
    assert outer_calls == []
    for row in result:
        assert row['old_logprobs'] == pytest.approx(row['reference_logprobs'], abs=1e-6)


def test_nonfinite_reference_aborts_batch(policy):
    policy.score = lambda *args, **kwargs: [float('nan')]
    with pytest.raises(RuntimeError, match='reference'):
        policy.generate([1, 2], 1)
    assert policy.batch_events[-1]['status'] == 'failed'
    assert policy.batch_events[-1]['forward_calls'] == 1
    assert policy.batch_events[-1]['sampled_tokens'] == [1]


def test_tiny_qwen35_peft_left_padding_positions_and_cached_linear_attention(monkeypatch):
    """Random tiny CPU Qwen, torch fallbacks only; never tests GPU fast kernels."""
    implementation = module()
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    # Optional FLA normalization and kernels require CUDA. Only this CPU unit's
    # process-local model uses torch fallbacks; production __init__ rejects them.
    for name in ['FusedRMSNormGated', 'causal_conv1d_fn', 'causal_conv1d_update',
                 'chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule']:
        monkeypatch.setattr(qwen, name, None)
    torch.manual_seed(23)
    config = Qwen3_5Config(text_config=dict(vocab_size=19, hidden_size=16,
        intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_conv_kernel_dim=2,
        layer_types=['linear_attention', 'full_attention'],
        max_position_embeddings=32, bos_token_id=1, pad_token_id=0, eos_token_id=18,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                         'partial_rotary_factor': 1., 'mrope_section': [1, 1, 2]}),
        vision_config=dict(depth=1, hidden_size=8, intermediate_size=16, num_heads=1,
                           out_hidden_size=16, num_position_embeddings=16),
        image_token_id=15, video_token_id=16, vision_start_token_id=17, vision_end_token_id=14)
    core = Qwen3_5ForConditionalGeneration(config)
    wrapped = get_peft_model(core, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=2, lora_alpha=4, lora_dropout=0., target_modules=['q_proj', 'in_proj_qkv']))

    class CPUQwenPolicy(implementation.BatchedPolicy):
        def score(self, prompt_ids, response_ids, *, reference=False):
            assert reference
            ids = torch.tensor([prompt_ids + response_ids])
            with wrapped.disable_adapter(), torch.no_grad():
                hidden = core.model.language_model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    position_ids=torch.arange(ids.shape[1])[None, :], use_cache=False,
                    past_key_values=None, return_dict=True).last_hidden_state
                logits = core.lm_head(hidden[0, len(prompt_ids)-1:-1]).float()
                return logits.log_softmax(-1).gather(-1, torch.tensor(response_ids)[:, None]).flatten().tolist()

    policy = CPUQwenPolicy.__new__(CPUQwenPolicy)
    policy.torch, policy.model, policy.tokenizer = torch, wrapped, Tokenizer()
    policy.eos_ids, policy.version, policy.batch_events = [18], 0, []
    positions = []
    def observe(model, args, kwargs):
        positions.append(kwargs['position_ids'].clone())
    handle = core.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        result = policy.generate_batch([{'prompt_ids': [1, 2], 'max_new_tokens': 3},
                                        {'prompt_ids': [1, 3, 4, 5], 'max_new_tokens': 3}])
    finally:
        handle.remove()
    assert positions[0].shape == (4, 2, 4)
    assert positions[0][0, 0, -2:].tolist() == [0, 1]
    assert positions[0][0, 1].tolist() == [0, 1, 2, 3]
    for row in result:
        assert row['old_logprobs'] == pytest.approx(row['reference_logprobs'], abs=1e-5)
    assert policy.batch_events[-1]['batch_size'] == 2
    assert policy.batch_events[-1]['forward_calls'] >= 2


def test_fast_kernel_requirement_fails_closed_when_bindings_missing():
    implementation = module()
    with pytest.raises(RuntimeError, match='fast'):
        implementation.require_fast_kernels(SimpleNamespace(is_fast_path_available=False), Transition())
