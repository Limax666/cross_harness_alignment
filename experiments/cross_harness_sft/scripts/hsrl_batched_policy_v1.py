"""Two-row genuine HF generation with pre-warp raw logprob capture.

Only generation differs from the frozen physical-GPU1 policy. Reference scoring,
LoRA, the objective, optimizer/checkpoint interfaces and RNG methods are inherited.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import importlib.metadata
import inspect
import math
from pathlib import Path
import threading
import time

from hsrl_selected_gpu_policy_v1 import SelectedGPU1PolicyV1, verify_initializer_contract
from hsrl_v3.policy import LOGPROB_CONTRACT

VERSION = 'hsrl-fast-batched-policy-v1'
FAST_BINDINGS = {
    'causal_conv1d_fn': 'causal_conv1d_fn',
    'causal_conv1d_update': 'causal_conv1d_update',
    'chunk_gated_delta_rule': 'chunk_gated_delta_rule',
    'recurrent_gated_delta_rule': 'fused_recurrent_gated_delta_rule',
}


def _main_thread():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('batched_policy_requires_main_thread')


def require_fast_kernels(module, model):
    """Reject both missing package globals and layers already bound to fallback."""
    if not getattr(module, 'is_fast_path_available', False):
        raise RuntimeError('Qwen35_fast_path_unavailable')
    for name in FAST_BINDINGS.values():
        fn = getattr(module, name, None)
        expected_package = 'causal_conv1d' if name.startswith('causal_conv') else 'fla.'
        if not callable(fn) or not getattr(fn, '__module__', '').startswith(expected_package):
            raise RuntimeError('Qwen35_fast_binding_missing:' + name)
    layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, getattr(module, 'Qwen3_5GatedDeltaNet', ())):
            if any(getattr(layer, attr, None) is not getattr(module, target)
                   for attr, target in FAST_BINDINGS.items()):
                raise RuntimeError('Qwen35_fast_layer_fallback:' + name)
            layers.append(name)
    if not layers:
        raise RuntimeError('Qwen35_fast_layers_missing')
    return layers


def _file_identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _package_identity(name, *, all_code=False):
    distribution = importlib.metadata.distribution(name)
    result = {'version': distribution.version,
              'root': str(Path(distribution.locate_file('')).resolve())}
    files = sorted(str(p) for p in distribution.files or []
                   if str(p).endswith(('.py', '.so')))
    if all_code:
        result['code_files'] = {name: _file_identity(distribution.locate_file(name))
                                for name in files}
    metadata = next((p for p in distribution.files or [] if str(p).endswith('.dist-info/METADATA')), None)
    if metadata is not None:
        result['metadata'] = _file_identity(distribution.locate_file(metadata))
    return result


class RawLogprobCapture:
    """One pending [batch,vocabulary] copy, consumed at the next actual forward.

    The base-model hook runs before GenerationMixin applies its processors. A
    private copy is necessary: upstream sampling may modify its output in place.
    Only scalar token/logprob histories survive; no [batch,time,vocabulary] cache.
    """
    def __init__(self, model, padded_prompt, eos_ids, pad_id):
        self.model, self.prompt = model, padded_prompt
        self.eos_ids, self.pad_id = set(eos_ids), pad_id
        self.pending = None
        self.tokens = [[] for _ in padded_prompt]
        self.logprobs = [[] for _ in padded_prompt]
        self.finished = [False for _ in padded_prompt]
        self.forward_calls = 0
        self.peak_pending_logits_elements = 0
        self.handles = []

    def __enter__(self):
        self.handles.append(self.model.register_forward_pre_hook(self.before_forward, with_kwargs=True))
        self.handles.append(self.model.register_forward_hook(self.after_forward, with_kwargs=True))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending = None

    def _consume(self, sampled):
        if self.pending is None or sampled.shape != (len(self.tokens),):
            raise RuntimeError('raw_capture_pending_token_shape_mismatch')
        if ((sampled < 0) | (sampled >= self.pending.shape[-1])).any():
            raise RuntimeError('raw_capture_token_out_of_vocabulary')
        logp = self.pending.gather(-1, sampled[:, None]).squeeze(-1) - self.pending.logsumexp(-1)
        if not logp.isfinite().all():
            raise RuntimeError('nonfinite_action_logprob')
        for row, (token, value) in enumerate(zip(sampled.tolist(), logp.tolist())):
            if self.finished[row]:
                if token != self.pad_id:
                    raise RuntimeError('raw_capture_nonpadding_after_EOS')
            else:
                self.tokens[row].append(token)
                self.logprobs[row].append(value)
                self.finished[row] = token in self.eos_ids
        self.pending = None

    def before_forward(self, model, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        if ids is None or ids.ndim != 2 or ids.shape[0] != len(self.tokens):
            raise RuntimeError('raw_capture_forward_input_mismatch')
        if self.forward_calls == 0:
            if not ids.equal(self.prompt):
                raise RuntimeError('raw_capture_initial_prompt_mismatch')
        else:
            if ids.shape[1] != 1:
                raise RuntimeError('raw_capture_requires_single_cached_decode_token')
            self._consume(ids[:, -1])

    def after_forward(self, model, args, kwargs, output):
        import torch
        logits = getattr(output, 'logits', None)
        if (self.pending is not None or logits is None or logits.ndim != 3
                or tuple(logits.shape[:2]) != (len(self.tokens), 1)):
            raise RuntimeError('raw_capture_requires_one_next_token_logits_slice')
        if not logits.isfinite().all():
            raise RuntimeError('nonfinite_raw_model_logits')
        self.pending = logits[:, 0, :].detach().to(dtype=torch.float32, copy=True)
        self.forward_calls += 1
        self.peak_pending_logits_elements = max(self.peak_pending_logits_elements, self.pending.numel())

    def finish(self, sequences):
        width = self.prompt.shape[1]
        if (sequences.ndim != 2 or sequences.shape[0] != len(self.tokens)
                or not sequences[:, :width].equal(self.prompt)
                or sequences.shape[1] - width != self.forward_calls or self.forward_calls == 0):
            raise RuntimeError('raw_capture_generated_sequence_shape_mismatch')
        self._consume(sequences[:, -1])
        tail = sequences[:, width:].tolist()
        for captured, actual in zip(self.tokens, tail):
            expected = actual
            for offset, token in enumerate(actual):
                if token in self.eos_ids:
                    expected = actual[:offset+1]
                    if any(t != self.pad_id for t in actual[offset+1:]):
                        raise RuntimeError('raw_capture_nonpadding_after_EOS')
                    break
            if captured != expected:
                raise RuntimeError('raw_capture_sampled_token_sequence_mismatch')
        return self.tokens, self.logprobs


class BatchedPolicy(SelectedGPU1PolicyV1):
    def __init__(self, model_path: Path, seed=267):
        _main_thread()
        verify_initializer_contract()
        super().__init__(model_path, seed=seed)
        from transformers.models.qwen3_5 import modeling_qwen3_5
        self._fast_layers = require_fast_kernels(modeling_qwen3_5, self.model)
        self.batch_events = []
        self._fast_identity = None

    def fast_runtime_identity(self):
        _main_thread()
        from transformers.models.qwen3_5 import modeling_qwen3_5 as module
        layers = require_fast_kernels(module, self.model)
        if self._fast_identity is None:
            import peft
            import transformers
            import triton
            bindings = {}
            for name in FAST_BINDINGS.values():
                function = getattr(module, name)
                bindings[name] = {'module': function.__module__, 'qualname': function.__qualname__,
                                  'source': _file_identity(inspect.getfile(function))}
            self._fast_identity = {
                'schema': 'hsrl-fast-runtime/v1', 'policy': VERSION,
                'generation_compile_disabled': True,
                'logprob_contract': LOGPROB_CONTRACT, 'linear_attention_layers': layers,
                'modeling': _file_identity(module.__file__), 'functions': bindings,
                'generation_source': _file_identity(inspect.getfile(self.model.get_base_model().generate)),
                'peft_model_source': _file_identity(inspect.getfile(type(self.model))),
                'packages': {name: _package_identity(name, all_code=name in {'causal-conv1d', 'fla-core'})
                             for name in ('torch', 'transformers', 'peft', 'triton', 'causal-conv1d', 'fla-core')},
                'runtime_sources': {name: _file_identity(value.__file__) for name, value in
                                    [('torch', self.torch), ('transformers', transformers),
                                     ('peft', peft), ('triton', triton)]},
                'cuda': self.torch.version.cuda, 'device': str(next(self.model.parameters()).device),
            }
        return copy.deepcopy(self._fast_identity)

    def generate(self, prompt_ids, max_new_tokens):
        return self.generate_batch([{'prompt_ids': prompt_ids, 'max_new_tokens': max_new_tokens}])[0]

    def generate_batch(self, requests):
        _main_thread()
        if not isinstance(requests, list) or not 1 <= len(requests) <= 2:
            raise ValueError('batched_policy_requires_one_or_two_requests')
        for request in requests:
            if (not isinstance(request, dict) or not isinstance(request.get('prompt_ids'), list)
                    or not request['prompt_ids']
                    or any(type(token) is not int or token < 0 for token in request['prompt_ids'])
                    or type(request.get('max_new_tokens')) is not int or request['max_new_tokens'] <= 0):
                raise ValueError('exact_nonempty_prompt_and_positive_integer_budget_required')
        budget = requests[0]['max_new_tokens']
        if any(request['max_new_tokens'] != budget for request in requests):
            raise ValueError('batched_policy_requires_equal_generation_budgets')
        torch = self.torch
        self.model.eval()
        device = next(self.model.parameters()).device
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_ids[0]
        width = max(len(request['prompt_ids']) for request in requests)
        ids = torch.tensor([[pad] * (width - len(r['prompt_ids'])) + r['prompt_ids'] for r in requests],
                           dtype=torch.long, device=device)
        mask = torch.tensor([[0] * (width - len(r['prompt_ids'])) + [1] * len(r['prompt_ids']) for r in requests],
                            dtype=torch.long, device=device)
        event = {'status': 'running', 'batch_size': len(requests), 'input_shape': [len(requests), width],
            'padded_prompt_tokens': len(requests) * width,
            'prompt_tokens': [len(r['prompt_ids']) for r in requests], 'max_new_tokens': budget,
            'forward_calls': 0, 'sampled_tokens': None, 'generation_seconds': None,
            'reference_seconds': None, 'peak_pending_logits_elements': 0,
            'logprob_contract': LOGPROB_CONTRACT}
        self.batch_events.append(event)
        start = time.perf_counter()
        generated = None
        capture = RawLogprobCapture(self.model.get_base_model(), ids, self.eos_ids, pad)
        try:
            autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
            # PEFT.generate delegates to the base's generate and bypasses its own
            # forward. Hook the actual base, with compiled forward disabled.
            with torch.inference_mode(), autocast, capture:
                sequences = self.model.generate(input_ids=ids, attention_mask=mask,
                    do_sample=True, temperature=0.7, top_p=0.95, top_k=0, repetition_penalty=1.0,
                    max_new_tokens=budget, use_cache=True, logits_to_keep=1,
                    eos_token_id=self.eos_ids, pad_token_id=pad,
                    output_scores=False, output_logits=False, return_dict_in_generate=False,
                    disable_compile=True)
                responses, old = capture.finish(sequences)
            generated = time.perf_counter()
            event['sampled_tokens'] = [len(r) for r in responses]
            del sequences, ids, mask
            rows = []
            for request, response, logp in zip(requests, responses, old):
                if not response or len(response) > budget:
                    raise RuntimeError('generation_violated_exact_token_budget')
                reference = self.score(request['prompt_ids'], response, reference=True)
                if len(reference) != len(response) or not all(math.isfinite(x) for x in reference):
                    raise RuntimeError('nonfinite_or_misaligned_reference_logprob')
                eos = response[-1] in self.eos_ids
                rows.append({'response_ids': response,
                    'text': self.tokenizer.decode(response[:-1] if eos else response, skip_special_tokens=False),
                    'old_logprobs': logp, 'reference_logprobs': reference, 'policy_version': self.version,
                    'length_limited': len(response) >= budget and not eos,
                    'stop_reason': 'eos' if eos else 'generation_limit',
                    'protocol_eos_ids': self.eos_ids,
                    'text_extraction': 'exact_sampled_ids_except_terminal_EOS_skip_special_false_v1',
                    'logprob_contract': LOGPROB_CONTRACT,
                    'sampling': {'temperature': .7, 'top_p': .95, 'top_k': 0}})
            event['status'] = 'complete'
            return rows
        except BaseException as error:
            event.update(status='failed', error_type=type(error).__name__)
            raise
        finally:
            now = time.perf_counter()
            event.update(forward_calls=capture.forward_calls,
                peak_pending_logits_elements=capture.peak_pending_logits_elements,
                generation_seconds=(generated if generated is not None else now) - start,
                reference_seconds=now - generated if generated is not None else 0.)
