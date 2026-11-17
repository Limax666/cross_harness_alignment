"""Eight-row policy; frozen v1 capture, initialization and training inherited.

Only the hard batch cap and explicit runtime identity change. generate_batch is
kept identical to the frozen v1 method except its request-count validation.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
import time

import hsrl_batched_policy_v1 as previous
from hsrl_batched_policy_v1 import RawLogprobCapture, _file_identity, _main_thread
from hsrl_v3.policy import LOGPROB_CONTRACT

VERSION = 'hsrl-fast-batched-policy-v2'


class BatchedPolicy(previous.BatchedPolicy):
    VERSION = VERSION

    def fast_runtime_identity(self):
        identity = super().fast_runtime_identity()
        identity.update(schema='hsrl-fast-runtime/v2', policy=self.VERSION,
            inherited_policy=previous.VERSION, max_batch_size=8,
            policy_class=__name__ + '.BatchedPolicy',
            policy_source=_file_identity(__file__))
        return identity

    def generate_batch(self, requests):
        _main_thread()
        if not isinstance(requests, list) or not 1 <= len(requests) <= 8:
            raise ValueError('batched_policy_requires_one_to_eight_requests')
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

