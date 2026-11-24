"""Full-parameter policy operations for the text-only Harness Scaling task pool."""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.optim._functional import adamw
from torch.utils.checkpoint import checkpoint

from hsrl_batched_policy_v1 import RawLogprobCapture
from hsrl_v3.policy import LOGPROB_CONTRACT

OFFICIAL_IDENTITY_PATH = Path(__file__).resolve().parents[1] / 'configs/hsrl_official4b_851bf6e.json'
OFFICIAL_IDENTITY_SHA256 = '56ee0dde7ea6e524d0a8ebc81dfc33e94a091976bba73df3325e2cf957e6c5a8'
GRAD_CLIP = 1.0
PPO_CLIP = 0.2
KL_COEFFICIENT = 0.001


def verify_model(path):
    authority_bytes = OFFICIAL_IDENTITY_PATH.read_bytes()
    if hashlib.sha256(authority_bytes).hexdigest() != OFFICIAL_IDENTITY_SHA256:
        raise ValueError('unexpected official model authority')
    pins = json.loads(authority_bytes)['artifacts']
    root = Path(path)
    suffixes = {'.json', '.safetensors', '.jinja', '.txt', '.model', '.bin', '.pt', '.pth', '.py'}
    present = {p.name for p in root.iterdir() if p.suffix in suffixes}
    if root.is_symlink() or present != set(pins):
        raise ValueError('model file set differs from the official release')
    verified = {}
    for name, pin in pins.items():
        artifact = root / name
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError('invalid official model artifact: ' + name)
        with artifact.open('rb') as stream:
            verified[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
            if pin['algorithm'] == 'sha256':
                measured = verified[name]
            elif pin['algorithm'] == 'git_blob_sha1':
                stream.seek(0)
                header = f'blob {artifact.stat().st_size}'.encode() + bytes([0])
                measured = hashlib.file_digest(stream, lambda: hashlib.sha1(header)).hexdigest()
            else:
                raise ValueError('unsupported model identity algorithm')
        if measured != pin['digest']:
            raise ValueError('model artifact checksum mismatch: ' + name)
    return verified


class HostAdamW:
    """Accumulate action gradients in FP32. Stage one gradient at a time for native CPU AdamW."""
    def __init__(self, named_parameters):
        self.params = dict(named_parameters)
        self.state = {}
        self.gradients = {}
        for name, param in self.params.items():
            if not param.requires_grad:
                raise ValueError(f'frozen_actor_parameter: {name}')
            param.register_post_accumulate_grad_hook(self._hook(name))

    def _hook(self, name):
        def accumulate(parameter):
            grad = parameter.grad.detach()
            if name in self.gradients:
                self.gradients[name].add_(grad)
            else:
                self.gradients[name] = grad.to(torch.float32, copy=True)
            parameter.grad = None
        return accumulate

    @torch.no_grad()
    def step(self, lr):
        if not self.gradients or not math.isfinite(lr) or lr <= 0:
            raise ValueError('missing gradients or invalid learning rate')
        norms = [g.norm(2) for g in self.gradients.values()]
        norm = float(torch.stack(norms).norm(2))
        if not math.isfinite(norm):
            raise FloatingPointError('nonfinite global gradient norm')
        clip = min(1., GRAD_CLIP / (norm + 1e-6))
        names = sorted(self.gradients)
        for name in names:
            param = self.params[name]
            grad = self.gradients.pop(name).to('cpu', copy=True)
            grad.mul_(clip)
            if name not in self.state:
                master = param.detach().to('cpu', dtype=torch.float32, copy=True)
                self.state[name] = dict(master=master, m=torch.zeros_like(master),
                                        v=torch.zeros_like(master), step=torch.tensor(0.))
            s = self.state[name]
            adamw([s['master']], [grad], [s['m']], [s['v']], [], [s['step']],
                  fused=True, amsgrad=False, beta1=.9, beta2=.999,
                  lr=lr, weight_decay=0., eps=1e-8, maximize=False)
            param.copy_(s['master'])
            del grad
        return dict(preclip_norm=norm, clip_factor=clip, gradient_tensors=len(names),
                    gradient_parameters=sum(self.params[n].numel() for n in names))


def action_loss(lp, old, reference, mask, advantage, pg_tokens, kl_tokens, pg_enabled):
    ratio = (lp - old).exp()
    delta = reference - lp
    kl = (delta.expm1() - delta).mul(mask).sum() * (KL_COEFFICIENT / kl_tokens)
    if not pg_enabled:
        return kl
    clipped = ratio.clamp(1 - PPO_CLIP, 1 + PPO_CLIP)
    pg = torch.minimum(ratio * advantage, clipped * advantage)
    return kl - pg.mul(mask).sum() / pg_tokens


def load_model(path, *, policy):
    from transformers import Qwen3_5ForConditionalGeneration
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16, device_map={'': 0},
        attn_implementation='sdpa', output_loading_info=True)
    # The pinned Transformers class does not instantiate auxiliary MTP weights.
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError(f'incomplete official model load: {info}')
    if any(not name.startswith('mtp.') for name in info.get('unexpected_keys', [])):
        raise ValueError(f'unexpected non-MTP weights: {info}')
    model.requires_grad_(policy)
    if policy:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model.eval()


def action_hidden(model, prompt_ids, response_ids):
    if not prompt_ids or not response_ids:
        raise ValueError('empty action segment')
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids + response_ids], device=device)
    output = model.model.language_model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       position_ids=torch.arange(ids.size(1), device=device)[None, :],
                                       use_cache=False, return_dict=True)
    start = len(prompt_ids) - 1
    return output.last_hidden_state[0, start:start + len(response_ids)]


def logprobs(head, hidden, targets):
    logits = head(hidden).float()
    return logits.gather(-1, targets[:, None]).squeeze(-1) - logits.logsumexp(-1)


@torch.no_grad()
def score(model, prompt_ids, response_ids):
    h = action_hidden(model.eval(), prompt_ids, response_ids)
    targets = torch.tensor(response_ids, device=h.device)
    lp = torch.cat([logprobs(model.lm_head, x, y) for x, y in
                    zip(h.split(64), targets.split(64))])
    if not lp.isfinite().all():
        raise FloatingPointError('nonfinite reference log probabilities')
    return lp.tolist()


def sampled_actions(model, tokenizer, requests):
    budgets = {r['max_new_tokens'] for r in requests}
    if len(budgets) != 1 or not 1 <= len(requests) <= 8:
        raise ValueError('expected up to eight requests with a shared token budget')
    budget = budgets.pop()
    prompts = [r['prompt_ids'] for r in requests]
    width = max(map(len, prompts))
    if min(map(len, prompts)) == 0 or budget <= 0 or width + budget > 16384:
        raise ValueError('invalid context or action length')
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0]
    device = next(model.parameters()).device
    ids = torch.tensor([[pad] * (width-len(p)) + p for p in prompts], device=device)
    mask = torch.tensor([[0] * (width-len(p)) + [1] * len(p) for p in prompts], device=device)
    with torch.inference_mode(), RawLogprobCapture(model.eval(), ids, eos, pad) as capture:
        seq = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=budget,
                             do_sample=True, temperature=.7, top_p=.95, top_k=0,
                             repetition_penalty=1., use_cache=True, logits_to_keep=1,
                             eos_token_id=eos, pad_token_id=pad, disable_compile=True)
        responses, old = capture.finish(seq)
    return responses, old, eos


def backward_segment(model, segment, *, version, advantage, pg_tokens, kl_tokens, pg_enabled, observe=None):
    response = segment['response_ids']
    mask = segment['action_mask']
    if (segment['policy_version'] != version or segment['logprob_contract'] != LOGPROB_CONTRACT
            or len(response) != len(mask) or any(type(v) is not bool for v in mask)
            or len(response) != len(segment['old_logprobs'])
            or len(response) != len(segment['reference_logprobs'])
            or kl_tokens <= 0 or (pg_enabled and pg_tokens <= 0)):
        raise ValueError('invalid or stale GRPO segment')
    hidden = action_hidden(model.train(), segment['prompt_ids'], response)
    device = hidden.device
    cols = [torch.tensor(c, device=device) for c in
            (response, segment['old_logprobs'], segment['reference_logprobs'], mask)]
    initial_forward = True
    def loss(h, y, old, ref, keep):
        lp = logprobs(model.lm_head, h, y)
        value = action_loss(lp, old, ref, keep, advantage, pg_tokens, kl_tokens, pg_enabled)
        if initial_forward and observe is not None:
            observe(value, lp, old, ref, keep, kl_tokens)
        return value
    chunks = zip(hidden.split(64), *(c.split(64) for c in cols))
    value = sum(checkpoint(loss, *args, use_reentrant=False) for args in chunks)
    initial_forward = False
    if not value.isfinite():
        raise FloatingPointError('nonfinite GRPO segment loss')
    value.backward()
    return value.item()
