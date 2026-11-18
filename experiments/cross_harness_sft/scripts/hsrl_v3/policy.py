"""Serial HF policy/reference on one explicitly leased GPU; exact action-token scoring."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

VERSION = 'hsrl-single-base-lora-raw-policy-v2'
LOGPROB_CONTRACT = 'raw_model_log_softmax_before_temperature_top_p_v1'


def _fsync_directory(path):
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def install_durable_checkpoint(temporary, directory):
    """Persist all populated directory entries before publishing the checkpoint.

    A failed barrier leaves the old journal authoritative. Staging/unreferenced
    directories survive as evidence; callers must never point at a failed save.
    """
    temporary, directory = Path(temporary), Path(directory)
    if temporary.parent != directory.parent or directory.exists():
        raise ValueError('checkpoint_install_requires_new_same_parent_destination')
    subdirectories = [p for p in temporary.rglob('*') if p.is_dir()]
    for path in sorted(subdirectories, key=lambda p: len(p.parts), reverse=True):
        _fsync_directory(path)
    _fsync_directory(temporary)
    _fsync_directory(directory.parent)
    os.rename(temporary, directory)
    _fsync_directory(directory.parent)
    # Also persist the entry for a newly created checkpoints/ directory.
    _fsync_directory(directory.parent.parent)


def validate_checkpoint_files(directory):
    """Filesystem-only recovery validation shared by restore and control tests."""
    from hsrl_v5.io import file_hash
    directory = Path(directory)
    marker = json.loads((directory / 'COMPLETE.json').read_text())
    files = marker.get('files', {})
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
              if p.is_file() and p.name != 'COMPLETE.json'}
    if (marker.get('schema') != 'hsrl-atomic-checkpoint/v1'
            or set(files) != actual or 'resume.pt' not in files
            or 'adapter/adapter_model.safetensors' not in files
            or any(Path(name).is_absolute() or '..' in Path(name).parts for name in files)
            or any(file_hash(directory / name) != sha for name, sha in files.items())):
        raise ValueError('incomplete_or_changed_checkpoint')


class Policy:
    def __init__(self, model_path: Path, seed=267):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise RuntimeError('HSRL_requires_exclusive_physical_GPU0')
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        self.torch = torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16,
                    device_map={'': 0}, attn_implementation='sdpa', local_files_only=True)
        base.requires_grad_(False)
        base.config.use_cache = False
        base.config.text_config.use_cache = False
        self.eos_ids = base.generation_config.eos_token_id
        if self.eos_ids is None:
            self.eos_ids = self.tokenizer.eos_token_id
        if isinstance(self.eos_ids, int):
            self.eos_ids = [self.eos_ids]
        if not self.eos_ids:
            raise RuntimeError('explicit_protocol_EOS_required')
        self.targets = []
        actual = dict(base.named_modules())
        for i, kind in enumerate(base.config.text_config.layer_types):
            root = f'model.language_model.layers.{i}'
            names = [f'{root}.mlp.{x}' for x in ('gate_proj', 'up_proj', 'down_proj')]
            if kind == 'linear_attention':
                names += [f'{root}.linear_attn.{x}' for x in ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj')]
            elif kind == 'full_attention':
                names += [f'{root}.self_attn.{x}' for x in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
            else:
                raise RuntimeError('unsupported_language_layer:' + kind)
            if any(name not in actual for name in names):
                raise RuntimeError('LoRA_module_contract_mismatch')
            self.targets += names
        self.model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32, lora_dropout=0.0, bias='none', target_modules=self.targets,
            init_lora_weights=True), autocast_adapter_dtype=True)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.version = 0
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not self.trainable or any('lora_' not in name for name, p in self.model.named_parameters() if p.requires_grad):
            raise RuntimeError('only_standard_LoRA_parameters_may_train')
        self.frozen_signature = self.frozen_versions()

    def frozen_versions(self):
        content = [(name, p._version, tuple(p.shape), str(p.dtype)) for name, p in self.model.named_parameters() if not p.requires_grad]
        if any(p.grad is not None for p in self.model.parameters() if not p.requires_grad):
            raise RuntimeError('frozen_base_has_gradient')
        return hashlib.sha256(json.dumps(content).encode()).hexdigest()

    def adapter_hash(self):
        from peft import get_peft_model_state_dict
        result = hashlib.sha256()
        for name, tensor in sorted(get_peft_model_state_dict(self.model).items()):
            result.update(name.encode())
            result.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
        return result.hexdigest()

    def hidden(self, prompt_ids, response_ids):
        torch = self.torch
        if not prompt_ids or not response_ids:
            raise ValueError('nonempty_exact_prompt_and_response_required')
        ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device='cuda:0')
        backbone = self.model.get_base_model().model.language_model
        hidden = backbone(input_ids=ids, attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(ids.shape[1], device=ids.device)[None, :],
            use_cache=False, past_key_values=None, return_dict=True).last_hidden_state
        return hidden[0, len(prompt_ids)-1:len(prompt_ids)+len(response_ids)-1]

    def score(self, prompt_ids, response_ids, *, reference=False):
        torch = self.torch
        self.model.eval()
        context = self.model.disable_adapter() if reference else nullcontext()
        output = []
        with context, torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            hidden = self.hidden(prompt_ids, response_ids)
            head = self.model.get_base_model().lm_head
            for start in range(0, len(response_ids), 64):
                target = torch.tensor(response_ids[start:start+64], dtype=torch.long, device='cuda:0')
                logits = head(hidden[start:start+64]).float()
                logp = logits.gather(-1, target[:, None]).squeeze(-1) - logits.logsumexp(-1)
                if not torch.isfinite(logp).all():
                    raise RuntimeError('nonfinite_action_logprob')
                output.extend(logp.cpu().tolist())
        return output

    def generate(self, prompt_ids, max_new_tokens):
        torch = self.torch
        self.model.eval()
        ids = torch.tensor([prompt_ids], dtype=torch.long, device='cuda:0')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            result = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                do_sample=True, temperature=0.7, top_p=0.95, top_k=0, repetition_penalty=1.0,
                max_new_tokens=max_new_tokens, use_cache=True, logits_to_keep=1,
                eos_token_id=self.eos_ids,
                pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                              else self.eos_ids[0]), output_scores=False,
                output_logits=False, return_dict_in_generate=False)
        response = result[0, len(prompt_ids):].cpu().tolist()
        del result, ids
        if not response:
            raise RuntimeError('generation_returned_no_sampled_tokens')
        old = self.score(prompt_ids, response)
        reference = self.score(prompt_ids, response, reference=True)
        visible_ids = response[:-1] if response[-1] in self.eos_ids else response
        return {'response_ids': response, 'text': self.tokenizer.decode(visible_ids, skip_special_tokens=False),
                'old_logprobs': old, 'reference_logprobs': reference, 'policy_version': self.version,
                'length_limited': len(response) >= max_new_tokens and response[-1] not in self.eos_ids,
                'stop_reason': 'eos' if response[-1] in self.eos_ids else 'generation_limit',
                'protocol_eos_ids': self.eos_ids,
                'text_extraction': 'exact_sampled_ids_except_terminal_EOS_skip_special_false_v1',
                'logprob_contract': LOGPROB_CONTRACT,
                'sampling': {'temperature': 0.7, 'top_p': 0.95, 'top_k': 0}}

    def backward_segment(self, segment, advantage, denominator=None, kl_coefficient=.001, clip=.2,
                         *, pg_denominator=None, kl_denominator=None, pg_enabled=True):
        torch = self.torch
        from torch.utils.checkpoint import checkpoint
        if segment['policy_version'] != self.version:
            raise RuntimeError('stale_rollout_policy')
        self.model.train()
        head = self.model.get_base_model().lm_head
        response = segment['response_ids']
        if not (len(response) == len(segment['old_logprobs']) == len(segment['reference_logprobs'])):
            raise RuntimeError('action_logprob_length_mismatch')
        if segment.get('logprob_contract', LOGPROB_CONTRACT) != LOGPROB_CONTRACT:
            raise ValueError('mixed_logprob_contract')
        pg_denominator = denominator if pg_denominator is None else pg_denominator
        kl_denominator = denominator if kl_denominator is None else kl_denominator
        action = segment.get('action_mask', [True] * len(response))
        if len(action) != len(response) or any(type(v) is not bool for v in action):
            raise ValueError('invalid_action_mask')
        pg_mask = [v and pg_enabled for v in action]
        if (any(pg_mask) and (pg_denominator is None or pg_denominator <= 0)
                or any(action) and (kl_denominator is None or kl_denominator <= 0)):
            raise ValueError('invalid_global_token_denominator')
        losses = []
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hidden = self.hidden(segment['prompt_ids'], response)
            for start in range(0, len(response), 64):
                target = torch.tensor(response[start:start+64], dtype=torch.long, device='cuda:0')
                old = torch.tensor(segment['old_logprobs'][start:start+64], device='cuda:0')
                ref = torch.tensor(segment['reference_logprobs'][start:start+64], device='cuda:0')
                pg_weights = torch.tensor(pg_mask[start:start+64], device='cuda:0', dtype=torch.float32)
                kl_weights = torch.tensor(action[start:start+64], device='cuda:0', dtype=torch.float32)
                pg_weights = pg_weights / (pg_denominator if any(pg_mask) else 1)
                kl_weights = kl_weights / kl_denominator
                def objective(h, target, old, ref, pg_weights, kl_weights):
                    logits = head(h).float()
                    lp = logits.gather(-1, target[:, None]).squeeze(-1) - logits.logsumexp(-1)
                    ratio = (lp-old).exp()
                    pg = -torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage)
                    delta = ref-lp
                    kl = delta.exp()-delta-1
                    return (pg * pg_weights).sum() + kl_coefficient * (kl * kl_weights).sum()
                loss = checkpoint(objective, hidden[start:start+64], target, old, ref,
                                  pg_weights, kl_weights, use_reentrant=False)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite_GRPO_loss')
                losses.append(loss)
            total = torch.stack(losses).sum()
            value = float(total.detach())
            total.backward()
        return value

    def rng_state(self):
        import numpy as np
        return {'torch_rng': self.torch.get_rng_state(), 'cuda_rng': self.torch.cuda.get_rng_state(),
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}

    def restore_rng(self, state):
        import numpy as np
        self.torch.set_rng_state(state['torch_rng'])
        self.torch.cuda.set_rng_state(state['cuda_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])

    def save(self, directory: Path, optimizer, state, scheduler=None):
        torch = self.torch
        from hsrl_v5.io import file_hash, atomic_json
        directory = Path(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        if directory.exists():
            raise FileExistsError(directory)
        temporary = Path(tempfile.mkdtemp(prefix='.' + directory.name + '-', dir=directory.parent))
        self.model.save_pretrained(temporary / 'adapter', safe_serialization=True)
        torch.save({'optimizer': optimizer.state_dict(), 'rng': self.rng_state(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None, 'state': state,
                    'version': self.version, 'adapter_hash': self.adapter_hash(),
                    'frozen_signature': self.frozen_signature}, temporary / 'resume.pt')
        files = {p.relative_to(temporary).as_posix(): file_hash(p)
                 for p in temporary.rglob('*') if p.is_file()}
        for name in files:
            with (temporary / name).open('rb') as stream:
                os.fsync(stream.fileno())
        atomic_json(temporary / 'COMPLETE.json', {'schema': 'hsrl-atomic-checkpoint/v1', 'files': files})
        install_durable_checkpoint(temporary, directory)

    def load(self, directory: Path, optimizer, scheduler=None):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        directory = Path(directory)
        validate_checkpoint_files(directory)
        state = self.torch.load(directory / 'resume.pt', map_location='cpu', weights_only=False)
        set_peft_model_state_dict(self.model, load_file(directory / 'adapter/adapter_model.safetensors'))
        optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        self.version = state['version']
        if state['adapter_hash'] != self.adapter_hash():
            raise RuntimeError('restored_adapter_hash_mismatch')
        self.restore_rng(state['rng'])
        # Parameter copies bump _version, so bind a fresh frozen signature after
        # restore, checking optimizer trainability again rather than old counters.
        self.frozen_signature = self.frozen_versions()
        return state['state']
