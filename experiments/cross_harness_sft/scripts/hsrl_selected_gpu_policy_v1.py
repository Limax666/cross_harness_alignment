"""Versioned physical-GPU1 initializer; all other policy behavior is inherited.

No environment spoofing: CUDA logical device 0 maps to the exclusively selected
physical GPU1 because the unchanged process environment is exactly ``1``.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import os
from pathlib import Path

from hsrl_v3 import policy as original_policy

VERSION = 'hsrl-selected-physical-gpu1-policy-v1'
ORIGINAL_POLICY_SHA256 = 'a73973caf58668ce14f20f3e34d5487a381bbaad3b6529df525acb3bfa913b47'


def verify_initializer_contract():
    """Refuse source drift; only the original initial GPU guard may differ."""
    source = Path(original_policy.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != ORIGINAL_POLICY_SHA256:
        raise RuntimeError('selected_GPU_policy_original_source_changed')
    def initializer(source_text, class_name):
        tree = ast.parse(source_text)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == class_name)
        return copy.deepcopy(next(node for node in cls.body
                                  if isinstance(node, ast.FunctionDef) and node.name == '__init__'))
    original = initializer(source, 'Policy')
    selected = initializer(Path(__file__).read_bytes(), 'SelectedGPU1PolicyV1')
    expected_guard = ast.parse("if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':\n"
                               "    raise RuntimeError('HSRL_requires_exclusive_physical_GPU1')").body[0]
    if ast.dump(selected.body[0]) != ast.dump(expected_guard):
        raise RuntimeError('selected_GPU_policy_guard_changed')
    original.body = original.body[1:]
    selected.body = selected.body[1:]
    if ast.dump(original) != ast.dump(selected):
        raise RuntimeError('selected_GPU_policy_initializer_semantics_changed')
    return ORIGINAL_POLICY_SHA256


class SelectedGPU1PolicyV1(original_policy.Policy):
    def __init__(self, model_path: Path, seed=267):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
            raise RuntimeError('HSRL_requires_exclusive_physical_GPU1')
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
