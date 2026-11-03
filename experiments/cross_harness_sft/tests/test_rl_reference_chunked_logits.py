import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'vendor/verl'), str(ROOT / 'scripts')]

from verl.utils.experimental.torch_functional import FusedLinearForPPO
from verl.utils.reference_chunked import reference_fused_model_config, reference_chunk_size


def test_chunked_logprobs_and_gradients_match_dense():
    torch.manual_seed(17)
    h = torch.randn(2, 7, 5, requires_grad=True)
    w = torch.randn(23, 5, requires_grad=True)
    labels = torch.randint(0, 23, (2, 7))
    weighting = torch.randn(2, 7)
    temp = 0.8
    logp, entropy = FusedLinearForPPO(chunk_size=3)(h, w, labels, temp)
    grads = torch.autograd.grad((logp * weighting + entropy * 0.1).sum(), (h, w))

    h2, w2 = h.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    logits = (h2 @ w2.T) / temp
    all_logp = logits.log_softmax(dim=-1)
    expected_logp = all_logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    expected_entropy = -(all_logp.exp() * all_logp).sum(dim=-1)
    expected_grads = torch.autograd.grad((expected_logp * weighting + expected_entropy * 0.1).sum(), (h2, w2))
    torch.testing.assert_close(logp, expected_logp, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(entropy, expected_entropy, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(grads, expected_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_bf16_chunked_reference_matches_normal_verl_logprobs():
    from verl.utils.torch_functional import logprobs_from_logits

    torch.manual_seed(31)
    h = torch.randn(1, 13, 16, dtype=torch.bfloat16)
    w = torch.randn(67, 16, dtype=torch.bfloat16)
    labels = torch.randint(0, 67, (1, 13))
    with torch.no_grad():
        actual, _ = FusedLinearForPPO(chunk_size=3)(h, w, labels, 0.8)
        logits = (h @ w.T).div_(torch.tensor(0.8, dtype=torch.bfloat16))
        expected = logprobs_from_logits(logits, labels)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-3, atol=2e-2)


def test_reference_override_preserves_actor_config():
    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class ModelConfig:
        use_fused_kernels: bool = False
        fused_kernel_options: dict = field(default_factory=dict)

    actor = ModelConfig()
    reference = reference_fused_model_config(actor)
    assert reference.use_fused_kernels is True
    assert reference.fused_kernel_options['impl_backend'] == 'torch'
    assert actor.use_fused_kernels is False


def test_chunk_size_validated(monkeypatch):
    monkeypatch.delenv('VERL_REF_FUSED_CHUNK_SIZE', raising=False)
    assert reference_chunk_size() == 512
    monkeypatch.setenv('VERL_REF_FUSED_CHUNK_SIZE', '128')
    assert reference_chunk_size() == 128
    monkeypatch.setenv('VERL_REF_FUSED_CHUNK_SIZE', '0')
    with pytest.raises(ValueError):
        reference_chunk_size()


def test_qwen_reference_world_one_dtensor_matches_dense_without_full_copy(monkeypatch):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, distribute_tensor
    from torch.distributed.tensor.placement_types import Shard
    from verl.models.transformers.qwen3_5 import forward_with_torch_backend

    class Output:
        hidden_states = None

        def __init__(self, hidden):
            self.hidden = hidden

        def __getitem__(self, index):
            assert index == 0
            return self.hidden

    with tempfile.TemporaryDirectory() as tmp:
        dist.init_process_group('gloo', init_method=f'file://{tmp}/rendezvous', rank=0, world_size=1)
        try:
            mesh = init_device_mesh('cpu', (1,))
            torch.manual_seed(29)
            hidden = torch.randn(1, 7, 5)
            weight = torch.randn(23, 5)
            labels = torch.randint(0, 23, (1, 7))
            dtensor = distribute_tensor(weight, mesh, [Shard(0)])
            model = SimpleNamespace(model=lambda *_args, **_kwargs: Output(hidden),
                                    lm_head=SimpleNamespace(weight=dtensor),
                                    _verl_fused_kernels_backend='torch')
            monkeypatch.setenv('VERL_REF_FUSED_CHUNK_SIZE', '3')
            monkeypatch.setattr(DTensor, 'full_tensor', lambda *_a, **_k: pytest.fail('unexpected full copy'))
            with torch.no_grad():
                result = forward_with_torch_backend(model, input_ids=labels, shift_labels=labels, temperature=0.8)
            expected = ((hidden @ weight.T) / 0.8).log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            torch.testing.assert_close(result.log_probs, expected, rtol=1e-5, atol=1e-6)
        finally:
            dist.destroy_process_group()
