import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'vendor/verl'),str(ROOT/'scripts')]
from verl.utils.scaled_chunked_outputs import scaled_chunked_logprobs_entropy


class CustomIdentity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad):
        return grad


def test_custom_function_view_logprobs_entropy_and_gradients():
    torch.manual_seed(13)
    source=torch.randn(1,7,17,requires_grad=True)
    labels=torch.randint(0,17,(7,))
    temps=torch.linspace(.65,1.1,7)
    w1=torch.randn(7)
    w2=torch.randn(7)
    logits=CustomIdentity.apply(source).squeeze(0)
    logp,entropy=scaled_chunked_logprobs_entropy(logits,labels,temps,chunk_size=3)
    ((logp*w1+entropy*w2).sum()).backward()
    reference=source.detach().clone().requires_grad_()
    scaled=reference.squeeze(0)/temps[:,None]
    all_logp=scaled.log_softmax(-1)
    expected=all_logp.gather(-1,labels[:,None]).squeeze(-1)
    expected_entropy=-(all_logp.exp()*all_logp).sum(-1)
    ((expected*w1+expected_entropy*w2).sum()).backward()
    torch.testing.assert_close(logp,expected,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(entropy,expected_entropy,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(source.grad,reference.grad,rtol=1e-5,atol=1e-6)


def test_bf16_matches_scaled_selected_logprobs():
    torch.manual_seed(21)
    logits=torch.randn(9,19,dtype=torch.bfloat16,requires_grad=True)
    labels=torch.randint(0,19,(9,))
    temps=torch.full((9,),0.8)
    logp,entropy=scaled_chunked_logprobs_entropy(logits,labels,temps,chunk_size=4)
    scaled=(logits.detach()/temps.to(torch.bfloat16)[:,None]).float()
    expected=scaled.log_softmax(-1).gather(-1,labels[:,None]).squeeze(-1)
    torch.testing.assert_close(logp,expected,rtol=5e-3,atol=2e-2)
    (logp.sum()+entropy.sum()*.1).backward()
    assert torch.isfinite(logits.grad).all()
