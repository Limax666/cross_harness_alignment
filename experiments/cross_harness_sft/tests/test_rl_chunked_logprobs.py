import torch

from verl.utils.torch_functional import logprobs_from_logits_v2
from verl.workers.engine.fsdp.transformer_impl import _scale_logits_by_temperature


def test_chunked_bfloat16_selected_logprobs_and_gradients():
    torch.manual_seed(7)
    logits = torch.randn(2, 41, 97, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 97, (2, 41))
    weights = torch.randn(2, 41)

    actual = logprobs_from_logits_v2(logits, labels)
    (actual * weights).sum().backward()

    reference_logits = logits.detach().float().requires_grad_()
    expected = reference_logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    (expected * weights).sum().backward()

    torch.testing.assert_close(actual.float(), expected, atol=0.025, rtol=0.002)
    torch.testing.assert_close(logits.grad.float(), reference_logits.grad, atol=0.008, rtol=0.02)


def test_inplace_temperature_scaling_retains_lm_head_gradient():
    torch.manual_seed(11)
    head = torch.nn.Linear(13, 97, bias=False)
    input_states = torch.randn(5, 13)
    targets = torch.randint(0, 97, (5,))
    temperature = torch.full((5, 1), 0.8)

    logits = head(input_states)
    scaled = _scale_logits_by_temperature(logits, temperature, is_unit_temperature=False)
    assert scaled.data_ptr() == logits.data_ptr()
    loss = scaled.log_softmax(-1).gather(-1, targets[:, None]).sum()
    loss.backward()

    reference = torch.nn.Linear(13, 97, bias=False)
    reference.load_state_dict(head.state_dict())
    reference_loss = (reference(input_states) / temperature).log_softmax(-1).gather(-1, targets[:, None]).sum()
    reference_loss.backward()
    torch.testing.assert_close(head.weight.grad, reference.weight.grad)
