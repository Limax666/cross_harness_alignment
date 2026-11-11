import pytest
from rl_harness_segments import Recorder, expand_rollouts
from rl_harness_segment_dataproto import pack_segment_rows, make_dataproto


def sample_segments():
    a = Recorder('a', 'g', 1.)
    a.generated([1, 2], [3, 4], [-.1, -.2])
    a.generated([8, 9, 10], [5], [-.3])
    b = Recorder('b', 'g', 0.)
    b.generated([1], [6], [-.4])
    return expand_rollouts([a, b], epsilon=0)


def test_packing_preserves_prefix_masks_and_rollout_advantages():
    batch = pack_segment_rows(sample_segments(), pad_token_id=0)
    assert batch['prompts'] == [[0, 1, 2], [8, 9, 10], [0, 0, 1]]
    assert batch['responses'] == [[3, 4], [5, 0], [6, 0]]
    assert batch['attention_mask'] == [[0, 1, 1, 1, 1], [1, 1, 1, 1, 0], [0, 0, 1, 1, 0]]
    assert batch['position_ids'] == [[0, 0, 1, 2, 3], [0, 1, 2, 3, 0], [0, 0, 0, 1, 0]]
    assert batch['response_mask'] == [[1, 1], [1, 0], [1, 0]]
    assert batch['advantages'][0] == pytest.approx([2**-.5] * 2)
    assert batch['advantages'][1] == pytest.approx([2**-.5, 0])
    assert batch['rollout_log_probs'][1] == pytest.approx([-.3, 0])
    assert batch['rollout_id'] == ['a', 'a', 'b']


def test_pack_never_truncates_prefix_or_generated_tokens():
    with pytest.raises(ValueError, match='prefix'):
        pack_segment_rows(sample_segments(), pad_token_id=0, max_prompt_length=2)
    with pytest.raises(ValueError, match='response'):
        pack_segment_rows(sample_segments(), pad_token_id=0, max_response_length=1)


def test_expansion_must_not_recompute_advantages():
    batch = pack_segment_rows(sample_segments(), pad_token_id=0)
    assert batch['advantages'][2][0] == pytest.approx(-(2**-.5))


def test_real_dataproto_and_autograd_global_token_mean():
    import torch
    from verl import DataProto
    batch = make_dataproto(sample_segments(), pad_token_id=0)
    assert isinstance(batch, DataProto)
    assert batch.batch['input_ids'].tolist()[1] == [8, 9, 10, 5, 0]
    assert batch.batch['response_mask'].sum().item() == 4
    assert batch.batch['advantages'][1, 0].item() == pytest.approx(2**-.5)
    # Simulate two DDP ranks with unequal valid-token counts. DDP averages
    # rank gradients; VeRL multiplies each local sum by dp_size/global_count.
    from verl.trainer.ppo.core_algos import agg_loss
    param = torch.tensor(0.2, requires_grad=True)
    masks = [torch.tensor([[1., 1., 1.]]), torch.tensor([[1., 0., 0.]])]
    rank_losses = [agg_loss((param - torch.tensor([[1., 2., 3.]])).square(), masks[0], 'token-mean', dp_size=2, batch_num_tokens=4),
                   agg_loss((param - torch.tensor([[4., 0., 0.]])).square(), masks[1], 'token-mean', dp_size=2, batch_num_tokens=4)]
    ddp_mean = sum(rank_losses) / 2
    ddp_mean.backward()
    gradient = param.grad.item()
    reference = torch.tensor(0.2, requires_grad=True)
    ((reference - torch.tensor([1., 2., 3., 4.])) ** 2).mean().backward()
    assert ddp_mean.item() == pytest.approx(((reference.detach() - torch.tensor([1., 2., 3., 4.])) ** 2).mean().item())
    assert gradient == pytest.approx(reference.grad.item())


def test_reject_inconsistent_advantage_for_same_rollout():
    from dataclasses import replace
    segments = sample_segments()
    segments[1] = replace(segments[1], advantage=0.0)
    with pytest.raises(ValueError, match='rollout'):
        pack_segment_rows(segments, pad_token_id=0)
