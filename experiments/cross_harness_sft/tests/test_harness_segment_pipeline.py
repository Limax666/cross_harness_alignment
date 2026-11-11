import pytest
from rl_harness_segment_pipeline import prepare_segmented_step
from test_harness_segment_admission import fixture


def test_segmented_step_fails_closed_for_unsupported_algorithm_before_scoring():
    rollouts, evidence, cells, assignments = fixture()
    class Trainer:
        def _compute_old_log_prob(self, batch): raise AssertionError('must not run')
    with pytest.raises(ValueError, match='GRPO'):
        prepare_segmented_step(Trainer(), rollouts, evidence, cells=cells, assignments=assignments,
            policy_snapshot='step:1', algorithm='CHS-PO', pad_token_id=0,
            optimizer_segment_capacity=5, max_prompt_length=8, max_response_length=8)


def test_cpu_dataproto_scores_and_admits_original_rollouts():
    import torch
    from verl import DataProto
    rollouts,evidence,cells,assignments = fixture()
    class Trainer:
        def _compute_old_log_prob(self,batch):
            return DataProto.from_dict(tensors={'old_log_probs': torch.full_like(batch.batch['advantages'], -.2)}), 0.
        def _compute_ref_log_prob(self,batch):
            return DataProto.from_dict(tensors={'ref_log_prob': torch.full_like(batch.batch['advantages'], -.3)})
    batch,report=prepare_segmented_step(Trainer(),rollouts,evidence,cells=cells,assignments=assignments,
        policy_snapshot='step:1',algorithm='GRPO',pad_token_id=0,optimizer_segment_capacity=5,
        max_prompt_length=8,max_response_length=8)
    assert report['rollouts'] == 4 and len(batch.batch['responses']) == 5
    assert batch.batch['prompts'][1].tolist() == [80]
    assert batch.batch['advantages'][0,0].item() == pytest.approx(batch.batch['advantages'][1,0].item())
