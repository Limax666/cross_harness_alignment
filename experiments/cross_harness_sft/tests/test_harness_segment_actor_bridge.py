import pytest
from rl_harness_segment_actor_bridge import prepare_actor_batch


class Matrix:
    def __init__(self, rows, cols): self.shape = (rows, cols)
    def isfinite(self):
        class Result:
            def all(self): return True
        return Result()


class Proto:
    def __init__(self, fields, meta=None): self.batch, self.meta_info = fields, meta or {}
    def union(self, other):
        assert not (self.batch.keys() & other.batch.keys())
        self.batch.update(other.batch)
        return self


def batch():
    return Proto({'responses': Matrix(3, 2), 'response_mask': Matrix(3, 2),
                  'advantages': Matrix(3, 2)}, {'segment_advantages_precomputed': True})


def test_actor_and_reference_score_same_full_segment_batch():
    item = batch()
    calls = []
    def old(given):
        assert given is item
        calls.append('actor')
        return Proto({'old_log_probs': Matrix(3, 2), 'entropys': Matrix(3, 2)}), 0.1
    def reference(given):
        assert given is item
        calls.append('reference')
        return Proto({'ref_log_prob': Matrix(3, 2)})
    result = prepare_actor_batch(item, old, reference, loss_agg_mode='token-mean',
                                 optimizer_segment_capacity=3)
    assert result is item and calls == ['actor', 'reference']
    assert 'old_log_probs' in item.batch and 'ref_log_prob' in item.batch
    assert 'entropys' not in item.batch
    assert item.meta_info['segment_advantages_precomputed']


def test_rejects_incorrect_loss_shape_and_unsafe_splitting():
    with pytest.raises(ValueError, match='minibatch'):
        prepare_actor_batch(batch(), None, None, loss_agg_mode='token-mean', optimizer_segment_capacity=2)
    with pytest.raises(ValueError, match='token-mean'):
        prepare_actor_batch(batch(), None, None, loss_agg_mode='seq-mean-token-mean', optimizer_segment_capacity=3)
    def bad(_): return Proto({'old_log_probs': Matrix(2, 2)}), 0
    with pytest.raises(ValueError, match='shape'):
        prepare_actor_batch(batch(), bad, lambda _: None, loss_agg_mode='token-mean', optimizer_segment_capacity=3)


def test_refuses_nonfinite_scoring():
    class BadMatrix(Matrix):
        def isfinite(self):
            class Result:
                def all(self): return False
            return Result()
    def old(_): return Proto({'old_log_probs': BadMatrix(3, 2)}), 0
    def ref(_): return Proto({'ref_log_prob': Matrix(3, 2)})
    with pytest.raises(ValueError, match='nonfinite'):
        prepare_actor_batch(batch(), old, ref, loss_agg_mode='token-mean', optimizer_segment_capacity=3)
