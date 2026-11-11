import itertools
from collections import Counter
import pytest

from rl_harness_scaling import TRAIN_CONFIGS, build_assignments
from rl_harness_segments import Recorder, expand_rollouts, token_mean


def test_design_and_k2_assignments_are_reproducible_and_nearest():
    assert len(set(TRAIN_CONFIGS)) == 18
    for i, j in itertools.combinations(range(5), 2):
        assert set(Counter((c[i], c[j]) for c in TRAIN_CONFIGS).values()) == {2}
        assert len(Counter((c[i], c[j]) for c in TRAIN_CONFIGS)) == 9
    tasks = [{'task_id': 'a', 'version': 'v1'}, {'task_id': 'b', 'version': 'v1'}]
    ledger = build_assignments(tasks, seed=7, policy_version='step:1')
    assert ledger == build_assignments(tasks, seed=7, policy_version='step:1')
    for row in ledger['assignments']:
        a, b = row['configs']
        distance = lambda c: sum(x != y for x, y in zip(a, c))
        assert a != b and distance(b) == min(distance(c) for c in TRAIN_CONFIGS if c != tuple(a))


def test_segments_preserve_actual_prefix_and_normalize_original_rollouts():
    a = Recorder('a', 'group', reward=1.0)
    a.generated([1, 2], [3, 4], [-.1, -.2])
    a.generated([9, 8], [5], [-.3])  # history changed by compaction
    b = Recorder('b', 'group', reward=0.0)
    b.generated([1], [6], [-.4])
    segments = expand_rollouts([a, b], epsilon=0)
    assert [s.prefix for s in segments] == [(1, 2), (9, 8), (1,)]
    assert segments[0].advantage == segments[1].advantage == pytest.approx(2**-.5)
    assert segments[2].advantage == pytest.approx(-(2**-.5))
    assert token_mean([[1., 2.], [3.], [4.]]) == 2.5


def test_segment_validation_rejects_missing_logprobs_and_duplicate_rollouts():
    a = Recorder('a', 'g', 1.)
    with pytest.raises(ValueError):
        a.generated([1], [2, 3], [-.1])
    a.generated([1], [2], [-.1])
    with pytest.raises(ValueError):
        expand_rollouts([a, a])
    with pytest.raises(ValueError):
        token_mean([[]])


def test_compaction_keeps_recent_turns_and_externalizes_c2():
    from rl_harness_segments import compact_history
    history = [{'role': 'user', 'content': str(i)} for i in range(7)]
    summarize = lambda old: 'summary:' + ','.join(x['content'] for x in old)
    in_context, external = compact_history(history, 1, summarize)
    assert in_context[0]['content'] == 'summary:0,1'
    assert in_context[1:] == history[-5:] and external is None
    in_context, external = compact_history(history, 2, summarize)
    assert in_context == history[-3:] and external == 'summary:0,1,2,3'
    assert compact_history(history, 0, summarize) == (history, None)
