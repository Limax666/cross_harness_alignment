import asyncio
import math
import pytest

from rl_harness_online import ContextSession, sample_segment, policy_loss_parts
from rl_harness_segments import expand_rollouts


def test_compaction_is_repeated_auditable_and_isolated():
    calls = []
    def summarize(previous, removed):
        calls.append((previous, tuple(removed)))
        return (previous or '') + ''.join(removed)
    a = ContextSession('a', 2, ('system', 'task'), summarize, policy_version='v1')
    b = ContextSession('b', 2, ('system', 'task'), summarize, policy_version='v1')
    for turn in 'abcdef':
        a.append_turn(turn)
    assert a.active_turns == ('d', 'e', 'f')
    assert a.read_external_summary() == 'abc'
    assert b.read_external_summary() is None
    assert a.protected == ('system', 'task')
    assert len(a.audit) == 3 and calls[-1] == ('ab', ('c',))
    assert all(event['old_summary_sha256'] != event['new_summary_sha256'] for event in a.audit[1:])


def test_failed_compaction_does_not_mutate_context():
    def fail(previous, turns):
        raise RuntimeError('summary failed')
    session = ContextSession('x', 1, ('system',), fail, policy_version='v1')
    for turn in 'abcde':
        session.append_turn(turn)
    with pytest.raises(RuntimeError):
        session.append_turn('f')
    assert session.active_turns == tuple('abcde') and session.audit == ()


def test_online_prefix_and_global_token_loss_across_unequal_segments():
    async def run():
        from rl_harness_segments import Recorder
        a, b = Recorder('a', 'g', 1), Recorder('b', 'g', 0)
        session = ContextSession('a', 1, ('system',), lambda prev, removed: 'summary', policy_version='v1')
        observed = []
        def render(protected, summary, turns):
            return [100, len(turns), len(summary or '')]
        async def generate(prompt):
            observed.append(tuple(prompt))
            return ([11, 12] if len(observed) == 1 else [13], [-.2, -.3] if len(observed) == 1 else [-.4], 'v1')
        await sample_segment(session, a, render, generate)
        for turn in 'abcdef': session.append_turn(turn)
        await sample_segment(session, a, render, generate)
        b.generated([100], [21], [-.1])
        segments = expand_rollouts([a, b], epsilon=0)
        assert [s.prefix for s in segments[:2]] == observed
        assert observed == [(100, 0, 0), (100, 5, 7)]
        assert segments[0].advantage == segments[1].advantage
        assert sum(len(s.tokens) for s in segments) == 4
        # Unequal microbatches must each use the same global denominator.
        mb1 = policy_loss_parts([[1., 2.], [3.]], total_tokens=4)
        mb2 = policy_loss_parts([[4.]], total_tokens=4)
        assert mb1 + mb2 == pytest.approx(2.5)
        # Per-token finite-difference derivative equals 1 / global token count.
        delta = 1e-5
        assert (policy_loss_parts([[1.+delta, 2.], [3.]], total_tokens=4) - mb1) / delta == pytest.approx(.25)
    asyncio.run(run())


def test_reject_bad_generator_and_wrong_global_count():
    from rl_harness_segments import Recorder
    s = ContextSession('a', 0, ('system',), None, policy_version='v1')
    r = Recorder('a', 'g', 1)
    async def broken(prompt): return ([1, 2], [-.2], 'v1')
    with pytest.raises(ValueError):
        asyncio.run(sample_segment(s, r, lambda *args: [4], broken))
    assert r.generations == []
    with pytest.raises(ValueError): policy_loss_parts([[1, 2]], total_tokens=1)


def test_policy_version_must_match_before_recording():
    from rl_harness_segments import Recorder
    s = ContextSession('a', 0, ('system',), None, policy_version='step:7')
    r = Recorder('a', 'g', 1)
    async def stale(prompt): return ([1], [-.2], 'step:6')
    with pytest.raises(ValueError, match='policy version'):
        asyncio.run(sample_segment(s, r, lambda *args: [4], stale))
    assert not r.generations
