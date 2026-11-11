"""Isolated CPU online segment adapter; not wired into the legacy VeRL loop.

Turns are complete serialized units supplied by the caller; protected context
must include immutable system instructions, original task and tool contract.
"""
from __future__ import annotations

import hashlib
import json


def _hash(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


class ContextSession:
    def __init__(self, rollout_id, mode, protected, summarize, *, policy_version):
        if not isinstance(rollout_id, str) or not rollout_id or type(mode) is not int or mode not in (0, 1, 2):
            raise ValueError('invalid session identity or context mode')
        if not isinstance(policy_version, str) or not policy_version:
            raise ValueError('policy version is required')
        self.policy_version = policy_version
        if mode and not callable(summarize):
            raise ValueError('compaction requires a summarizer')
        self.rollout_id, self.mode = rollout_id, mode
        self.protected = tuple(protected)
        if not self.protected or any(not isinstance(x, str) or not x for x in self.protected):
            raise ValueError('protected instructions and task are required')
        self._summarize, self._turns, self._summary, self._audit = summarize, (), None, ()
        self._sampling = False

    @property
    def active_turns(self):
        return self._turns

    @property
    def audit(self):
        return tuple(dict(event) for event in self._audit)

    def read_external_summary(self):
        if self.mode != 2:
            raise ValueError('external summary is only available in C2')
        return self._summary

    def append_turn(self, turn):
        if self._sampling:
            raise ValueError('cannot change context during sampling')
        if not isinstance(turn, str) or not turn:
            raise ValueError('expected a complete serialized turn')
        pending = self._turns + (turn,)
        keep = {0: None, 1: 5, 2: 3}[self.mode]
        if keep is None or len(pending) <= keep:
            self._turns = pending
            return
        removed, retained = pending[:-keep], pending[-keep:]
        summary = self._summarize(self._summary, removed)
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError('summary must be a nonempty string')
        event = {'rollout_id': self.rollout_id, 'mode': self.mode,
                 'old_summary_sha256': _hash(self._summary), 'removed_sha256': _hash(removed),
                 'new_summary_sha256': _hash(summary), 'retained_sha256': _hash(retained)}
        self._summary, self._turns, self._audit = summary, retained, self._audit + (event,)


async def sample_segment(session, recorder, render, generate):
    """Record exactly the immutable token prefix given to the async generator."""
    if session.rollout_id != recorder.rollout_id or session._sampling:
        raise ValueError('rollout identity mismatch or concurrent generation')
    session._sampling = True
    try:
        visible_summary = session._summary if session.mode == 1 else None
        prompt = tuple(render(session.protected, visible_summary, session.active_turns))
        if not prompt or any(type(x) is not int or x < 0 for x in prompt):
            raise ValueError('renderer returned invalid token prefix')
        tokens, logprobs, observed_version = await generate(prompt)
        if observed_version != session.policy_version:
            raise ValueError('generation policy version mismatch')
        recorder.generated(prompt, tokens, logprobs)
        return tuple(tokens)
    finally:
        session._sampling = False


def policy_loss_parts(per_segment_losses, *, total_tokens):
    """Return one microbatch numerator / global token count, preserving gradients.

    Distributed training must additionally account for the framework's gradient
    averaging and must use a single count covering the entire optimizer batch.
    """
    if type(total_tokens) is not int or total_tokens <= 0:
        raise ValueError('positive global token count required')
    values = [v for segment in per_segment_losses for v in segment]
    if not values or len(values) > total_tokens:
        raise ValueError('invalid local token count')
    return sum(values) / total_tokens
