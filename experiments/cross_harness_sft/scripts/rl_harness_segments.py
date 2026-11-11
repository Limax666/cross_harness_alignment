"""Lossless generation records and rollout-level GRPO expansion (CPU framework neutral)."""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass(frozen=True)
class Segment:
    rollout_id: str
    group_id: str
    prefix: tuple
    tokens: tuple
    logprobs: tuple
    reward: float
    advantage: float


class Recorder:
    def __init__(self, rollout_id, group_id, reward):
        if not isinstance(rollout_id, str) or not rollout_id or not isinstance(group_id, str) or not group_id or not math.isfinite(reward):
            raise ValueError('invalid rollout identity or reward')
        self.rollout_id, self.group_id, self.reward = rollout_id, group_id, float(reward)
        self.generations = []

    def generated(self, prefix, tokens, logprobs):
        prefix, tokens, logprobs = tuple(prefix), tuple(tokens), tuple(logprobs)
        if not prefix or not tokens or len(tokens) != len(logprobs) or any(type(x) is not int or x < 0 for x in prefix + tokens) or any(not math.isfinite(x) for x in logprobs):
            raise ValueError('invalid generation or sampled log probabilities')
        self.generations.append((prefix, tokens, logprobs))


def expand_rollouts(rollouts, *, epsilon=1e-6):
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError('invalid epsilon')
    groups, ids = defaultdict(list), set()
    for rollout in rollouts:
        if rollout.rollout_id in ids or not rollout.generations:
            raise ValueError('duplicate or empty rollout')
        ids.add(rollout.rollout_id)
        groups[rollout.group_id].append(rollout)
    if not ids:
        raise ValueError('empty batch')
    advantages = {}
    for group in groups.values():
        if len(group) < 2:
            raise ValueError('GRPO group requires at least two original rollouts')
        mean = sum(r.reward for r in group) / len(group)
        std = math.sqrt(sum((r.reward - mean) ** 2 for r in group) / (len(group) - 1))
        for r in group:
            advantages[r.rollout_id] = (r.reward - mean) / (std + epsilon) if std + epsilon else 0.0
    return [Segment(r.rollout_id, r.group_id, p, t, lp, r.reward, advantages[r.rollout_id])
            for r in rollouts for p, t, lp in r.generations]


def token_mean(per_segment_losses):
    """Mean scalar policy losses over tokens, not over segments."""
    tokens = [value for segment in per_segment_losses for value in segment]
    if not tokens:
        raise ValueError('no policy tokens')
    return sum(tokens) / len(tokens)


def compact_history(turns, mode, summarize):
    """Compact complete caller-defined turns; caller must persist and expose C2's summary.

    The caller supplies all previous summary material in the old turns when
    compacting again. This function does not execute a summarizer or write files.
    """
    if type(mode) is not int or mode not in (0, 1, 2):
        raise ValueError('unknown context mode')
    turns = list(turns)
    if mode == 0 or len(turns) <= (5 if mode == 1 else 3):
        return turns, None
    keep = 5 if mode == 1 else 3
    summary = summarize(turns[:-keep])
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError('compaction requires a nonempty summary')
    if mode == 1:
        return [{'role': 'user', 'content': summary}] + turns[-keep:], None
    return turns[-keep:], summary
