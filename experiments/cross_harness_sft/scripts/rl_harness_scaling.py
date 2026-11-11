"""Five-factor selection and immutable assignment metadata, independent of the legacy pool."""
from __future__ import annotations

import hashlib
import json
import random

from rl_whitebox_harness import _OA

TRAIN_CONFIGS = _OA


def build_assignments(tasks, *, seed, policy_version):
    """Assign each versioned task a uniformly sampled anchor and nearest distinct peer."""
    if type(seed) is not int or not isinstance(policy_version, str) or not policy_version:
        raise ValueError('seed and policy version are required')
    rng = random.Random(seed)
    rows, seen = [], set()
    for task in tasks:
        task_id, version = task.get('task_id'), task.get('version')
        if not isinstance(task_id, str) or not task_id or not isinstance(version, str) or not version or task_id in seen:
            raise ValueError('task IDs must be unique and versioned')
        seen.add(task_id)
        anchor = rng.choice(TRAIN_CONFIGS)
        distance = lambda c: sum(a != b for a, b in zip(anchor, c))
        alternatives = [c for c in TRAIN_CONFIGS if c != anchor]
        nearest = min(map(distance, alternatives))
        peer = rng.choice([c for c in alternatives if distance(c) == nearest])
        rows.append({'task_id': task_id, 'task_version': version,
                     'policy_version': policy_version, 'configs': [list(anchor), list(peer)]})
    if not rows:
        raise ValueError('empty task set')
    core = {'version': 'harness-scaling-k2-v1', 'seed': seed, 'assignments': rows}
    encoded = json.dumps(core, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return {**core, 'sha256': hashlib.sha256(encoded).hexdigest()}
