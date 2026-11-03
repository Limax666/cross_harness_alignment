from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'vendor/verl'), str(ROOT / 'scripts')]

from verl.trainer.ppo.utils import Role
from rl_9b_resource_layout import separate_reference_layout, validate_device_assignments


def test_separate_reference_retains_same_worker_class_and_reserves_three_devices():
    worker = object()
    classes = {Role.ActorRolloutRef: worker, Role.Critic: object()}
    mapping = {Role.ActorRolloutRef: 'global_pool', Role.Critic: 'global_pool'}
    specs = separate_reference_layout(classes, mapping, actor_gpus=2, nodes=1)
    assert classes[Role.ActorRollout] is worker
    assert classes[Role.RefPolicy] is worker
    assert Role.ActorRolloutRef not in classes
    assert mapping[Role.ActorRollout] == 'global_pool'
    assert mapping[Role.RefPolicy] == 'reference_pool'
    assert specs == {'global_pool': [2], 'reference_pool': [1]}


def test_assignment_guard_requires_three_distinct_expected_devices():
    validate_device_assignments(['0', '2'], ['3'], ['0', '2', '3'])
    with pytest.raises(ValueError, match='disjoint'):
        validate_device_assignments(['0', '2'], ['2'], ['0', '2', '3'])
    with pytest.raises(ValueError, match='visible'):
        validate_device_assignments(['0', '2'], ['3'], ['0', '1', '2'])


def test_reference_layout_rejects_other_world_sizes():
    with pytest.raises(ValueError):
        separate_reference_layout({Role.ActorRolloutRef: object()}, {Role.ActorRolloutRef: 'global_pool'}, actor_gpus=3, nodes=1)
