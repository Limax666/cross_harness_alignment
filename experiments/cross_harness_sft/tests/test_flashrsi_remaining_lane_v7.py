"""Completion queues stay on authorized lanes and never fabricate controls."""
from pathlib import Path
import pytest


def test_lane_plans_preserve_protocol_and_declared_algorithm_order():
    from flashrsi_matrix_remaining_lane_v7_20261116 import lane_plan
    h = lane_plan('hermes')
    o = lane_plan('opencode')
    assert h['port'] == 18761 and o['port'] == 18764
    assert h['algorithms'] == ['opro', 'evoprompt', 'ace', 'skillopt', 'evo_harness']
    assert o['algorithms'] == ['evoprompt', 'ace']
    assert 'v14' in h['host'] and 'v11' in o['host']
    assert h['reference'].name == 'hermes_initial_full'
    assert o['reference'].name == 'opencode_initial_full'
    with pytest.raises(ValueError):
        lane_plan('other')


def test_missing_real_control_returns_false_without_creating_files(tmp_path):
    from flashrsi_matrix_remaining_lane_v7_20261116 import reference_ready
    path = tmp_path / 'missing-control'
    assert not reference_ready(path, {'case_1'})
    assert not path.exists()
