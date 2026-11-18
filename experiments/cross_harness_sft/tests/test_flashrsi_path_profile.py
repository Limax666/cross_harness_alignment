import pytest

from flashrsi_path_profile import selected_families


def test_default_retains_existing_qualified_set():
    assert selected_families('standard', 'workspace:user_task_3:clean', frozenset({'old'})) == frozenset({'old'})


def test_development_controls_only_enable_exact_calibration_cases():
    base = frozenset({'old'})
    assert selected_families('calibration-v1', 'workspace:user_task_3:injection_task_1', base) == base | {'workspace:user_task_3'}
    assert selected_families('calibration-v1', 'workspace:user_task_3:injection_task_2', base) == base
    assert selected_families('calibration-v1', 'workspace:user_task_17:clean', base) == base | {'workspace:user_task_17'}
    with pytest.raises(ValueError):
        selected_families('unknown', 'workspace:user_task_3:clean', base)
