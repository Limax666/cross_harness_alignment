"""Scope calibration-only path controls to their exact verified cells."""

_CALIBRATION_CELLS = frozenset({
    'workspace:user_task_3:clean', 'workspace:user_task_3:injection_task_1',
    'workspace:user_task_17:clean', 'workspace:user_task_17:injection_task_1',
})


def selected_families(profile, task_id, baseline):
    if profile not in ('standard', 'calibration-v1'):
        raise ValueError('unknown path audit profile')
    if profile == 'calibration-v1' and task_id in _CALIBRATION_CELLS:
        return frozenset(baseline) | {task_id.rsplit(':', 1)[0]}
    return frozenset(baseline)
