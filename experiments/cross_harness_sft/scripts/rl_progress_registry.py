"""Explicit oracle versions, selected per run and frozen in checkpoint metadata."""
from importlib import import_module
_VERSIONS={'agentdojo-workspace-progress-v3':'rl_task_progress','agentdojo-workspace-progress-v4-candidate':'rl_task_progress_v4_candidate'}
DEFAULT='agentdojo-workspace-progress-v3'
def resolve(version):
    if not isinstance(version,str) or version not in _VERSIONS:raise ValueError('unregistered progress oracle version')
    module=import_module(_VERSIONS[version])
    if module.VERSION!=version:raise ValueError('progress oracle version mismatch')
    return module
