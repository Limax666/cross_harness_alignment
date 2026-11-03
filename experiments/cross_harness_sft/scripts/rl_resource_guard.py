"""Small, side-effect-free resource selection helpers for guarded RL launch."""
from itertools import combinations


def select_idle_pair(candidates, used_mib, gpu_uuids, active_uuids, *, max_used_mib=255):
    """Return the first ordered pair with no compute process and low VRAM use."""
    candidates=tuple(int(x) for x in candidates)
    if len(set(candidates)) != len(candidates):
        raise ValueError('GPU candidates must be distinct')
    if len(candidates) < 2:
        raise ValueError('at least two GPU candidates are required')
    idle=[]
    for device in candidates:
        if device not in used_mib or device not in gpu_uuids:
            raise ValueError(f'missing GPU observation for {device}')
        if int(used_mib[device]) <= max_used_mib and gpu_uuids[device] not in active_uuids:
            idle.append(device)
    return next(combinations(idle,2),None)
