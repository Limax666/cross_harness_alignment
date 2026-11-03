"""Stop a read-only rollout diagnostic before any actor update."""
import json
from pathlib import Path

class RolloutOnlyComplete(RuntimeError):
    pass

def stop_after_rollout(enabled,run_dir,iteration,audit_file,groups,rollouts):
    if not enabled:return
    audit_file=Path(audit_file);run_dir=Path(run_dir)
    if iteration != 1 or groups != 24 or rollouts != 96 or not audit_file.is_file() or not audit_file.stat().st_size:
        raise ValueError('rollout-only diagnostic missing complete evidence')
    marker=run_dir/'rollout_only_complete.json'
    if marker.exists():raise FileExistsError(marker)
    payload={'iteration':iteration,'groups':groups,'rollouts':rollouts,'optimizer_updates':0,'audit_file':str(audit_file),'status':'rollout_only_complete'}
    tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\n');tmp.replace(marker)
    raise RolloutOnlyComplete('rollout-only diagnostic completed; no optimizer update performed')
