"""Place ACE's playbook before the immutable safety policy."""
from flashrsi.algorithms.ace import ACE, _MARKER
from flashrsi_gepa_security import SAFETY_MARKER
from flashrsi_snapshot_guard import validate_snapshot_candidate


class SafetyACE(ACE):
    def __init__(self, config, *, target, worker, benchmark, safety_policy):
        self._pinned_safety_suffix=SAFETY_MARKER+safety_policy.strip()+'\n'
        self._security_initial=target.snapshot
        super().__init__(config,target=target,worker=worker,benchmark=benchmark)

    def _check(self):
        super()._check()
        if not self._read_text(self.target.snapshot).get(self._path,'').endswith(self._pinned_safety_suffix):
            raise ValueError('initial Snapshot is missing the pinned policy')

    def _candidate(self, baseline, files):
        text=files.get(self._path)
        if text is not None:
            anchor=self._pinned_safety_suffix+_MARKER
            if text.count(anchor)!=1 or text.count(SAFETY_MARKER)!=1:
                raise ValueError('ACE candidate changed the pinned policy')
            prefix,playbook=text.split(anchor,1)
            files={**files,self._path:prefix+_MARKER+playbook.rstrip()+'\n\n'+self._pinned_safety_suffix}
        candidate=super()._candidate(baseline,files)
        validate_snapshot_candidate(self._security_initial,candidate,{self._path},self._path)
        return candidate
