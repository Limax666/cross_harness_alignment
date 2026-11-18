"""FlashRSI EvoPrompt with a pinned, non-editable policy suffix."""
from flashrsi.algorithms.evoprompt import EvoPrompt
from flashrsi_gepa_security import SAFETY_MARKER
from flashrsi_safe_opro import protect_policy_suffix
from flashrsi_snapshot_guard import validate_snapshot_candidate


class SafetyEvoPrompt(EvoPrompt):
    def __init__(self, config, *, target, worker, benchmark, safety_policy):
        self._pinned_safety_suffix=SAFETY_MARKER+safety_policy.strip()+'\n'
        self._security_initial=target.snapshot
        super().__init__(config,target=target,worker=worker,benchmark=benchmark)

    def _check(self):
        super()._check()
        if not self._read_text(self.target.snapshot).get(self._path,'').endswith(self._pinned_safety_suffix):
            raise ValueError('initial Snapshot is missing the pinned policy')

    def _candidate(self, baseline, files):
        if self._path in files:
            try:
                files={**files,self._path:protect_policy_suffix(files[self._path],self._pinned_safety_suffix)}
            except ValueError:
                return baseline
        candidate=super()._candidate(baseline,files)
        validate_snapshot_candidate(self._security_initial,candidate,{self._path},self._path)
        return candidate
