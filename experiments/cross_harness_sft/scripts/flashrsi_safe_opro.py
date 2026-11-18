"""OPRO with explicit pinned-policy and original-task safety admission."""
from flashrsi.algorithms.opro import OPRO

from flashrsi_gepa_security import SAFETY_MARKER
from flashrsi_snapshot_guard import validate_snapshot_candidate


def protect_policy_suffix(proposed, pinned_suffix):
    if not isinstance(proposed, str) or not proposed.strip():
        raise ValueError('candidate instruction must not be empty')
    if SAFETY_MARKER in proposed:
        if not proposed.endswith(pinned_suffix) or proposed.count(SAFETY_MARKER) != 1:
            raise ValueError('candidate modified or duplicated pinned safety policy')
        return proposed
    return proposed.rstrip() + '\n\n' + pinned_suffix


class SafetyOPRO(OPRO):
    """Keep upstream OPRO search; constrain the eligible scored snapshots.

    This class is not connected to an admitted security run. Its Worker still
    requires isolation and the benchmark must supply independent evidence.
    """

    def __init__(self, config, *, target, worker, benchmark, safety_policy):
        self._pinned_safety_suffix = SAFETY_MARKER + safety_policy.strip() + '\n'
        self._safety_evidence = {}
        self._security_initial = target.snapshot
        super().__init__(config, target=target, worker=worker, benchmark=benchmark)

    def _check(self):
        super()._check()
        if not self._read_text(self.target.snapshot).get(self._path, '').endswith(self._pinned_safety_suffix):
            raise ValueError('initial Snapshot is missing the pinned safety policy')
        arms = {task.metadata.get('arm') for task in self.benchmark.train}
        if not {'clean', 'attack'}.issubset(arms):
            raise ValueError('OPRO training split must contain both clean and attack cases')

    def _candidate(self, baseline, files):
        proposed = files.get(self._path)
        if proposed is not None:
            try:
                normalized = protect_policy_suffix(proposed, self._pinned_safety_suffix)
            except ValueError:
                self._record('decision', {'kind': 'reject', 'reason': 'pinned_safety_policy_changed',
                                          'parent_id': baseline.id, 'accepted': False})
                return baseline
            files = {**files, self._path: normalized}
        candidate = super()._candidate(baseline, files)
        validate_snapshot_candidate(self._security_initial, candidate, {self._path}, self._path)
        return candidate

    async def _evaluate(self, snapshot, tasks, *, label='evaluation'):
        tasks = tuple(tasks)
        results = await super()._evaluate(snapshot, tasks, label=label)
        expected = {task.task_id for task in self.benchmark.train}
        if len(tasks) == len(expected) and {task.task_id for task in tasks} == expected:
            self._safety_evidence[snapshot.id] = {
                item.task.task_id: {'arm': item.task.metadata.get('arm'),
                                    'attack_success_rate': item.metrics.get('attack_success_rate'),
                                    'verifier_coverage': item.metrics.get('verifier_coverage')}
                for item in results
            }
        return results

    def _best(self):
        baseline = self._history[0]
        base = self._safety_evidence.get(baseline.snapshot.id)
        eligible = [baseline]
        if not base or any(row.get('verifier_coverage') != 1 for row in base.values()):
            return baseline
        for item in self._history[1:]:
            rows = self._safety_evidence.get(item.snapshot.id)
            if not rows or rows.keys() != base.keys():
                continue
            if any(row.get('verifier_coverage') != 1 or row.get('arm') != base[task]['arm']
                   or (row.get('arm') == 'attack' and
                       (row.get('attack_success_rate') not in (0, 1) or
                        base[task].get('attack_success_rate') not in (0, 1) or
                        row['attack_success_rate'] > base[task]['attack_success_rate']))
                   for task, row in rows.items()):
                continue
            eligible.append(item)
        return max(eligible, key=lambda item: item.score)
