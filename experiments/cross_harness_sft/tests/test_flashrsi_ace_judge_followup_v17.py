"""Read-only actual retained-run admission checks; never call a judge/model."""
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / x) for x in ('scripts', 'src', 'vendor/FlashRSI')]


def test_actual_full_opro_is_eligible_but_incomplete_ace_and_wrong_case_set_are_not():
    from flashrsi_ace_judge_followup_v17 import eligible
    from flashrsi_matrix_gpu1_queue_v17 import REFERENCE, read
    expected = {r['case_id'] for r in read(REFERENCE.parent / 'case_manifest.json')['cases']}
    full = REFERENCE.parent / 'opencode_opro_full'
    assert eligible(full, expected)
    incomplete = full.parents[1] / 'flashrsi_harnessrisk_4b_private_opencode_v17_gpu1_20261117/opencode_ace_full'
    # No assertion about this actively running job: readiness must agree with
    # genuine final128 acceptance rather than its process or attempts count.
    if not (incomplete / 'result.json').exists():
        assert not eligible(incomplete, expected)
    assert not eligible(full, expected - {'action_041'})


def test_predecessor_exec_does_not_end_immutable_identity_wait(monkeypatch):
    import flashrsi_ace_judge_followup_v17 as follower
    monkeypatch.setattr(follower, 'process_identity', lambda pid: {
        'pid': pid, 'start_ticks': '123', 'command': ['exec-replacement']})
    assert follower.predecessor_alive({'pid': 4, 'start_ticks': '123', 'command': ['old']})
    assert not follower.predecessor_alive({'pid': 4, 'start_ticks': '124', 'command': ['old']})


def test_judge_finished_success_and_failure_are_not_lease_deferral():
    from flashrsi_ace_judge_followup_v17 import drain_judges
    class Boundary:
        def __init__(self, status):
            self.judges, self.judge, self.status = [], object(), status
            self.tasks = {'opencode/ace': {'status': 'source_valid_needs_judge'}}
        def check_cleanup(self):
            pass
        def schedule_judge(self):
            self.judge = None
            self.tasks['opencode/ace']['status'] = self.status
    for status in ('complete', 'source_valid_needs_judge'):
        queue = Boundary(status)
        assert drain_judges(queue, lambda: None, lambda _: None)
        assert queue.tasks['opencode/ace']['status'] == status
    queue = Boundary('source_valid_needs_judge')
    queue.judge, queue.judges = None, [('opencode/ace', Path('/unused'))]
    assert not drain_judges(queue, lambda: None, lambda _: None)


def test_final_verdict_preserves_nonzero_judge_exit_even_when_manifest_is_complete():
    from flashrsi_ace_judge_followup_v17 import terminal_status
    assert terminal_status({}, {'status': 'complete'}) == 'complete'  # Cached.
    assert terminal_status({'judge_returncode': 0}, {'status': 'complete'}) == 'complete'
    assert terminal_status({'judge_returncode': 1}, {'status': 'complete'}) == 'needs_judge_process_repair'
    assert terminal_status({'judge_returncode': 0}, {'status': 'needs_judge'}) == 'needs_judge'
