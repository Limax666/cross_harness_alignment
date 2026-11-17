"""Finite cached judge follow-up for the existing real OpenCode ACE stage.

No model stage/service/cron is created. Exact predecessor identities are waited
out; actual strict final128 acceptance is required before any judge dispatch.
"""
import argparse
import hashlib
import os
from pathlib import Path
import time

from flashrsi_matrix_gpu1_queue_v17 import Queue, process_identity, read, save, sha
from flashrsi_matrix_repair_loop import inspect_run


def eligible(run, expected):
    return len(expected) == 128 and inspect_run(Path(run), expected, None)['status'] in ('complete', 'needs_judge')


def run_followup(run, out, predecessors):
    run, out = Path(run).resolve(), Path(out).absolute()
    if out.exists():
        raise ValueError('followup_output_exists_preserved')
    manifest = run.parent / 'case_manifest.json'
    raw = manifest.read_bytes()
    expected = {r['case_id'] for r in read(manifest)['cases']}
    if len(expected) != 128:
        raise ValueError('exact_128_case_manifest_required')
    out.mkdir(mode=0o700)
    (out / 'case_manifest.json').write_bytes(raw)
    save(out / 'protocol.json', {'kind': 'one_finite_ace_cached_judge_followup/v17',
        'run': str(run), 'case_manifest_sha256': hashlib.sha256(raw).hexdigest(),
        'predecessors': predecessors, 'judge_overwrite': False, 'model_dispatch': False})
    queue = Queue(out)
    key = 'opencode/ace'
    queue.tasks = {key: {'status': 'waiting_actual_final128', 'path': str(run)}}
    queue.expected = expected
    queue.predecessor_scopes = [run]
    while any(process_identity(p['pid']) == p for p in predecessors):
        queue.event(key, 'waiting_actual_stage_exit')
        time.sleep(15)
    queue.check_cleanup()
    if sha(manifest) != hashlib.sha256(raw).hexdigest():
        raise ValueError('case_manifest_drift')
    if not eligible(run, expected):
        queue.tasks[key].update(status='requires_repair', acceptance=inspect_run(run, expected, None))
        queue.event(key, 'source_stage_invalid_no_judge_called')
        return 1
    binding = {str(path.relative_to(run)): sha(path) for path in
               (run / 'result.json', run / 'run_manifest.json', run / 'snapshots/selected.json')}
    save(out / 'accepted_source_binding.json', binding)
    queue.judges = [(key, run / 'selected_cases')]
    while queue.judges or queue.judge is not None:
        if any(sha(run / name) != value for name, value in binding.items()):
            raise ValueError('accepted_source_binding_changed')
        queue.check_cleanup()
        before = len(queue.judges)
        queue.schedule_judge()
        if queue.judge is not None:
            time.sleep(5)
        elif len(queue.judges) == before:
            queue.tasks[key]['judge_status'] = 'needs_judge_lease_or_billing'
            queue.event(key, 'cached_judge_deferred')
            return 1
    accepted = inspect_run(run, expected, None)
    queue.tasks[key].update(status=accepted['status'], acceptance=accepted)
    queue.event(key, accepted['status'])
    return 0 if accepted['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--predecessor-pid', type=int, action='append', default=[])
    args = parser.parse_args()
    predecessors = [identity for pid in args.predecessor_pid if (identity := process_identity(pid)) is not None]
    raise SystemExit(run_followup(args.run, args.out, predecessors))
