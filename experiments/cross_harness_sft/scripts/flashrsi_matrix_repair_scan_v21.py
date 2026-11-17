"""Read-only finite GPU1 view with v20/v21 tasks and strict native results."""
from flashrsi_matrix_repair_scan_v19 import build_source as prior_source
from flashrsi_harnessrisk_codex_context_v18 import replace_once

FINITE_PROTOCOLS = tuple('gpu1-finite-strict-matrix/v' + str(v) for v in (17, 18, 19, 20, 21))
NATIVE_REFERENCES = {'flashrsi_harnessrisk_pi_native_owner_reference_v6.py',
                     'flashrsi_harnessrisk_claude_native_owner_reference_v9.py'}


def inspect_scanned_run(run, expected, pid):
    """Use native proof for new profiles; retain historical legacy inspection."""
    manifest = read(run / 'run_manifest.json', {})
    native = (manifest.get('target_execution_protocol') in {p['profile'] for p in NATIVE_PROFILES.values()}
              or manifest.get('host_script') in {p['host'] for p in NATIVE_PROFILES.values()}
              or (run.parent.name in NATIVE_PROFILES and
                  read(run.parent.parent / 'repair_protocol.json', {}).get('protocol') in FINITE_PROTOCOLS[-2:]))
    return strict_inspect_run(run, expected, pid) if native else matrix.inspect_run(run, expected, pid)


def collect_queue_entries(roots):
    queues = []
    for root in roots:
        protocol = read(root / 'repair_protocol.json', {})
        if protocol.get('protocol') not in FINITE_PROTOCOLS:
            continue
        state = read(root / 'launch_state.json', {})
        retained = read(root / 'pidboard.json', {}).get('launcher')
        pid = state.get('launcher_pid')
        identity = process_identity(pid) if isinstance(pid, int) else None
        command = identity.get('command', []) if identity else []
        scope = None
        if '--out' in command and command.index('--out') + 1 < len(command):
            scope = command[command.index('--out') + 1]
        else:
            scope = next((arg[len('--out='):] for arg in command if arg.startswith('--out=')), None)
        launcher = protocol.get('queue_launcher')
        matches = bool(identity is not None and identity == retained and launcher and scope
            and Path(scope).resolve() == root.resolve()
            and any(Path(arg).name == Path(launcher).name for arg in command))
        queues.append({'path': str(root), 'protocol': protocol.get('protocol'), 'state': state,
            'tasks': read(root / 'remaining_tasks.json', {}),
            'launcher_identity_matches': matches, 'launcher_identity': identity if matches else None})
    return queues


def queued_job_entries(queues, key):
    entries = []
    for queue in queues:
        document = queue['tasks']
        tasks = document.get('tasks') if isinstance(document, dict) else None
        task = tasks.get(key) if isinstance(tasks, dict) else None
        if queue['launcher_identity_matches'] and isinstance(task, dict) and task.get('status') == 'queued':
            entries.append({'path': queue['path'], 'task': task, 'launcher_identity': queue['launcher_identity']})
    return entries


def current_completion_proof(queue, key, task, expected, active, inspected):
    """A retained complete row still needs its exact source/judge evidence."""
    try:
        harness, stage = key.split('/')
        if harness not in matrix.HARNESSES or stage not in ('initial', *matrix.ALGORITHMS):
            raise ValueError('unknown_current_task_identity')
        root = Path(queue['path'])
        run = root / harness / (harness + '_' + stage + '_full')
        if task.get('path') and Path(task['path']).resolve() != run.resolve():
            raise ValueError('current_task_evidence_path_mismatch')
        if stage == 'initial':
            contract = expected_initial_contract((root / 'case_manifest.json').read_bytes(), harness)
            good = initial_acceptance(run, expected, contract) and judge_acceptance(run / 'cases', expected)
            return {'path': str(run), 'status': 'complete' if good else 'needs_repair',
                    'proof': 'matched_H0_contract_and_exact_cached_judge'}
        cached = inspected.get(str(run.resolve()))
        return cached if cached is not None else inspect_scanned_run(run, expected, active.get(str(run.resolve())))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'status': 'needs_repair', 'error': str(exc), 'proof': 'current_completion_evidence_invalid'}


def with_current_queue_view(result, expected, active):
    """Keep historical precedence labelled; guard completion with owned tasks."""
    inspected = {}
    for job in result['jobs']:
        job['aggregation_mode'] = 'best_historical_verified_run'
        verified = [row for row in job['runs'] if row.get('status') == 'complete']
        best = verified[0] if verified else None
        job['best_historical_verified_run'] = ({'path': best.get('path'),
            'protocol': best.get('execution_protocol') or best.get('protocol') or 'unknown'} if best else None)
        for row in job['runs']:
            if row.get('path'):
                inspected[str(Path(row['path']).resolve())] = row
    current = []
    for queue in result['queues']:
        if not queue['launcher_identity_matches']:
            continue
        retained = queue['tasks']
        tasks = retained.get('tasks') if isinstance(retained, dict) else None
        if not isinstance(tasks, dict) or not tasks:
            tasks = {'queue/task_manifest': {'status': 'invalid_task_manifest', 'retained_manifest': retained}}
        for key, task in tasks.items():
            status = task.get('status', 'unknown_task_state') if isinstance(task, dict) else 'invalid_task_state'
            row = {'job_id': key, 'status': status, 'queue_path': queue['path'],
                'queue_protocol': queue['protocol'], 'launcher_identity': queue['launcher_identity'],
                'retained_task': task}
            if status == 'complete':
                proof = current_completion_proof(queue, key, task, expected, active, inspected)
                row['source_acceptance'] = proof
                if proof.get('status') != 'complete':
                    row['status'] = 'current_completion_not_verified'
            current.append(row)
    historical_complete = result['all_complete']
    current_complete = all(row['status'] == 'complete' for row in current)
    result.update(aggregation_mode='best_historical_verified_run', historical_all_complete=historical_complete,
        current_queue_jobs=current, current_queue_counts=dict(Counter(row['status'] for row in current)),
        current_queue_status=('complete' if current_complete else 'incomplete') if current else 'no_identity_matched_live_queue',
        all_complete=historical_complete and current_complete)
    return result


def build_source():
    source = prior_source()
    old = "('gpu1-finite-strict-matrix/v17', 'gpu1-finite-strict-matrix/v18', 'gpu1-finite-strict-matrix/v19')"
    if source.count(old) != 2:
        raise ValueError('reviewed_scanner_protocol_shape_changed')
    source = source.replace(old, 'FINITE_PROTOCOLS')
    source = replace_once(source, 'from flashrsi_matrix_gpu1_queue_v17 import read, process_identity',
        'from flashrsi_matrix_gpu1_queue_v21 import read, process_identity, NATIVE_PROFILES, '
        'expected_initial_contract, initial_acceptance, judge_acceptance, inspect_run as strict_inspect_run')
    names = "{'flashrsi_harnessrisk_algorithm_reference_v17.py', 'flashrsi_harnessrisk_algorithm_reference_v18.py', 'flashrsi_harnessrisk_algorithm_reference_v19.py'}"
    source = replace_once(source, names, '(' + names + ' | NATIVE_REFERENCES)')
    marker = "            env = dict(v.split(b'=', 1) for v in (path / 'environ').read_bytes().split(b'\\0') if b'=' in v)"
    source = replace_once(source, marker,
        "            if names.intersection(NATIVE_REFERENCES):\n"
        "                if identity.get('run_scope'):\n"
        "                    runs[identity['run_scope']] = int(path.name)\n"
        "                continue\n" + marker)
    source = replace_once(source, '    jobs = []\n', '    queues = collect_queue_entries(roots)\n    jobs = []\n')
    source = replace_once(source, 'runs.append(matrix.inspect_run(', 'runs.append(inspect_scanned_run(')
    source = replace_once(source, "            states = {row['status'] for row in runs}",
        "            queued = queued_job_entries(queues, harness + '/' + algorithm)\n"
        "            states = {row['status'] for row in runs}")
    source = replace_once(source, ", 'needs_repair' if runs else 'pending')",
                          ", 'queued' if queued else 'needs_repair' if runs else 'pending')")
    source = replace_once(source, "'status': status, 'runs': runs}",
                          "'status': status, 'runs': runs, 'queued_launches': queued}")
    old_queue_report = '''    queues = []
    for root in roots:
        protocol = read(root / 'repair_protocol.json', {})
        if protocol.get('protocol') in FINITE_PROTOCOLS:
            queues.append({'path': str(root), 'state': read(root / 'launch_state.json'), 'tasks': read(root / 'remaining_tasks.json')})
'''
    source = replace_once(source, old_queue_report, '')
    source = replace_once(source, 'v17/v18 registered; original strict acceptance',
                          'v17-v21 registered; native full proof and scoped live queue tasks')
    source = replace_once(source, "    return {'jobs': jobs,", "    result = {'jobs': jobs,")
    source = replace_once(source, "\nif __name__ == '__main__':",
        "\n    return with_current_queue_view(result, expected, active)\n\nif __name__ == '__main__':")
    return source


exec(compile(build_source(), __file__, 'exec'))
