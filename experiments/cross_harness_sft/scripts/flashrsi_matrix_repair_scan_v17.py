"""Read-only process/matrix view including v17/v18 and finite GPU1 stages."""
import argparse
from collections import Counter
import json
from pathlib import Path
import flashrsi_matrix_repair_loop as matrix
from flashrsi_matrix_repair_scan_v16 import active_runs as prior_active_runs
from flashrsi_matrix_gpu1_queue_v17 import read, process_identity


def active_runs():
    runs = prior_active_runs()
    for path in Path('/proc').glob('[0-9]*'):
        try:
            identity = process_identity(int(path.name))
            if not identity:
                continue
            args = identity['command']
            names = {Path(arg).name for arg in args}
            if not names.intersection({'flashrsi_harnessrisk_algorithm_reference_v17.py', 'flashrsi_harnessrisk_algorithm_reference_v18.py', 'flashrsi_harnessrisk_algorithm_reference_v19.py'}):
                continue
            env = dict(v.split(b'=', 1) for v in (path / 'environ').read_bytes().split(b'\0') if b'=' in v)
            harness = env[b'HR_HARNESS'].decode()
            algorithm = args[args.index('--algorithm') + 1]
            runs[str((Path(env[b'HR_OUTPUT_ROOT'].decode()) / f'{harness}_{algorithm}_full').resolve())] = int(path.name)
        except (OSError, KeyError, IndexError, ValueError):
            continue
    return runs


def scan():
    expected = {row['case_id'] for row in read(matrix.OUTPUTS / 'flashrsi_harnessrisk_4b_private_v1_20261113/case_manifest.json')['cases']}
    active = active_runs()
    roots = [path for path in matrix.OUTPUTS.iterdir() if path.is_dir() and path.name.startswith('flashrsi_')]
    # Exactly one level inside our versioned queue roots; never recursively scan
    # copied references, snapshots, caches or huge evidence graphs.
    for root in list(roots):
        if read(root / 'repair_protocol.json', {}).get('protocol') == 'gpu1-finite-strict-matrix/v17':
            roots.extend(path for path in root.iterdir() if path.name in ('opencode', 'openclaw', 'codex', 'claude_code', 'pi') and path.is_dir())
    jobs = []
    for harness in matrix.HARNESSES:
        for algorithm in matrix.ALGORITHMS:
            if harness == 'opencode' and algorithm in ('skillopt', 'evo_harness'):
                jobs.append({'job_id': harness + '/' + algorithm, 'status': 'not_applicable', 'runs': []})
                continue
            runs = []
            for base in roots:
                run = base / f'{harness}_{algorithm}_full'
                if run.is_dir():
                    runs.append(matrix.inspect_run(run, expected, active.get(str(run.resolve()))))
            states = {row['status'] for row in runs}
            status = next((value for value in ('complete', 'running', 'suspected_stall', 'needs_judge') if value in states), 'needs_repair' if runs else 'pending')
            jobs.append({'job_id': harness + '/' + algorithm, 'status': status, 'runs': runs})
    queues = []
    for root in roots:
        protocol = read(root / 'repair_protocol.json', {})
        if protocol.get('protocol') == 'gpu1-finite-strict-matrix/v17':
            queues.append({'path': str(root), 'state': read(root / 'launch_state.json'), 'tasks': read(root / 'remaining_tasks.json')})
    return {'jobs': jobs, 'counts': dict(Counter(row['status'] for row in jobs)), 'queues': queues,
            'all_complete': all(row['status'] in ('complete', 'not_applicable') for row in jobs),
            'read_only': True, 'scanner': 'v17/v18 registered; original strict acceptance'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', action='store_true')
    parser.parse_args()
    print(json.dumps(scan()))
