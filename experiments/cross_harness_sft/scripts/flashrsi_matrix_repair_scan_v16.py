"""Register reference-only entries and multi-algorithm waiting queues."""
import argparse
import json
from pathlib import Path
import flashrsi_matrix_repair_loop as matrix
from flashrsi_matrix_repair_scan_v15 import active_runs as previous_active_runs

previous_queued_launches = matrix.queued_launches


def active_runs():
    runs = previous_active_runs()
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if not any(a.rsplit(b'/', 1)[-1] == b'flashrsi_harnessrisk_algorithm_reference_v7.py' for a in args):
                continue
            env = dict(x.split(b'=', 1) for x in (p / 'environ').read_bytes().split(b'\0') if b'=' in x)
            h = env[b'HR_HARNESS'].decode()
            a = args[args.index(b'--algorithm') + 1].decode()
            runs[str((Path(env[b'HR_OUTPUT_ROOT'].decode()) / f'{h}_{a}_full').resolve())] = int(p.name)
        except (OSError, ValueError, KeyError, IndexError):
            continue
    return runs


def queued_launches(roots):
    queues = previous_queued_launches(roots)
    for root in roots:
        protocol = matrix.read(root / 'repair_protocol.json', {})
        state = matrix.read(root / 'launch_state.json', {})
        jobs = protocol.get('job_ids', [])
        if not jobs or state.get('status') not in ('waiting_endpoint_capacity', 'running', 'exited', 'acceptance'):
            continue
        if state.get('status') != 'waiting_endpoint_capacity':
            tasks = matrix.read(root / 'remaining_tasks.json', {}).get('tasks', {})
            jobs = [job for job in jobs if tasks.get(job.split('/', 1)[1], {}).get('status') == 'queued']
        try:
            args = (Path('/proc') / str(state['launcher_pid']) / 'cmdline').read_bytes().split(b'\0')
            script = protocol['queue_launcher']
            if not any(a.decode(errors='replace') in (script, str(matrix.ROOT / script)) for a in args):
                continue
        except (OSError, KeyError):
            continue
        for job in jobs:
            queues.setdefault(job, []).append({'path': str(root), 'launcher_pid': state['launcher_pid'],
                'pid': None, 'status': 'waiting_endpoint_capacity' if state.get('status') == 'waiting_endpoint_capacity' else 'waiting_previous_stage', 'protocol': protocol.get('protocol'),
                'last_update': state.get('at'), 'clients': state.get('clients', [])})
    return queues


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', action='store_true')
    parser.parse_args()
    matrix.active_runs = active_runs
    matrix.queued_launches = queued_launches
    data = matrix.scan()
    print(json.dumps({'counts': data['counts'], 'all_complete': data['all_complete'],
                     'scanner': 'original acceptance plus v16 entry and multi-job queue registry'}))
