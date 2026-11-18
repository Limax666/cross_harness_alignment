"""Complete remaining real algorithms after existing authorized lane queues."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from flashrsi_matrix_hermes_native_prune_v14_20261116 import reference_is_valid
from flashrsi_matrix_repair_loop import inspect_run

ROOT = Path(__file__).resolve().parents[1]


def lane_plan(harness):
    if harness == 'hermes':
        prior = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_hermes_native_prune_v14_20261116'
        return {'port': 18761, 'prior': prior, 'reference': prior / 'hermes_initial_full',
            'out': ROOT / 'outputs/flashrsi_harnessrisk_4b_private_hermes_v14_remaining_20261116',
            'algorithms': ['opro', 'evoprompt', 'ace', 'skillopt', 'evo_harness'],
            'host': 'flashrsi_harnessrisk_host_hermes_workspace_v14.py', 'protocol': 'hermes-v14-native-prune',
            'parents': ['flashrsi_matrix_hermes_native_prune_v14_20261116.py']}
    if harness == 'opencode':
        prior = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_gepa_contextfit_v6_20261116'
        return {'port': 18764, 'prior': prior,
            'reference': ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_opro_suffix_v5_20261116/opencode_initial_full',
            'out': ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_v11_remaining_20261116',
            'algorithms': ['evoprompt', 'ace'], 'host': 'flashrsi_harnessrisk_host_opencode_session_v11.py',
            'protocol': 'opencode-v11-owner-suffix',
            'parents': ['flashrsi_matrix_opencode_gpu1_v4_20261115.py',
                        'flashrsi_matrix_opencode_opro_suffix_v5_20261116.py',
                        'flashrsi_matrix_opencode_gepa_contextfit_v6_20261116.py']}
    raise ValueError('only the two authorized harness lanes are supported')


def reference_ready(path, expected):
    try:
        return reference_is_valid(Path(path), expected)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def lane_clients(plan):
    clients = []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if int(p.name) == os.getpid() or p.stat().st_uid != os.getuid():
                continue
            args = (p / 'cmdline').read_bytes().split(b'\0')
            names = [a.rsplit(b'/', 1)[-1].decode(errors='replace') for a in args]
            if any(name in plan['parents'] for name in names):
                clients.append({'pid': int(p.name), 'kind': 'predecessor_queue'})
                continue
            if not any(name.startswith('flashrsi_harnessrisk_algorithm') or
                       name.startswith('flashrsi_harnessrisk_host_') or
                       name == 'flashrsi_harnessrisk_miniagent_initial.py' for name in names):
                continue
            env = dict(x.split(b'=', 1) for x in (p / 'environ').read_bytes().split(b'\0') if b'=' in x)
            if env.get(b'HARNESSRISK_MODEL_PORT') == str(plan['port']).encode():
                clients.append({'pid': int(p.name), 'kind': 'lane_target_client'})
        except (OSError, ValueError):
            continue
    return clients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harness', choices=['hermes', 'opencode'], required=True)
    harness = parser.parse_args().harness
    plan = lane_plan(harness)
    out = plan['out']
    out.mkdir(exist_ok=False)
    queue_lock = (out / 'queue.lock').open('x')
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = json.loads((plan['prior'] / 'repair_protocol.json').read_text())
    names = set(previous['sources_sha256']) | {
        str(Path(__file__).relative_to(ROOT)), 'scripts/flashrsi_harnessrisk_algorithm_reference_v7.py',
        'scripts/flashrsi_matrix_repair_scan_v16.py', 'scripts/flashrsi_matrix_repair_scan_v15.py',
        'scripts/flashrsi_matrix_hermes_native_prune_v14_20261116.py',
        'scripts/flashrsi_harnessrisk_hermes_context_profile_v13.py'}
    hashes = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in sorted(names)}
    shutil.copy2(plan['prior'] / 'case_manifest.json', out / 'case_manifest.json')
    expected = {c['case_id'] for c in json.loads((out / 'case_manifest.json').read_text())['cases']}
    if len(expected) != 128:
        raise RuntimeError('trusted full case set must contain128 unique cases')
    protocol = {**previous, 'protocol': plan['protocol'] + '-remaining-algorithms',
        'sources_sha256': hashes, 'queue_launcher': str(Path(__file__).relative_to(ROOT)),
        'job_ids': [harness + '/' + a for a in plan['algorithms']], 'queue': plan['algorithms'],
        'initial_reference': str(plan['reference']), 'port': plan['port'],
        'changes': 'Scheduling and real reference only; target host, budgets, algorithm configs and strict scoring unchanged',
        'pairing': 'Old target protocols remain separate; only same-protocol actual control is referenced',
        'waiting_policy': 'Wait predecessor queues and target clients; no duplicate model service or initial evaluation'}
    protocol.pop('job_id', None)
    (out / 'repair_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for name in hashes:
        dest = out / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    tasks = {a: {'status': 'queued'} for a in plan['algorithms']}
    def event(stage, **data):
        row = {'at': datetime.datetime.now().astimezone().isoformat(), 'launcher_pid': os.getpid(),
               'harness': harness, 'stage': stage, **data}
        with (out / 'queue_events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        (out / 'launch_state.json').write_text(json.dumps(row) + '\n')
        (out / 'remaining_tasks.json').write_text(json.dumps({'at': row['at'], 'tasks': tasks}, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    while clients := lane_clients(plan):
        event('waiting', status='waiting_endpoint_capacity', clients=clients)
        time.sleep(30)
    lane_lock_name = '.opencode_gpu1_deferred_repair.lock' if harness == 'opencode' else '.hermes_gpu0_completion.lock'
    lane_lock = (out.parent / lane_lock_name).open('a')
    fcntl.flock(lane_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if lane_clients(plan):
        raise RuntimeError('another target appeared while reserving lane')
    if not reference_ready(plan['reference'], expected):
        for a in tasks:
            tasks[a] = {'status': 'blocked_real_control', 'reference': str(plan['reference'])}
        event('reference', status='failed_reference_gate', next_action='repair actual control; never fabricate Snapshot or scores')
        return 1
    (out / 'initial_reference_verified.json').write_text(json.dumps({
        'path': str(plan['reference']), 'validated_cases': 128,
        'snapshot_sha256': hashlib.sha256((plan['reference'] / 'initial_snapshot.json').read_bytes()).hexdigest()}) + '\n')
    env = {**os.environ, 'PYTHONPATH': 'src:scripts:vendor/FlashRSI', 'HR_HARNESS': harness,
        'HARNESSRISK_MODEL_PORT': str(plan['port']), 'HR_OUTPUT_ROOT': str(out),
        'HR_INITIAL_REFERENCE': str(plan['reference']), 'HR_HOST_SCRIPT': plan['host'],
        'HR_TARGET_PROTOCOL': plan['protocol'], 'HR_STRICT_EVOLUTION': '1',
        'HR_EVOLUTION_CASE_ATTEMPTS': '1', 'HR_WORKER_MAX_TOKENS': '1024'}
    def execute(stage, args):
        if not all(hashlib.sha256((ROOT / k).read_bytes()).hexdigest() == v for k, v in hashes.items()):
            raise RuntimeError('source drift; refusing mixed protocol')
        with (out / (stage + '.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            event(stage, status='running', pid=child.pid)
            code = child.wait()
        event(stage, status='exited', exit_code=code)
        return code
    for algorithm in plan['algorithms']:
        stage = f'{harness}_{algorithm}_full'
        tasks[algorithm] = {'status': 'running'}
        code = execute(stage, ['scripts/flashrsi_harnessrisk_algorithm_reference_v7.py', '--algorithm', algorithm])
        gate = inspect_run(out / stage, expected, None)
        tasks[algorithm] = {'status': gate['status'], 'exit_code': code, 'acceptance': gate}
        if not code and gate['status'] == 'needs_judge':
            execute(stage + '_judge', ['scripts/judge_harnessrisk_shengsuanyun.py', str(out / stage / 'selected_cases'), '--deployment', 'openai/gpt-5.4-nano'])
            gate = inspect_run(out / stage, expected, None)
            tasks[algorithm] = {'status': gate['status'], 'acceptance': gate}
        event(stage, status='acceptance', acceptance=gate)
        # A failed algorithm stays in tasks, while independent following algorithms run.
    subprocess.run([sys.executable, 'scripts/flashrsi_matrix_repair_scan_v16.py', '--scan'], cwd=ROOT, env=env, check=True)
    event('queue', status='stages_ended_not_necessarily_accepted')
    return 0 if all(t['status'] == 'complete' for t in tasks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
