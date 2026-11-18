"""Deferred complete GEPA repair, reusing only the matching real v5 initial.

Wait for both original and v5 repair queues. Do not duplicate their control or
score artifacts, interrupt their models, or rank incomplete trajectories.
"""
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
from flashrsi_matrix_opencode_opro_suffix_v5_20261116 import endpoint_clients as previous_clients
from flashrsi_matrix_hermes_native_prune_v14_20261116 import reference_is_valid
from flashrsi_matrix_repair_loop import inspect_run

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_opro_suffix_v5_20261116'
OUT = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_gepa_contextfit_v6_20261116'
REFERENCE = PRIOR / 'opencode_initial_full'


def clients():
    result = previous_clients()
    prior_state = json.loads((PRIOR / 'launch_state.json').read_text())
    pid = prior_state['launcher_pid']
    try:
        args = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
        if any(a.rsplit(b'/', 1)[-1] == b'flashrsi_matrix_opencode_opro_suffix_v5_20261116.py' for a in args):
            result.append({'pid': pid, 'kind': 'prior_v5_repair_queue'})
    except OSError:
        pass
    return result


def main():
    OUT.mkdir(exist_ok=False)
    lock = (OUT / 'queue.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = json.loads((PRIOR / 'repair_protocol.json').read_text())
    names = set(prior['sources_sha256']) | {
        str(Path(__file__).relative_to(ROOT)),
        'scripts/flashrsi_harnessrisk_algorithm_opencode_gepa_v6.py',
        'scripts/flashrsi_opencode_gepa_contextfit_v6.py',
        'scripts/flashrsi_matrix_repair_scan_v15.py',
        'scripts/flashrsi_matrix_hermes_native_prune_v14_20261116.py',
        'scripts/flashrsi_harnessrisk_hermes_context_profile_v13.py',
    }
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(names)}
    shutil.copy2(PRIOR / 'case_manifest.json', OUT / 'case_manifest.json')
    expected = {c['case_id'] for c in json.loads((OUT / 'case_manifest.json').read_text())['cases']}
    if len(expected) != 128:
        raise RuntimeError('trusted full case set mismatch')
    protocol = {**prior, 'protocol': 'private-v1-opencode-v11-target-gepa-v6-contextfit-worker',
        'job_id': 'opencode/gepa', 'queue_launcher': str(Path(__file__).relative_to(ROOT)),
        'predecessor': str(PRIOR), 'sources_sha256': hashes,
        'root_cause': 'v4 GEPA worker29 input9707/output1024 ended length while writing a long task-specific manual',
        'changes': ['Bounded data-only feedback retaining all minibatch rows, metrics, feedback and full trajectory hashes',
                    'Require one complete reusable instruction at most300 words; worker budget1024 unchanged',
                    'Same v11 target/runtime as v5; reference its actual completed128 control without copying scores'],
        'worker_reflection_protocol': 'opencode-v6-contextfit',
        'initial_reference': str(REFERENCE), 'queue': ['opencode_gepa_full'],
        'waiting_policy': 'Wait original GPU1 queue and v5 initial/OPRO queue; no duplicate inference or baseline',
        'not_applicable': ['skillopt', 'evo_harness']}
    (OUT / 'repair_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for name in hashes:
        dest = OUT / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    def event(stage, **data):
        row = {'at': datetime.datetime.now().astimezone().isoformat(), 'launcher_pid': os.getpid(), 'stage': stage, **data}
        with (OUT / 'queue_events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        (OUT / 'launch_state.json').write_text(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    while busy := clients():
        event('waiting', status='waiting_endpoint_capacity', clients=busy)
        time.sleep(30)
    lane_lock = (OUT.parent / '.opencode_gpu1_deferred_repair.lock').open('a')
    fcntl.flock(lane_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if clients():
        raise RuntimeError('GPU1 client appeared during lane reservation')
    if not reference_is_valid(REFERENCE, expected):
        event('initial_reference', status='failed_reference_gate', next_action='retain ticket and repair v5 control; never fabricate Snapshot or rank incomplete control')
        return 1
    (OUT / 'initial_reference_verified.json').write_text(json.dumps({
        'path': str(REFERENCE), 'snapshot_sha256': hashlib.sha256((REFERENCE / 'initial_snapshot.json').read_bytes()).hexdigest(),
        'completed_identity_privacy_mock_official_cases': 128}) + '\n')
    env = {**os.environ, 'PYTHONPATH': 'src:scripts:vendor/FlashRSI', 'HR_HARNESS': 'opencode',
        'HARNESSRISK_MODEL_PORT': '18764', 'HR_OUTPUT_ROOT': str(OUT), 'HR_INITIAL_REFERENCE': str(REFERENCE),
        'HR_HOST_SCRIPT': 'flashrsi_harnessrisk_host_opencode_session_v11.py', 'HR_STRICT_EVOLUTION': '1',
        'HR_EVOLUTION_CASE_ATTEMPTS': '1', 'HR_WORKER_MAX_TOKENS': '1024',
        'HR_GEPA_REFLECTION_PROTOCOL': 'opencode-v6-contextfit'}
    def execute(stage, args):
        if not all(hashlib.sha256((ROOT / k).read_bytes()).hexdigest() == v for k, v in hashes.items()):
            raise RuntimeError('frozen source drift; refusing mixed protocol')
        with (OUT / (stage + '.log')).open('x') as log:
            p = subprocess.Popen([sys.executable, '-u', *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            event(stage, status='running', pid=p.pid)
            code = p.wait()
        event(stage, status='exited', exit_code=code)
        return code
    code = execute('opencode_gepa_full', ['scripts/flashrsi_harnessrisk_algorithm_opencode_gepa_v6.py', '--algorithm', 'gepa'])
    gate = inspect_run(OUT / 'opencode_gepa_full', expected, None)
    event('opencode_gepa_full', status='acceptance', acceptance=gate)
    if not code and gate['status'] == 'needs_judge':
        execute('opencode_gepa_judge', ['scripts/judge_harnessrisk_shengsuanyun.py', str(OUT / 'opencode_gepa_full/selected_cases'), '--deployment', 'openai/gpt-5.4-nano'])
    subprocess.run([sys.executable, 'scripts/flashrsi_matrix_repair_scan_v15.py', '--scan'], cwd=ROOT, env=env, check=True)
    event('queue', status='stages_ended_not_necessarily_accepted')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
