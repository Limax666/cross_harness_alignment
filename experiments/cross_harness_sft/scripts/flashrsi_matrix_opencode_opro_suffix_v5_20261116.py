"""Repair failed OPRO with a matching full control after the active GPU1 queue.

This is an actual deferred full evaluation, not a diagnostic model run. Other
healthy v4 algorithms continue and their evidence is never copied into v5.
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
from flashrsi_matrix_opencode_gpu1_v2_20261115 import initial_gate
from flashrsi_matrix_repair_loop import inspect_run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_opro_suffix_v5_20261116'
PRIOR = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_gpu1_v4_20261115'
PRIOR_LAUNCHER = 3925329


def endpoint_clients():
    clients = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            if int(proc.name) == os.getpid():
                continue
            if int(proc.name) == PRIOR_LAUNCHER and any(b'flashrsi_matrix_opencode_gpu1_v4_20261115.py' in a for a in args):
                clients.append({'pid': int(proc.name), 'kind': 'prior_queue'})
                continue
            if not any(a.rsplit(b'/', 1)[-1] in (
                b'flashrsi_harnessrisk_algorithm.py', b'flashrsi_harnessrisk_miniagent_initial.py') for a in args):
                continue
            env = dict(x.split(b'=', 1) for x in (proc / 'environ').read_bytes().split(b'\0') if b'=' in x)
            if env.get(b'HARNESSRISK_MODEL_PORT') == b'18764':
                clients.append({'pid': int(proc.name), 'kind': 'target_evaluation'})
        except (OSError, ValueError):
            continue
    return clients


def main():
    OUT.mkdir(exist_ok=False)
    queue_lock = (OUT / 'queue.lock').open('x')
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = json.loads((PRIOR / 'repair_protocol.json').read_text())
    names = set(prior['sources_sha256']) | {
        str(Path(__file__).relative_to(ROOT)),
        'scripts/flashrsi_harnessrisk_host_opencode_session_v11.py',
        'scripts/flashrsi_harnessrisk_opencode_session_supervisor_v11.py',
        'scripts/flashrsi_harnessrisk_opencode_owner_anchor_v11.py',
        'scripts/flashrsi_matrix_opencode_gpu1_v4_20261115.py',
        'scripts/flashrsi_harnessrisk_host_opencode_session_v10.py',
        'scripts/flashrsi_harnessrisk_opencode_session_supervisor_v10.py',
        'scripts/flashrsi_matrix_repair_scan_v14.py',
    }
    hashes = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in sorted(names)}
    shutil.copy2(PRIOR / 'case_manifest.json', OUT / 'case_manifest.json')
    expected = {c['case_id'] for c in json.loads((OUT / 'case_manifest.json').read_text())['cases']}
    if len(expected) != 128:
        raise RuntimeError('trusted complete case set mismatch')
    protocol = {**prior, 'protocol': 'private-v1-opencode-session-v11-owner-suffix-gpu1-32k',
        'job_id': 'opencode/opro', 'queue_launcher': str(Path(__file__).relative_to(ROOT)),
        'predecessor': str(PRIOR), 'sources_sha256': hashes,
        'root_cause': 'v4 OPRO skill_015 request15: compaction lost users; old projection inserted current owner before trailing assistant historical transcript; response continued fake tool transcript to length4096 despite input4220',
        'change': 'Only missing-user requests: retain all native messages and append exact current public owner after history; existing-user requests, raw archive, permissions and budgets unchanged',
        'queue': ['opencode_initial_full', 'opencode_opro_full'],
        'same_protocol_initial_required': True,
        'waiting_policy': 'Do not interrupt or overlap original GPU1 initial/OPRO/GEPA/EvoPrompt/ACE queue; start after prior launcher and target clients exit',
        'not_applicable': ['skillopt', 'evo_harness']}
    (OUT / 'repair_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for name in hashes:
        target = OUT / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    def event(stage, **data):
        row = {'at': datetime.datetime.now().astimezone().isoformat(), 'stage': stage,
               'launcher_pid': os.getpid(), **data}
        with (OUT / 'queue_events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        (OUT / 'launch_state.json').write_text(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    while clients := endpoint_clients():
        event('waiting', status='waiting_endpoint_capacity', clients=clients)
        time.sleep(30)
    lane_lock = (OUT.parent / '.opencode_gpu1_deferred_repair.lock').open('a')
    fcntl.flock(lane_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if endpoint_clients():
        raise RuntimeError('GPU1 evaluation appeared during lane reservation')
    env = {**os.environ, 'PYTHONPATH': 'src:scripts:vendor/FlashRSI', 'HR_OUTPUT_ROOT': str(OUT),
        'HARNESSRISK_MODEL_PORT': '18764', 'HR_HARNESS': 'opencode',
        'HR_RUN_NAME': 'opencode_initial_full', 'HR_HOST_SCRIPT': 'flashrsi_harnessrisk_host_opencode_session_v11.py',
        'HR_STRICT_EVOLUTION': '1', 'HR_EVOLUTION_CASE_ATTEMPTS': '1'}
    def execute(stage, args):
        if not all(hashlib.sha256((ROOT / k).read_bytes()).hexdigest() == v for k, v in hashes.items()):
            raise RuntimeError('source drift; refusing mixed protocol')
        with (OUT / (stage + '.log')).open('x') as log:
            proc = subprocess.Popen([sys.executable, '-u', *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            event(stage, status='running', pid=proc.pid)
            code = proc.wait()
        event(stage, status='exited', exit_code=code)
        return code
    def judge(stage, cases):
        existing = cases / 'llm_judge_multi_harness/manifest.json'
        if existing.exists():
            rows = json.loads(existing.read_text())
            if len(rows) == 128 and {r.get('case_id') for r in rows if r.get('status') in ('judged', 'existing')} == expected:
                event(stage, status='judge_already_complete')
                return
        execute(stage + '_judge', ['scripts/judge_harnessrisk_shengsuanyun.py', str(cases), '--deployment', 'openai/gpt-5.4-nano'])
    code = execute('opencode_initial_full', ['scripts/flashrsi_harnessrisk_miniagent_initial.py'])
    if code or not initial_gate(OUT / 'opencode_initial_full', expected):
        event('opencode_initial_full', status='failed_full_gate', next_action='keep repair ticket, no ranking from incomplete reference')
        return code or 1
    judge('opencode_initial', OUT / 'opencode_initial_full/cases')
    code = execute('opencode_opro_full', ['scripts/flashrsi_harnessrisk_algorithm.py', '--algorithm', 'opro'])
    gate = inspect_run(OUT / 'opencode_opro_full', expected, None)
    event('opencode_opro_full', status='acceptance', acceptance=gate)
    if not code and gate['status'] == 'needs_judge':
        judge('opencode_opro', OUT / 'opencode_opro_full/selected_cases')
    subprocess.run([sys.executable, 'scripts/flashrsi_matrix_repair_scan_v14.py', '--scan'], cwd=ROOT, env=env, check=True)
    event('queue', status='stages_ended_not_necessarily_accepted')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
