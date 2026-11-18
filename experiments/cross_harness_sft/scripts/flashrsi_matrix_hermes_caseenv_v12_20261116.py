"""Run a genuinely matched Hermes initial control then GEPA after fixture repair."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from flashrsi_matrix_opencode_gpu1_v2_20261115 import initial_gate
from flashrsi_matrix_repair_loop import active_runs, inspect_run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_hermes_caseenv_v12_20261116'
OLD = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_hermes_gepa_contextfit_v11_20261115'


def main():
    # Only a single evaluator may use this endpoint. Never stop other clients here.
    conflicts = []
    for path, pid in active_runs().items():
        try:
            values = dict(x.split(b'=', 1) for x in (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0') if b'=' in x)
            if values.get(b'HARNESSRISK_MODEL_PORT', b'18761') == b'18761':
                conflicts.append({'pid': pid, 'run': path})
        except OSError:
            continue
    if conflicts:
        raise RuntimeError('endpoint has an existing evaluator; refusing duplicate launch')
    OUT.mkdir(exist_ok=False)
    lock = (OUT / 'launch.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    shutil.copy2(OLD / 'case_manifest.json', OUT / 'case_manifest.json')
    previous = json.loads((OLD / 'repair_protocol.json').read_text())
    names = set(previous['sources_sha256']) | {
        str(Path(__file__).relative_to(ROOT)),
        'scripts/flashrsi_harnessrisk_supervisor_hermes_workspace_v12.py',
        'scripts/flashrsi_harnessrisk_host_hermes_workspace_v12.py',
        'scripts/flashrsi_harnessrisk_opencode_environment_v9.py',
        'scripts/flashrsi_harnessrisk_opencode_mock_identity_v10.py',
        'scripts/flashrsi_harnessrisk_miniagent_initial.py',
        'scripts/flashrsi_matrix_opencode_gpu1_v2_20261115.py',
        'scripts/flashrsi_matrix_repair_loop.py',
        'scripts/flashrsi_harnessrisk_privacy.py', 'scripts/flashrsi_harnessrisk_sandbox.py',
        'scripts/flashrsi_harnessrisk_port_entry.py', 'scripts/judge_harnessrisk_shengsuanyun.py',
    }
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)}
    protocol = {**previous, 'at': datetime.datetime.now().astimezone().isoformat(),
        'protocol': 'private-v1-hermes-gepa-v12-contextfit16k-public-caseenv',
        'sources_sha256': hashes, 'predecessor': str(OLD), 'job_id': 'hermes/gepa',
        'root_cause': 'v11 tool env omitted mockctl public current-case IDs and fake fixture auth; observed strict routing rejection in real tool results',
        'changes': ['Keep v11 16K context, bounded 1024-token Worker and workspace copy',
                    'Add already-allowed runtime interpreter PATH plus validated public case IDs/fake fixture auth only',
                    'Run real full128 initial control before GEPA with the identical repaired target environment'],
        'pairing': 'Only v12 initial/GEPA target protocols are matched; old v11 episodes remain diagnostic, never merged',
        'queue': ['hermes_initial_full', 'hermes_gepa_full'],
        'initial_snapshot_file_sha256': None}
    (OUT / 'repair_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for name in hashes:
        dst = OUT / 'source_snapshot' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dst)
    env = {**os.environ, 'PYTHONPATH': 'src:scripts:vendor/FlashRSI', 'HR_HARNESS': 'hermes',
        'HARNESSRISK_MODEL_PORT': '18761', 'HR_STRICT_EVOLUTION': '1', 'HR_EVOLUTION_CASE_ATTEMPTS': '1',
        'HR_WORKER_MAX_TOKENS': '1024', 'HR_OUTPUT_ROOT': str(OUT),
        'HR_RUN_NAME': 'hermes_initial_full', 'HR_HOST_SCRIPT': 'flashrsi_harnessrisk_host_hermes_workspace_v12.py',
        'HR_GEPA_REFLECTION_PROTOCOL': 'hermes-v10-contextfit'}
    expected = {c['case_id'] for c in json.loads((OUT / 'case_manifest.json').read_text())['cases']}
    def event(stage, **data):
        row = {'at': datetime.datetime.now().astimezone().isoformat(), 'stage': stage, **data}
        with (OUT / 'queue_events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        (OUT / 'launch_state.json').write_text(json.dumps({'launcher_pid': os.getpid(), **row}) + '\n')
        print(json.dumps(row), flush=True)
    def start(stage, args):
        if not all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == value for n, value in hashes.items()):
            raise RuntimeError('frozen source drift; refusing protocol mixing')
        with (OUT / (stage + '.log')).open('x') as log:
            p = subprocess.Popen([sys.executable, '-u', *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        event(stage, event='running', pid=p.pid)
        return p
    first = start('hermes_initial_full', ['scripts/flashrsi_harnessrisk_miniagent_initial.py'])
    code = first.wait()
    if code or not initial_gate(OUT / 'hermes_initial_full', expected):
        event('hermes_initial_full', event='failed_full_gate', exit_code=code, next_action='retain errors and repair; GEPA remains queued, no partial judge')
        return code or 2
    judge = start('hermes_initial_judge', ['scripts/judge_harnessrisk_shengsuanyun.py', str(OUT / 'hermes_initial_full/cases'), '--deployment', 'openai/gpt-5.4-nano'])
    algorithm = start('hermes_gepa_full', ['scripts/flashrsi_harnessrisk_algorithm.py', '--algorithm', 'gepa'])
    code = algorithm.wait()
    accepted = inspect_run(OUT / 'hermes_gepa_full', expected, None)
    (OUT / 'acceptance_before_judge.json').write_text(json.dumps(accepted, indent=2))
    if not code and accepted['status'] == 'needs_judge':
        score = start('hermes_gepa_judge', ['scripts/judge_harnessrisk_shengsuanyun.py', str(OUT / 'hermes_gepa_full/selected_cases'), '--deployment', 'openai/gpt-5.4-nano'])
        score.wait()
    judge.wait()
    event('hermes_gepa_full', event='acceptance', **{k:v for k,v in inspect_run(OUT / 'hermes_gepa_full', expected, None).items() if k in ('status', 'valid', 'judge_valid', 'reasons')})
    subprocess.run([sys.executable, 'scripts/flashrsi_matrix_repair_loop.py', '--scan'], cwd=ROOT, env=env, check=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
