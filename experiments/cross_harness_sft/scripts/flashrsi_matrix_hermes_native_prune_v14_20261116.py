"""Real complete GEPA rerun, then a matched v14 initial control; no model smoke run."""
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
from flashrsi_matrix_repair_loop import inspect_run
from flashrsi_matrix_repair_scan_v14 import active_runs
from flashrsi_harnessrisk_hermes_context_profile_v13 import native_context_profile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_hermes_native_prune_v14_20261116'
PRIOR = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_hermes_caseenv_v12_20261116'


def reference_is_valid(reference, expected):
    summary = json.loads((reference / 'completed.json').read_text())
    sid = json.loads((reference / 'initial_snapshot.json').read_text())['id']
    if summary.get('completed_scored') != 128 or summary.get('snapshot_id') != sid:
        return False
    cases = reference / 'cases'
    actual = {p.name for p in cases.iterdir() if p.is_dir() and (p / 'harness_result.json').exists()}
    if actual != expected:
        return False
    for cid in expected:
        p = cases / cid
        result = json.loads((p / 'harness_result.json').read_text())
        audit = json.loads((p / 'privacy_audit.json').read_text())
        mock = json.loads((p / 'mock_client_environment_audit.json').read_text())
        rule = json.loads((p / 'official_rule_evaluation.json').read_text())['runs']
        if result.get('status') != 'completed' or result.get('process_exit_code') != 0 or result.get('snapshot_id') != sid:
            return False
        if audit.get('protocol') != 'harnessrisk-private-case-v1' or mock.get('case_id') != cid or not mock.get('only_adapter_generated_fake_credentials'):
            return False
        if len(rule) != 1 or rule[0].get('case_id') != cid or rule[0].get('status') != 'completed':
            return False
    return True


def main():
    for path, pid in active_runs().items():
        try:
            env = dict(x.split(b'=', 1) for x in (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0') if b'=' in x)
            if env.get(b'HARNESSRISK_MODEL_PORT', b'18761') == b'18761':
                raise RuntimeError('GPU0 already has an evaluator; refusing duplicate')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    OUT.mkdir(exist_ok=False)
    lock = (OUT / 'launch.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    shutil.copy2(PRIOR / 'case_manifest.json', OUT / 'case_manifest.json')
    previous = json.loads((PRIOR / 'repair_protocol.json').read_text())
    names = set(previous['sources_sha256']) | {
        str(Path(__file__).relative_to(ROOT)),
        'scripts/flashrsi_harnessrisk_host_hermes_workspace_v14.py',
        'scripts/flashrsi_harnessrisk_supervisor_hermes_workspace_v14.py',
        'scripts/flashrsi_harnessrisk_hermes_context_profile_v13.py',
        'scripts/flashrsi_harnessrisk_algorithm_hermes_v14.py',
        'scripts/flashrsi_matrix_hermes_caseenv_v12_20261116.py',
        'scripts/flashrsi_harnessrisk_host_hermes_workspace_v13.py',
        'scripts/flashrsi_harnessrisk_supervisor_hermes_workspace_v13.py',
        'scripts/flashrsi_harnessrisk_algorithm_hermes_v13.py',
        'scripts/flashrsi_matrix_repair_scan_v13.py',
        'scripts/flashrsi_matrix_repair_scan_v14.py',
    }
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)}
    reference = PRIOR / 'hermes_initial_full'
    manifest = json.loads((OUT / 'case_manifest.json').read_text())
    expected = {c['case_id'] for c in manifest['cases']}
    if not reference_is_valid(reference, expected):
        raise RuntimeError('prior starting Snapshot lacks full identity/privacy/rule evidence')
    data = ROOT.parents[1] / 'HarnessRisk/data/HarnessRisk'
    for row in manifest['cases']:
        if hashlib.sha256((data / (row['case_id'] + '.json')).read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('trusted dataset case changed')
    protocol = {**previous, 'at': datetime.datetime.now().astimezone().isoformat(),
        'protocol': 'private-v1-hermes-gepa-v14-native-proactive-prune',
        'predecessor': str(PRIOR), 'sources_sha256': hashes,
        'wrapper_repair': 'v13 audit expression lacked mock_client_environment in its exec namespace; no real model calls, all diagnostics preserved',
        'root_cause': 'skill_074 candidate input16463 exceeded16384 after two native would_grow rejections; pruning disabled and default20 protected tail blocks short-turn reclamation',
        'changes': ['Enable the existing native deterministic proactive tool-result reclamation with a bounded recent tail',
                    'Context16K, target4096, worker1024,12 turns, private/sandbox and reject_unscorable gates unchanged'],
        'native_compression': native_context_profile(),
        'initial_reference': {'path': str(reference), 'snapshot_sha256': hashlib.sha256((reference / 'initial_snapshot.json').read_bytes()).hexdigest(),
                              'use': 'Starting Snapshot only; old v12 metrics are not a matched v14 control'},
        'queue': ['hermes_gepa_full', 'hermes_initial_full'],
        'pairing': 'Must finish new v14 initial control before matched safety/utility gain claims; no old scores/evidence merged',
        'full_case_count': 128}
    (OUT / 'repair_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for name in hashes:
        dst = OUT / 'source_snapshot' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dst)
    env = {**os.environ, 'PYTHONPATH': 'src:scripts:vendor/FlashRSI', 'HR_HARNESS': 'hermes',
        'HARNESSRISK_MODEL_PORT': '18761', 'HR_OUTPUT_ROOT': str(OUT),
        'HR_HOST_SCRIPT': 'flashrsi_harnessrisk_host_hermes_workspace_v14.py',
        'HR_INITIAL_REFERENCE': str(reference), 'HR_RUN_NAME': 'hermes_initial_full',
        'HR_STRICT_EVOLUTION': '1', 'HR_EVOLUTION_CASE_ATTEMPTS': '1',
        'HR_WORKER_MAX_TOKENS': '1024', 'HR_GEPA_REFLECTION_PROTOCOL': 'hermes-v10-contextfit'}
    def event(stage, **data):
        row = {'at': datetime.datetime.now().astimezone().isoformat(), 'stage': stage, **data}
        with (OUT / 'queue_events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        (OUT / 'launch_state.json').write_text(json.dumps({'launcher_pid': os.getpid(), **row}) + '\n')
        print(json.dumps(row), flush=True)
    def start(stage, args):
        if not all(hashlib.sha256((ROOT / k).read_bytes()).hexdigest() == value for k, value in hashes.items()):
            raise RuntimeError('source drift; refusing protocol mixing')
        with (OUT / (stage + '.log')).open('x') as log:
            p = subprocess.Popen([sys.executable, '-u', *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        event(stage, event='running', pid=p.pid)
        return p
    judges = []
    algorithm = start('hermes_gepa_full', ['scripts/flashrsi_harnessrisk_algorithm_hermes_v14.py', '--algorithm', 'gepa'])
    code = algorithm.wait()
    accepted = inspect_run(OUT / 'hermes_gepa_full', expected, None)
    (OUT / 'acceptance_before_judge.json').write_text(json.dumps(accepted, indent=2))
    event('hermes_gepa_full', event='model_stage_ended', exit_code=code, acceptance=accepted['status'])
    if not code and accepted['status'] == 'needs_judge':
        judges.append(start('hermes_gepa_judge', ['scripts/judge_harnessrisk_shengsuanyun.py', str(OUT / 'hermes_gepa_full/selected_cases'), '--deployment', 'openai/gpt-5.4-nano']))
    control = start('hermes_initial_full', ['scripts/flashrsi_harnessrisk_miniagent_initial.py'])
    control_code = control.wait()
    if not control_code and initial_gate(OUT / 'hermes_initial_full', expected):
        judges.append(start('hermes_initial_judge', ['scripts/judge_harnessrisk_shengsuanyun.py', str(OUT / 'hermes_initial_full/cases'), '--deployment', 'openai/gpt-5.4-nano']))
    else:
        event('hermes_initial_full', event='failed_full_gate', exit_code=control_code)
    for judge in judges:
        judge.wait()
    subprocess.run([sys.executable, 'scripts/flashrsi_matrix_repair_scan_v14.py', '--scan'], cwd=ROOT, env=env, check=True)
    event('queue', event='stages_ended_not_necessarily_accepted')
    return code or control_code


if __name__ == '__main__':
    raise SystemExit(main())
