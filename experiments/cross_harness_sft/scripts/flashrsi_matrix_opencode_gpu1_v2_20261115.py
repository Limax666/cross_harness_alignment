"""Run independent v8 full initial/OPRO/GEPA/EvoPrompt/ACE on GPU1, with gated judge."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_gpu1_v2_20261115'
OLD = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_gpu1_v1_20261115'
PYTHON = ROOT / '.venv-rl-pilot/bin/python'
ALGORITHMS = ['opro', 'gepa', 'evoprompt', 'ace']


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def initial_gate(run, expected):
    summary = read(run / 'completed.json', {})
    sid = read(run / 'initial_snapshot.json', {}).get('id')
    if summary.get('completed_scored') != 128 or not sid:
        return False
    cases = run / 'cases'
    if {p.name for p in cases.iterdir() if p.is_dir()} != expected:
        return False
    for cid in expected:
        case = cases / cid
        h = read(case / 'harness_result.json', {})
        audit = read(case / 'privacy_audit.json', {})
        official = read(case / 'official_rule_evaluation.json', {}).get('runs', [])
        if h.get('status') != 'completed' or h.get('snapshot_id') != sid:
            return False
        if audit.get('protocol') != 'harnessrisk-private-case-v1':
            return False
        if len(official) != 1 or official[0].get('case_id') != cid or official[0].get('status') != 'completed':
            return False
    return True


def main():
    OUT.mkdir(exist_ok=False)
    lock = (OUT / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = read(OLD / 'repair_protocol.json', {})
    names = set(previous['sources_sha256']) | {
        str(Path(__file__).relative_to(ROOT)),
        'scripts/flashrsi_harnessrisk_host_opencode_session_v8.py',
        'scripts/flashrsi_harnessrisk_opencode_session_supervisor_v8.py',
        'scripts/flashrsi_harnessrisk_opencode_owner_anchor_v8.py',
        'scripts/flashrsi_matrix_repair_loop.py',
    }
    hashes = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in sorted(names)}
    shutil.copy2(OLD / 'case_manifest.json', OUT / 'case_manifest.json')
    protocol = {**previous, 'protocol': 'private-v1-opencode-native-session-v8-user-anchor-gpu1-32k',
                'sources_sha256': hashes, 'predecessor': str(OLD),
                'root_cause': 'v7 deterministic compaction can remove all user roles; Qwen chat template rejects those requests before inference',
                'change': 'Separate audited provider projection restores exact current public owner as user only when native request has no user; raw SQLite/native request and actual forwarded request both retained',
                'context_limit_unchanged': 32768, 'target_max_tokens_unchanged': 4096,
                'same_protocol_initial_required': True, 'judge_gate': 'same-run exact 128 identity/privacy/official/selected; no partial judge',
                'not_applicable': ['skillopt', 'evo_harness']}
    (OUT / 'repair_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    frozen = OUT / 'source_snapshot'
    for name in hashes:
        dest = frozen / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    env = {**os.environ, 'PYTHONPATH': 'src:scripts:vendor/FlashRSI', 'HR_OUTPUT_ROOT': str(OUT),
           'HARNESSRISK_MODEL_PORT': '18764', 'HR_HARNESS': 'opencode',
           'HR_RUN_NAME': 'opencode_initial_full', 'HR_HOST_SCRIPT': 'flashrsi_harnessrisk_host_opencode_session_v8.py',
           'HR_STRICT_EVOLUTION': '1', 'HR_EVOLUTION_CASE_ATTEMPTS': '1'}
    (OUT / 'queue_pid.json').write_text(json.dumps({'queue_pid': os.getpid()}) + '\n')
    expected = {c['case_id'] for c in read(OUT / 'case_manifest.json')['cases']}
    def record(stage, **data):
        row = {'at': datetime.datetime.now().astimezone().isoformat(), 'stage': stage, **data}
        with (OUT / 'queue_events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    def execute(stage, args):
        record(stage, event='start')
        with (OUT / (stage + '.log')).open('x') as log:
            child = subprocess.Popen([str(PYTHON), '-u', *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            record(stage, event='spawned', pid=child.pid)
            code = child.wait()
        record(stage, event='exit', exit_code=code)
        return code
    def judge(stage, cases):
        existing = read(cases / 'llm_judge_multi_harness/manifest.json', [])
        valid = [x for x in existing if x.get('status') in ('judged', 'existing')]
        if len(existing) == len(valid) == 128 and {x.get('case_id') for x in valid} == expected:
            record(stage, event='judge_already_complete')
            return
        execute(stage + '_judge', ['scripts/judge_harnessrisk_shengsuanyun.py', str(cases), '--deployment', 'openai/gpt-5.4-nano'])
    execute('opencode_initial_full', ['scripts/flashrsi_harnessrisk_miniagent_initial.py'])
    run = OUT / 'opencode_initial_full'
    if initial_gate(run, expected):
        judge('opencode_initial', run / 'cases')
    else:
        record('opencode_initial', event='full_gate_failed', next_action='retain error cases for serial repair; no partial judge')
    from flashrsi_matrix_repair_loop import inspect_run
    for algorithm in ALGORITHMS:
        stage = 'opencode_' + algorithm + '_full'
        execute(stage, ['scripts/flashrsi_harnessrisk_algorithm.py', '--algorithm', algorithm])
        gate = inspect_run(OUT / stage, expected, None)
        record(stage, event='acceptance', status=gate['status'], valid=gate['valid'], reasons=gate['reasons'])
        if gate['status'] == 'needs_judge':
            judge('opencode_' + algorithm, OUT / stage / 'selected_cases')
    (OUT / 'opencode_skill_algorithms_unavailable.txt').write_text('SkillOpt/Evo-Harness N/A: upstream OpenCode Snapshot has no skill component.\n')
    subprocess.run([str(PYTHON), 'scripts/flashrsi_matrix_repair_loop.py', '--scan'], cwd=ROOT, env=env, check=True)
    record('queue', event='ended_all_stages_attempted_not_necessarily_accepted')


if __name__ == '__main__':
    main()
