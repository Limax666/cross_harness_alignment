"""Launch the OpenCode initial and four applicable RSI baselines on GPU1."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_gpu1_v1_20261115'
SOURCE = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_v1_20261113/case_manifest.json'
FILES = ['scripts/flashrsi_harnessrisk_private_queue.sh', 'scripts/flashrsi_harnessrisk_miniagent_initial.py',
         'scripts/flashrsi_harnessrisk_algorithm.py', 'scripts/flashrsi_harnessrisk_benchmark.py',
         'scripts/flashrsi_harnessrisk_host_opencode_session_v7.py', 'scripts/flashrsi_harnessrisk_host_opencode_session_v6.py',
         'scripts/flashrsi_harnessrisk_host_opencode_session_v4.py', 'scripts/flashrsi_harnessrisk_opencode_session_supervisor_v7.py',
         'scripts/flashrsi_harnessrisk_opencode_session_supervisor_v6.py', 'scripts/flashrsi_harnessrisk_opencode_session_audit_v7.py',
         'scripts/flashrsi_harnessrisk_privacy.py', 'scripts/flashrsi_harnessrisk_sandbox.py', 'scripts/judge_harnessrisk_shengsuanyun.py']

def main():
    OUT.mkdir(exist_ok=False)
    shutil.copy2(SOURCE, OUT / 'case_manifest.json')
    previous = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_gepa_session_v7_20261115/repair_protocol.json'
    names = set(FILES) | set(json.loads(previous.read_text())['sources_sha256'])
    names.add(str(Path(__file__).resolve().relative_to(ROOT)))
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)}
    (OUT / 'repair_protocol.json').write_text(json.dumps({
        'protocol': 'private-v1-opencode-native-session-v7-gpu1-32k', 'model_port': 18764,
        'harness': 'opencode', 'algorithms': ['opro', 'gepa', 'evoprompt', 'ace'],
        'not_applicable': ['skillopt', 'evo_harness'], 'sources_sha256': hashes,
        'initial_and_evolution_same_host': True, 'case_count': 128,
        'privacy': 'harnessrisk-private-case-v1', 'strict_evolution': True,
        'comparison_limit': '32K GPU1 and session-v7; do not mix with old 16K runs',
        'queue_note': 'initial attempted then completed-only judge and four algorithm runs; incomplete results retain errors and cannot pass full acceptance'}, indent=2) + '\n')
    env = {**os.environ, 'HR_OUTPUT_ROOT': str(OUT), 'HARNESSRISK_MODEL_PORT': '18764',
           'HR_HOST_SCRIPT': 'flashrsi_harnessrisk_host_opencode_session_v7.py',
           'HR_STRICT_EVOLUTION': '1', 'HR_EVOLUTION_CASE_ATTEMPTS': '1'}
    with (OUT / 'queue_gpu1.log').open('x') as log:
        child = subprocess.Popen(['bash', 'scripts/flashrsi_harnessrisk_private_queue.sh', 'opencode'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        (OUT / 'queue_pid.json').write_text(json.dumps({'launcher_pid': os.getpid(), 'queue_pid': child.pid}) + '\n')
        code = child.wait()
    (OUT / 'queue_exit.json').write_text(json.dumps({'exit_code': code}) + '\n')
    return code

if __name__ == '__main__':
    raise SystemExit(main())
