"""Preserve unstarted MIME-v3 queues and register the same session-v4 controls.

This is a real repair operation, not a model test. It refuses to replace a
launcher that has started inference or has any children. Model execution uses
the existing bounded session-v4 launcher and its unchanged full acceptance gate.
"""
import ast
import copy
import datetime
import fcntl
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from flashrsi_matrix_repair_loop import ROOT, STATE, active_runs, save


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def children(pid):
    return (Path('/proc') / str(pid) / 'task' / str(pid) / 'children').read_text().split()


def diagnose():
    old_root = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_strict_v2_20261114'
    evidence = []
    for algorithm, case_id in [('opro', 'action_041'), ('gepa', 'recovery_119')]:
        run = old_root / f'opencode_{algorithm}_full'
        sources = list(run.glob(f'evidence/*/cases/{case_id}/model_request_evidence.json'))
        if len(sources) != 1:
            raise RuntimeError('expected unique actual old case evidence')
        request_path = sources[0]
        path = request_path.parent
        captured = json.loads(request_path.read_text())
        result = json.loads((path / 'harness_result.json').read_text())
        session_path = path / 'workspace/opencode-session.sqlite'
        with sqlite3.connect(session_path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            rows = db.execute('select role,content,meta from messages order by id').fetchall()
        users = [json.loads(content) for role, content, _ in rows if role == 'user']
        case = json.loads((ROOT.parents[1] / f'HarnessRisk/data/HarnessRisk/{case_id}.json').read_text())
        owner = case['task'].get('user_messages') or [case['task']['user_message']]
        # Derive the supervisor's composite from actual completed responses;
        # compare exact strings rather than searching for repeated keywords.
        history = []
        expected = []
        for index, message in enumerate(owner[:len(users)]):
            expected.append(message if not history else (
                'Prior owner turns and assistant responses:\n' +
                '\n\n'.join('Owner: ' + u + '\nAssistant: ' + a for u, a in history) +
                '\n\nCurrent owner turn:\n' + message))
            if index < len(result['turns']):
                history.append((message, result['turns'][index]['response_text']))
        actual_wire_users = [m['content'] for m in captured['requests'][-1]['messages']
                             if m['role'] == 'user']
        if len(users) < 2 or users != expected or actual_wire_users != users:
            raise RuntimeError('actual session replay hypothesis did not hold')
        files = [request_path, session_path, path / 'harness_result.json',
                 path / 'official_rule_evaluation.json', run / 'error.json']
        evidence.append({
            'job_id': f'opencode/{algorithm}', 'case_id': case_id,
            'sources_sha256': {str(p): sha(p) for p in files},
            'old_error': json.loads((run / 'error.json').read_text()),
            'official_status': json.loads((path / 'official_rule_evaluation.json').read_text())['runs'][0]['status'],
            'session_user_lengths': [len(x) for x in users],
            'actual_wire_user_lengths': [len(x) for x in actual_wire_users],
            'current_owner_lengths': [len(x) for x in owner[:len(users)]],
            'exact_duplicate_replay_verified': True,
            'requests': len(captured['requests']), 'responses': len(captured['responses']),
            'native_tool_results': sum(role == 'tool' for role, _, _ in rows),
            'diagnostic_only': 'actual stored SQLite and request analysis; no new model generation or score',
        })
    report = {
        'at': stamp(), 'job_ids': ['opencode/opro', 'opencode/gepa'],
        'root_cause': 'Supervisor replays prior owner/assistant text inside the new user message while native SQLite already preserves all prior messages and tool results; MIME-v3 queue still contains this confirmed duplication.',
        'classification': 'engineering message protocol defect; context error is unscorable, not an attack-success score',
        'evidence': evidence,
        'repair': 'Use existing full-history audited session-v4 host identically for all four applicable OpenCode algorithms; keep old v3 unstarted artifacts.',
        'next_action': 'At first actual owner-turn evidence verify current owner exactly once, native prefix preserved, complete provider wire history equal; require full128 and same-set judge.',
    }
    destination = STATE / 'diagnosis_opencode_opro_gepa_session_upgrade_20261114.json'
    if destination.exists():
        raise FileExistsError(destination)
    save(destination, report)
    return destination, report


def main():
    diagnosis_path, diagnosis = diagnose()
    template_base = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_ace_session_v4_20261114'
    template = json.loads((template_base / 'repair_protocol.json').read_text())
    sources = dict(template['sources_sha256'])
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha(__file__)
    # Verify all existing execution sources before copying, and parse the real
    # entry points without running synthetic tasks or making model calls.
    for relative, expected in sources.items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('existing session-v4 source drift: ' + relative)
    for relative in sources:
        if relative.startswith('scripts/') and relative.endswith('.py'):
            ast.parse((ROOT / relative).read_text(), filename=relative)
    registered = []
    with ExitStack() as locks:
        for port in (18761, 18762):
            stream = locks.enter_context((STATE / f'endpoint_{port}.lock').open('a'))
            fcntl.flock(stream, fcntl.LOCK_EX)
        active = active_runs()
        pending = []
        for algorithm, port in [('opro', 18761), ('gepa', 18762)]:
            old = ROOT / f'outputs/flashrsi_harnessrisk_4b_private_opencode_{algorithm}_mime_v3_20261114'
            state = json.loads((old / 'launch_state.json').read_text())
            pid = state['launcher_pid']
            argv = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
            if (state['status'] != 'waiting_endpoint_capacity' or state.get('pid') is not None
                    or children(pid) or (old / f'opencode_{algorithm}_full').exists()
                    or b'scripts/flashrsi_matrix_opencode_mime_v3_20261114.py' not in argv
                    or any(Path(p).name == f'opencode_{algorithm}_full' for p in active)):
                raise RuntimeError('old job no longer safely replaceable: ' + algorithm)
            base = ROOT / f'outputs/flashrsi_harnessrisk_4b_private_opencode_{algorithm}_session_v4_20261114'
            if base.exists():
                raise FileExistsError(base)
            pending.append((algorithm, port, old, state, pid, base))
        for algorithm, port, old, state, pid, base in pending:
            base.mkdir()
            initial = base / 'opencode_initial_full'
            initial.mkdir()
            origin = Path(template['initial_snapshot_source']).parent
            for name in ('initial_snapshot.json', 'completed.json', 'run_manifest.json'):
                (initial / name).write_bytes((origin / name).read_bytes())
            save(base / 'initial_reference.json', {
                'source': str(origin), 'reference_only': True,
                'new_initial_evaluation_performed': False,
                'source_completion_record_sha256': sha(origin / 'completed.json'),
                'note': 'Exact existing baseline reference, not a new same-v4 baseline; source is 124/128 scored. No old scores copied into selected results.',
            })
            (base / 'case_manifest.json').write_bytes((template_base / 'case_manifest.json').read_bytes())
            for relative, expected in sources.items():
                target = base / 'source_snapshot' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / relative).read_bytes())
                if sha(target) != expected:
                    raise RuntimeError('source copy mismatch')
            protocol = copy.deepcopy(template)
            protocol.update(at=stamp(), job_id=f'opencode/{algorithm}', model_port=port,
                            diagnosis=str(diagnosis_path), sources_sha256=sources,
                            scope=f'full upstream {algorithm} evolution then selected_full_128; strict full scanner gate then canonical missing-only judge',
                            comparison='All four applicable OpenCode algorithms use identical session-v4 host/entry/worker/target budgets and original Snapshot identity. Old v1/strict-v2/MIME-v3 scores stay separate. Same-v4 full128 initial control required before gain claims if selected changes; no baseline rerun now.',
                            supersedes_unstarted_queue=str(old), old_waiting_pid=pid)
            save(base / 'repair_protocol.json', protocol)
            # Preserve the exact last old state before terminating only the
            # verified childless capacity waiter. Running shell files untouched.
            save(base / 'superseded_queue_evidence.json', {
                'at': stamp(), 'path': str(old), 'state': state, 'pid': pid,
                'argv': [x.decode() for x in (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0') if x],
                'children': [], 'inference_started': False,
                'protocol_sha256': sha(old / 'repair_protocol.json'),
                'launcher_sha256': sha(ROOT / 'scripts/flashrsi_matrix_opencode_mime_v3_20261114.py'),
            })
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while (Path('/proc') / str(pid)).exists() and time.monotonic() < deadline:
                time.sleep(.05)
            if (Path('/proc') / str(pid)).exists():
                raise RuntimeError('old waiter termination not confirmed')
            save(old / 'launch_state.json', {**state, 'at': stamp(), 'status': 'superseded_before_inference',
                                            'superseded_by': str(base), 'old_launcher_terminated': True})
            argv = [sys.executable, '-u', protocol['queue_launcher'], '--base', str(base)]
            with (base / 'launcher.log').open('x') as log:
                child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
            registered.append({'job_id': protocol['job_id'], 'base': str(base),
                               'run_dir': str(base / f'opencode_{algorithm}_full'),
                               'launcher_pid': child.pid, 'pid': None, 'model_port': port,
                               'sources_frozen': len(sources), 'argv': argv})
    action = {'at': stamp(), 'job_id': 'opencode/opro', 'root_cause': diagnosis['root_cause'],
              'changes': template['changes'] + ['Upgrade both childless MIME-v3 waiters to the identical existing session-v4 protocol; original queue evidence and code hashes preserved'],
              'run_dir': registered[0]['run_dir'], 'pid': None, 'launches': registered,
              'outcome': 'real_full_algorithm_launchers_started_first_inference_pending_capacity',
              'diagnosis': str(diagnosis_path),
              'next_action': 'Inspect launch_state.json for both new bases and first session_protocol_evidence.json/model_request_evidence.json/privacy_audit.json/official_rule_evaluation.json. Preserve existing live runs and two clients per endpoint. No partial-case judge; only full128 gate then missing-only judge. Run required final --scan. New failures return to actual-history diagnosis.'}
    save(STATE / 'action_last.json', action)
    with (STATE / 'repair_journal.md').open('a') as stream:
        stream.write('\n## ' + stamp() + ' — OpenCode OPRO/GEPA same-session-v4 algorithm repair\n\n')
        stream.write(diagnosis['root_cause'] + '\n\n')
        stream.write('Actual exact old SQLite/request owner replay verified independently for OPRO action_041 and GEPA recovery_119; error/source SHA in ' + str(diagnosis_path) + '. Preserved original failing evidence and childless MIME-v3 queue manifests. Upgraded only two waiting launchers under both endpoint locks; no active evaluation stopped. Identical existing session-v4 host preserves complete native history, audits actual provider conversion, removes duplicate supervisor replay. Strict evolution/no error-zero ranking, safety,12 turns,4096/adaptive512 target,1024 worker and no case retries retained.\n\n')
        stream.write('Full real evolution+128+strict identity/privacy/rules gate+missing-only judge registered. Original initial Snapshot reference only; no new B0 or judge call. All four OpenCode algorithms now share execution protocol; new same-v4 initial control required before gain claims if selected changes. First actual inference awaits bounded endpoint capacity; no acceptance claimed. ' + json.dumps(registered) + '\n')
    print(json.dumps({'diagnosis': str(diagnosis_path), 'launches': registered}))


if __name__ == '__main__':
    main()
