"""NEW Claude case owner schedule: native session, unchanged confinement and caps."""
from __future__ import annotations

import copy
import base64
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse

from flashrsi_claude_owner_profile_v9 import (CONTRACT, PROFILE, ProfileError, byte_hash, digest,
    save_json, save_bytes, process_receipt, target_config, snapshot_source_map,
    locate_session, validate_owner, archive_native_tree)
from flashrsi_claude_owner_model_barrier_v9 import ModelBarrier
from flashrsi_claude_owner_identity_v9 import IdentityObserver
from flashrsi_pi_owner_mock_barrier_v6 import MockBarrier

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT.parents[1] / 'HarnessRisk'
ADAPTER, DATA = BENCH / 'harness_adapter/harness_adapter.py', BENCH / 'data/HarnessRisk'
MODEL = 'qwen35-4b-preflight'


def census():
    result = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            raw = path.read_text()
            fields = raw[raw.rfind(')') + 2:].split()
            result[int(path.parent.name)] = {'state': fields[0], 'ppid': int(fields[1]), 'start': int(fields[19])}
        except FileNotFoundError:
            continue
    return result


def main():
    if os.getpid() != 1 or os.environ.get('HR_HARNESS') != 'claude_code':
        raise ProfileError('dedicated_PID_namespace_Claude_supervisor_required')
    from flashrsi.core import Snapshot
    from flashrsi_native_protocol_relay import create_native_gateway
    import flashrsi_native_protocol_relay as native_protocol
    from flashrsi_harnessrisk_privacy import PROTOCOL, assert_no_oracle, audit_case_inputs
    case, root = os.environ['HR_CASE_ID'], Path(os.environ['HR_RUN_ROOT'])
    raw_source = (DATA / (case + '.json')).read_bytes()
    task = json.loads(raw_source)['task']
    owners = task.get('user_messages') or [task['user_message']]
    if not owners or any(not isinstance(o, str) or not o for o in owners):
        raise ProfileError('original_owner_schedule_invalid')
    snapshot = Snapshot.load(os.environ['HR_SNAPSHOT'])
    if snapshot.id != os.environ['HR_EXPECTED_ID'] or snapshot.agent_type != 'claude_code':
        raise ProfileError('Snapshot_identity_or_harness_changed')
    path = root / case
    if path.exists():
        raise FileExistsError('existing_case_preserved:' + str(path))
    records, observed_outcomes = [], []
    lock = threading.Lock()
    original_native_request = native_protocol.native_relay_request
    def observed_request(relay_path, relay_token, route, request):
        assert_no_oracle(request)
        with lock:
            record = {'path': route, 'request': copy.deepcopy(request), 'settled': False}
            records.append(record)
        try:
            status, content_type, body = original_native_request(relay_path, relay_token, route, request)
            record.update(status=status, content_type=content_type, response_sha256=byte_hash(body),
                          response_body_b64=base64.b64encode(body).decode())
            return status, content_type, body
        except Exception as exc:
            record['failure'] = {'type':type(exc).__name__, 'detail':str(exc)[:500]}
            raise
        finally:
            with lock: record['settled'] = True
    native_protocol.native_relay_request = observed_request
    token = secrets.token_urlsafe(32)
    model_barrier = ModelBarrier(lambda: create_native_gateway(os.environ['NATIVE_RELAY_PATH'],
        os.environ['NATIVE_RELAY_TOKEN'], token, MODEL, completion_budget=4096))
    server = model_barrier.server
    with tempfile.TemporaryDirectory(prefix='hr-supervisor-') as temporary:
        tmp = Path(temporary)
        manager, session_file = None, None
        mock_barrier = MockBarrier(case, snapshot.id)
        try:
            with (tmp / 'manager.out').open('w') as out, (tmp / 'manager.err').open('w') as err:
                manager = subprocess.Popen([sys.executable, str(ROOT / 'scripts/flashrsi_pi_owner_mock_manager_v6.py'),
                    mock_barrier.url, 'serve-mocks', '--mock-backend', 'process',
                    '--run-root', str(root), '--run-id', case, '--data-dir', str(DATA), '--case-id', case,
                    '--harness', 'claude_code'], stdout=out, stderr=err, start_new_session=True,
                    env={**os.environ, 'HARNESSRISK_PRIVATE_CASE': '1'})
            deadline, info = time.monotonic() + 90, None
            while time.monotonic() < deadline:
                if manager.poll() is not None:
                    raise ProfileError('mock_manager_exited:' + (tmp / 'manager.err').read_text()[-1000:])
                try:
                    info = json.loads((tmp / 'manager.out').read_text())
                    break
                except (json.JSONDecodeError, FileNotFoundError): time.sleep(.25)
            if info is None:
                raise ProfileError('mock_manager_startup_timeout')
            mock_barrier.upstream = info['gateway']
            work = Path(info['workspace']).resolve()
            if work != (path / 'workspace').resolve():
                raise ProfileError('mock_workspace_identity_changed')
            save_json(path / 'privacy_audit.json', audit_case_inputs(path))
            scratch = tmp / 'scratch'
            scratch.mkdir()
            (scratch / 'native-case').mkdir()
            state_dir = scratch / 'native-case/claude'; state_dir.mkdir(mode=0o700)
            private = path / 'native_session'
            private.mkdir(mode=0o700)
            result = {'case_id': case, 'harness': 'flashrsi_claude_code', 'snapshot_id': snapshot.id,
                'native_source_sha256': snapshot_source_map(snapshot),
                'source_case_sha256': byte_hash(raw_source), 'working_dir': str(work),
                'privacy_protocol': PROTOCOL, 'execution_protocol': PROFILE, 'contract': CONTRACT,
                'status': 'error', 'driver_status': 'owner_schedule_incomplete',
                'status_semantics': CONTRACT['status_semantics'],
                'measurement_validity': {'valid': False, 'profile': PROFILE},
                'original_owner_hashes': [digest(o) for o in owners], 'turns': [], 'owner_receipts': []}
            manager_start = census()[manager.pid]['start']
            result['mock_manager_identity'] = {'pid': manager.pid, 'start': manager_start}
            before, session_id, previous = b'', None, None
            failure = None
            with snapshot._export(full=True) as exported:
                entry = str(ROOT / 'scripts/flashrsi_harnessrisk_claude_native_owner_entry_v9.py')
                ro = sorted(set(['/usr', '/lib', '/lib64', '/etc/ssl/certs', str(Path(sys.prefix).resolve()),
                    str(Path(sys.base_prefix).resolve()), str(Path(exported).resolve()), str(ROOT / 'scripts'),
                    str(ROOT / 'vendor/FlashRSI'),
                    '/data/home/liumingxiao/.npm-global/lib/node_modules/@anthropic-ai/claude-code/bin']))
                env = {'PATH': str(work / 'bin') + ':' + str(Path(sys.executable).parent) + ':/usr/bin:/bin',
                    'LANG': 'C.UTF-8', 'HOME': str(scratch), 'TMPDIR': str(scratch), 'MOCK_GATEWAY_BASE_URL': mock_barrier.url}
                sandbox = {'python': sys.executable, 'entry': entry, 'readonly': ro, 'writable': [str(work), str(scratch)],
                    'ports': [server.server_port, mock_barrier.server.server_port], 'environment': env}
                save_json(path / 'runtime_environment_evidence.json', {'profile': PROFILE,
                    'sandbox_policy_unchanged': True, 'sandbox': {k: v for k, v in sandbox.items() if k != 'environment'},
                    'python_bin_on_path': str(Path(sys.executable).parent), 'contract': CONTRACT})
                save_json(path / 'mock_transport_mapping.json', {'case_id': case, 'snapshot_id': snapshot.id,
                    'public_gateway': mock_barrier.url, 'original_gateway': info['gateway'],
                    'original_backend': info['backend'], 'allowed_target_mock_ports': [mock_barrier.server.server_port],
                    'mapping': 'transport endpoint only; original adapter generated URLs; original methods/headers/body/response bytes',
                    'settlement': 'accepted/unparsed/listen backlog accounted; full upstream response; gateway599 invalid'})
                for index, owner in enumerate(owners, 1):
                    owner_deadline = time.monotonic() + 600
                    model_barrier.open_owner(index)
                    mock_barrier.open_owner(index, owner_deadline)
                    offset = len(records)
                    request = {'profile': PROFILE, 'harness': 'claude_code', 'question': owner, 'owner_index': index,
                        'workspace_dir': str(exported), 'working_dir': str(work), 'state_dir': str(state_dir), 'resume': session_id,
                        'expected_session_sha256': byte_hash(before),
                        'identity_binding':{'case_id':case,'snapshot_id':snapshot.id,'owner_index':index,'nonce':secrets.token_hex(32)},
                        'identity_request':str(scratch / f'identity_{index:03d}.request.json'),
                        'identity_response':str(scratch / f'identity_{index:03d}.response.json'),
                        'config': target_config(token, f'http://127.0.0.1:{server.server_port}')}
                    prefix = f'owner_{index:03d}'
                    observer=None
                    try:
                        # Anonymous input keeps the ephemeral gateway capability
                        # out of persistent experiment configuration/evidence.
                        with tempfile.TemporaryFile(mode='w+', dir=scratch) as input_stream, \
                                (private / (prefix + '.stdout.log')).open('w') as out, \
                                (private / (prefix + '.stderr.log')).open('w') as err:
                            json.dump(request, input_stream)
                            input_stream.seek(0)
                            child = subprocess.Popen([str(ROOT / 'scripts/flashrsi_harnessrisk_sandbox.py'), '-I', '-B', entry],
                                stdin=input_stream, stdout=out, stderr=err,
                                env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'HARNESSRISK_SANDBOX_CONFIG': json.dumps(sandbox)})
                            observer=IdentityObserver(request['identity_request'],request['identity_response'],scratch,child.pid,owner_deadline,
                                expected_executable=request['config']['cli_path'],binding=request['identity_binding'])
                            try: child.wait(timeout=900)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait(timeout=5)
                                raise ProfileError('native_sandbox_wall_timeout_requires_case_namespace_exit')
                        if child.returncode:
                            raise ProfileError('native_sandbox_failed:' + str(child.returncode))
                        identity=observer.finish()
                        save_json(private / (prefix + '.identity.json'), identity)
                        output = json.loads((private / (prefix + '.stdout.log')).read_text())
                        save_json(private / (prefix + '.native.json'), output)
                        observed_outcomes.extend(f for f in output.get('raw_frames',[]) if f.get('type')=='result')
                        processes = census()
                        if manager.poll() is not None or processes.get(manager.pid, {}).get('start') != manager_start:
                            raise ProfileError('original_mock_manager_identity_or_liveness_lost')
                        settlement = process_receipt(processes, manager.pid, namespace_pid=os.getpid())
                        model_settlement = model_barrier.freeze()
                        save_json(private / (prefix + '.model_transport.json'), model_settlement)
                        mock_settlement = mock_barrier.freeze()
                        save_json(private / (prefix + '.mock.json'), mock_settlement)
                        with lock:
                            owner_records = copy.deepcopy(records[offset:])
                        save_json(private / (prefix + '.provider.json'), owner_records)
                        if any(r.get('settled') is not True for r in owner_records) or server.errors:
                            raise ProfileError('native_model_requests_or_gateway_not_settled')
                        session_file, archive = locate_session(state_dir / 'state', scratch, output['session_id'])
                        save_bytes(private / (prefix + '.jsonl'), archive)
                        # Raw errors/null result survive even when validation fails.
                        result['turns'].append({'owner_turn': index, 'prompt': owner,
                            'response_text': output['output'], 'messages':output['messages'], 'usage':output['usage'],
                            'native_result':output['native_result'], 'stop_reason':(output['native_result'] or {}).get('stop_reason')})
                        native_receipt = validate_owner(output, before, archive, owner, index, session_id,
                                                        owner_records, str(work), previous, trusted_identity=identity)
                        result['owner_receipts'].append({'owner_index':index, 'native_receipt':native_receipt,
                            'process_settlement':settlement, 'mock_settlement':mock_settlement,
                            'model_transport_settlement':model_settlement,
                            'model_transport_archive':f'native_session/{prefix}.model_transport.json',
                            'identity_archive':f'native_session/{prefix}.identity.json',
                            'session_archive':f'native_session/{prefix}.jsonl',
                            'native_archive':f'native_session/{prefix}.native.json',
                            'provider_archive':f'native_session/{prefix}.provider.json',
                            'mock_archive':f'native_session/{prefix}.mock.json'})
                        before, session_id = archive, native_receipt['session']['session_id']
                        previous = native_receipt['terminal']
                        save_json(path / 'native_owner_progress.json', {'profile': PROFILE,
                            'actual_owner_receipts': result['owner_receipts'], 'original_owner_hashes': result['original_owner_hashes']})
                    except Exception as exc:
                        failure = {'owner_index': index, 'type': type(exc).__name__, 'detail': str(exc)}
                        if observer is not None:
                            try:save_json(private / (prefix + '.identity.json'),observer.finish())
                            except ProfileError as identity_error:
                                save_json(private / (prefix + '.identity_failure.json'),{'error':str(identity_error),'observed':observer.receipt})
                        save_json(path / 'mock_request_failure.json', {'failure': failure,
                            'requests': mock_barrier.ledger.requests, 'failures': mock_barrier.ledger.failures})
                        try:
                            raw = locate_session(state_dir / 'state', scratch, output.get('session_id', session_id))[1]
                            if raw: save_bytes(private / (prefix + '.failed.jsonl'), raw)
                        except (ProfileError, UnboundLocalError, KeyError) as archive_error:
                            failure['archive_error'] = str(archive_error)
                        break
            result['native_outcomes'] = observed_outcomes
            result['native_max_turns_count'] = sum(t.get('native_result',{}).get('subtype')=='error_max_turns' for t in result['turns'] if t.get('native_result'))
            # Invalid/uncertain work must never race the source exporter. A
            # failed owner retains evidence and exits its owned case namespace.
            if failure is not None:
                result['failure'] = failure
                save_json(path / 'harness_result.json', result)
                return 2
            complete = len(result['owner_receipts']) == len(owners) and failure is None
            if complete:
                result.update(status='completed', driver_status='owner_schedule_completed')
            result['failure'] = failure
            save_json(path / 'harness_result.json', result)
            (path / 'trajectory/hermes_turns.jsonl').write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in result['turns']))
            save_json(path / 'model_request_evidence.json', {'requests': records, 'profile': PROFILE})
            try:
                export = subprocess.run([sys.executable, str(ADAPTER), 'export', str(path), '--case-id', case, '--harness', 'claude_code'],
                                        capture_output=True, text=True, timeout=60)
                if export.returncode:
                    raise ProfileError('unchanged_source_export_failed:' + export.stderr[-1000:])
                graded = subprocess.run([sys.executable, str(BENCH / 'harness_adapter/scripts/evaluate_run.py'), str(path),
                    '--data-dir', str(DATA), '--output', str(path / 'official_rule_evaluation.json'), '--pretty'],
                    capture_output=True, text=True, timeout=60)
                if graded.returncode:
                    raise ProfileError('unchanged_official_evaluation_failed:' + graded.stderr[-1000:])
                if complete:
                    from flashrsi_claude_owner_profile_v9 import validate_case
                    validate_case(path, case, snapshot.id, owners, byte_hash(raw_source), require_host=False,
                                  _pending=True, native_source=result['native_source_sha256'])
                    result['measurement_validity'] = {'valid': True, 'profile': PROFILE}
                    save_json(path / 'harness_result.json', result)
            except Exception as exc:
                result.update(status='error', driver_status='source_or_profile_admission_invalid',
                    measurement_validity={'valid': False, 'profile': PROFILE},
                    failure={'type': type(exc).__name__, 'detail': str(exc)})
                save_json(path / 'harness_result.json', result)
                raise
            print(json.dumps({'case_id': case, 'profile': PROFILE, 'driver_status': result['driver_status'],
                              'measurement_valid': complete, 'native_outcomes': [r['native_receipt']['terminal'] for r in result['owner_receipts']]}))
            return 0 if complete else 2
        finally:
            try:
                if path.is_dir():
                    with lock: save_json(path / 'native_session/all_provider_records.json', records)
                    try:
                        archive_native_tree(state_dir, scratch, path / 'native_session/final_native_tree')
                    except (ProfileError, UnboundLocalError) as exc:
                        save_json(path / 'native_session/native_tree_archive_failure.json', {'error':str(exc)})
                if path.is_dir() and session_file is not None and session_file.exists():
                    try:
                        save_bytes(path / 'native_session/final_session.jsonl', locate_session(state_dir / 'state', scratch, session_id)[1])
                    except ProfileError as exc:
                        save_json(path / 'native_session/final_archive_failure.json', {'error': str(exc)})
            finally:
                native_protocol.native_relay_request = original_native_request
                mock_barrier.close()
                if manager is not None:
                    manager.terminate()
                    try: manager.wait(timeout=30)
                    except subprocess.TimeoutExpired: manager.kill(); manager.wait()
                model_barrier.close()


if __name__ == '__main__':
    raise SystemExit(main())
