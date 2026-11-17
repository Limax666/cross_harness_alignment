"""NEW Pi case owner schedule: native session, unchanged confinement and caps."""
from __future__ import annotations

import copy
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

from flashrsi_pi_owner_profile_v6 import (CONTRACT, PROFILE, ProfileError, byte_hash, digest,
    save_json, save_bytes, session_receipt, terminal_receipt, provider_receipt, process_receipt, read_native_session,
    target_config, snapshot_source_map)
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
    if os.getpid() != 1 or os.environ.get('HR_HARNESS') != 'pi':
        raise ProfileError('dedicated_PID_namespace_Pi_supervisor_required')
    from flashrsi.core import Snapshot
    from flashrsi_gateway import create_gateway
    from flashrsi_model_relay import relay_request
    from flashrsi_harnessrisk_privacy import PROTOCOL, assert_no_oracle, audit_case_inputs
    case, root = os.environ['HR_CASE_ID'], Path(os.environ['HR_RUN_ROOT'])
    raw_source = (DATA / (case + '.json')).read_bytes()
    task = json.loads(raw_source)['task']
    owners = task.get('user_messages') or [task['user_message']]
    if not owners or any(not isinstance(o, str) or not o for o in owners):
        raise ProfileError('original_owner_schedule_invalid')
    snapshot = Snapshot.load(os.environ['HR_SNAPSHOT'])
    if snapshot.id != os.environ['HR_EXPECTED_ID'] or snapshot.agent_type != 'pi':
        raise ProfileError('Snapshot_identity_or_harness_changed')
    path = root / case
    if path.exists():
        raise FileExistsError('existing_case_preserved:' + str(path))
    records, gate_errors = [], []
    class Proxy:
        def forward(self, request):
            assert_no_oracle(request)
            record = {'request': copy.deepcopy(request)}
            records.append(record)
            try:
                record['response'] = relay_request(os.environ['DEV_RELAY_PATH'], os.environ['DEV_RELAY_TOKEN'], request)
                return record['response']
            except Exception as exc:
                record['failure'] = {'type': type(exc).__name__, 'detail': str(exc)[:500]}
                raise
    class Gate:
        def abort(self): gate_errors.append('gateway_request_invalid')
    class Dispatcher:
        def dispatch(self, *_args): raise ProfileError('unexpected_extra_tool_gateway')
    token = secrets.token_urlsafe(32)
    server = create_gateway(Proxy(), Dispatcher(), Gate(), token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with tempfile.TemporaryDirectory(prefix='hr-supervisor-') as temporary:
        tmp = Path(temporary)
        manager, session_file = None, None
        mock_barrier = MockBarrier(case, snapshot.id)
        try:
            with (tmp / 'manager.out').open('w') as out, (tmp / 'manager.err').open('w') as err:
                manager = subprocess.Popen([sys.executable, str(ROOT / 'scripts/flashrsi_pi_owner_mock_manager_v6.py'),
                    mock_barrier.url, 'serve-mocks', '--mock-backend', 'process',
                    '--run-root', str(root), '--run-id', case, '--data-dir', str(DATA), '--case-id', case,
                    '--harness', 'pi'], stdout=out, stderr=err, start_new_session=True,
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
            session_file = scratch / 'native-case/pi-session.jsonl'
            private = path / 'native_session'
            private.mkdir(mode=0o700)
            result = {'case_id': case, 'harness': 'flashrsi_pi', 'snapshot_id': snapshot.id,
                'native_source_sha256': snapshot_source_map(snapshot),
                'source_case_sha256': byte_hash(raw_source), 'working_dir': str(work),
                'privacy_protocol': PROTOCOL, 'execution_protocol': PROFILE, 'contract': CONTRACT,
                'status': 'error', 'driver_status': 'owner_schedule_incomplete',
                'status_semantics': CONTRACT['status_semantics'],
                'measurement_validity': {'valid': False, 'profile': PROFILE},
                'original_owner_hashes': [digest(o) for o in owners], 'turns': [], 'owner_receipts': []}
            manager_start = census()[manager.pid]['start']
            result['mock_manager_identity'] = {'pid': manager.pid, 'start': manager_start}
            before, session_id = b'', None
            failure = None
            with snapshot._export(full=True) as exported:
                entry = str(ROOT / 'scripts/flashrsi_harnessrisk_pi_native_owner_entry_v6.py')
                ro = sorted(set(['/usr', '/lib', '/lib64', '/etc/ssl/certs', str(Path(sys.prefix).resolve()),
                    str(Path(sys.base_prefix).resolve()), str(Path(exported).resolve()), str(ROOT / 'scripts'),
                    str(ROOT / 'vendor/FlashRSI')]))
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
                    owner_deadline = time.monotonic() + 360
                    mock_barrier.open_owner(index, owner_deadline)
                    offset = len(records)
                    request = {'profile': PROFILE, 'harness': 'pi', 'question': owner, 'owner_index': index,
                        'workspace_dir': str(exported), 'working_dir': str(work), 'session_file': str(session_file),
                        'expected_session_sha256': byte_hash(before),
                        'config': target_config(token, f'http://127.0.0.1:{server.server_port}/v1')}
                    prefix = f'owner_{index:03d}'
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
                            try: child.wait(timeout=360)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait(timeout=5)
                                raise ProfileError('native_sandbox_wall_timeout_requires_case_namespace_exit')
                        if child.returncode:
                            raise ProfileError('native_sandbox_failed:' + str(child.returncode))
                        processes = census()
                        if manager.poll() is not None or processes.get(manager.pid, {}).get('start') != manager_start:
                            raise ProfileError('original_mock_manager_identity_or_liveness_lost')
                        settlement = process_receipt(processes, manager.pid, namespace_pid=os.getpid())
                        mock_settlement = mock_barrier.freeze()
                        save_json(private / (prefix + '.mock.json'), mock_settlement)
                        output = json.loads((private / (prefix + '.stdout.log')).read_text())
                        native, boundaries = output['native'], output['boundaries']
                        archive = read_native_session(session_file, scratch)
                        save_bytes(private / (prefix + '.jsonl'), archive)
                        save_json(private / (prefix + '.native.json'), native)
                        save_json(private / (prefix + '.boundaries.json'), boundaries)
                        save_json(private / (prefix + '.provider.json'), records[offset:])
                        result['turns'].append({'owner_turn': index, 'prompt': owner, 'response_text': native['output'],
                            'messages': native['events'], 'usage': native['usage'],
                            'stop_reason': native['stop_reason'], 'native_status': native['status'],
                            'native_error_message': native['error_message']})
                        if (output.get('profile') != PROFILE or output.get('owner_index') != index
                                or output.get('owner_sha256') != digest(owner)
                                or output.get('session_before_sha256') != byte_hash(before)
                                or output.get('session_after_sha256') != byte_hash(archive)
                                or output.get('native_model_limits') != {'context_window': 32768, 'model_max_tokens': 4096}
                                or not isinstance(output.get('native_compaction_settings'), dict) or gate_errors):
                            raise ProfileError('native_entry_boundary_identity_or_gateway_failure')
                        session = session_receipt(before, archive, owner, str(work), session_id)
                        terminal = terminal_receipt(native, owner)
                        provider = provider_receipt(boundaries, records[offset:], native['usage'], owner)
                        result['owner_receipts'].append({'owner_index': index, 'session': session, 'terminal': terminal,
                            'provider': provider, 'process_settlement': settlement, 'mock_settlement': mock_settlement,
                            'runtime_settings': {'native_model_limits': output['native_model_limits'],
                                                 'native_compaction_settings': output['native_compaction_settings']},
                            'session_archive': f'native_session/{prefix}.jsonl',
                            'native_archive': f'native_session/{prefix}.native.json',
                            'boundary_archive': f'native_session/{prefix}.boundaries.json',
                            'mock_archive': f'native_session/{prefix}.mock.json',
                            'provider_archive': f'native_session/{prefix}.provider.json'})
                        before, session_id = archive, session['session_id']
                        save_json(path / 'native_owner_progress.json', {'profile': PROFILE,
                            'actual_owner_receipts': result['owner_receipts'], 'original_owner_hashes': result['original_owner_hashes']})
                    except Exception as exc:
                        failure = {'owner_index': index, 'type': type(exc).__name__, 'detail': str(exc)}
                        save_json(path / 'mock_request_failure.json', {'failure': failure,
                            'requests': mock_barrier.ledger.requests, 'failures': mock_barrier.ledger.failures})
                        try:
                            raw = read_native_session(session_file, scratch, missing_ok=True)
                            if raw: save_bytes(private / (prefix + '.failed.jsonl'), raw)
                        except ProfileError as archive_error:
                            failure['archive_error'] = str(archive_error)
                        break
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
            save_json(path / 'model_request_evidence.json', {'requests': [r['request'] for r in records],
                'responses': [r.get('response') for r in records], 'profile': PROFILE})
            try:
                export = subprocess.run([sys.executable, str(ADAPTER), 'export', str(path), '--case-id', case, '--harness', 'pi'],
                                        capture_output=True, text=True, timeout=60)
                if export.returncode:
                    raise ProfileError('unchanged_source_export_failed:' + export.stderr[-1000:])
                graded = subprocess.run([sys.executable, str(BENCH / 'harness_adapter/scripts/evaluate_run.py'), str(path),
                    '--data-dir', str(DATA), '--output', str(path / 'official_rule_evaluation.json'), '--pretty'],
                    capture_output=True, text=True, timeout=60)
                if graded.returncode:
                    raise ProfileError('unchanged_official_evaluation_failed:' + graded.stderr[-1000:])
                if complete:
                    from flashrsi_pi_owner_profile_v6 import validate_case
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
                              'measurement_valid': complete, 'native_outcomes': [r['terminal'] for r in result['owner_receipts']]}))
            return 0 if complete else 2
        finally:
            try:
                if path.is_dir() and session_file is not None and session_file.exists():
                    try:
                        save_bytes(path / 'native_session/final_session.jsonl', read_native_session(session_file, scratch))
                    except ProfileError as exc:
                        save_json(path / 'native_session/final_archive_failure.json', {'error': str(exc)})
            finally:
                mock_barrier.close()
                if manager is not None:
                    manager.terminate()
                    try: manager.wait(timeout=30)
                    except subprocess.TimeoutExpired: manager.kill(); manager.wait()
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()


if __name__ == '__main__':
    raise SystemExit(main())
