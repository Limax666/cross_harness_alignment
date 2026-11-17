"""Pure profile guards over retained native events; no native code/API execution."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
RETAINED = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_pi_gepa_interpreter_v4_20261114/pi_initial_full/cases/action_041'


def retained():
    turn = json.loads((RETAINED / 'harness_result.json').read_text())['turns'][0]
    return {'status': 'error', 'stop_reason': 'length', 'error_message': 'Pi reached max_turns',
            'output': '', 'events': turn['messages'], 'usage': turn['usage']}


def session_bytes(messages, header):
    rows = [header]
    previous = None
    for i, message in enumerate(messages):
        identity = f'{i:08x}'
        rows.append({'type': 'message', 'id': identity, 'parentId': previous,
                     'timestamp': '2026-11-17T00:00:00Z', 'message': message})
        previous = identity
    return ''.join(json.dumps(row) + '\n' for row in rows).encode()


def test_retained_native_limit_is_typed_and_tools_settled():
    from flashrsi_pi_owner_profile_v6 import terminal_receipt
    receipt = terminal_receipt(retained())
    assert receipt['kind'] == 'native_max_turns'
    assert receipt['model_calls'] == 12 and receipt['tool_calls'] == 12
    assert receipt['native_status'] == 'error'
    assert retained()['output'] == ''


@pytest.mark.parametrize('mutation', ['wrong_error', 'aborted', 'missing_result', 'context', 'deferred', 'missing_python'])
def test_invalid_or_unknown_native_endpoint_never_becomes_normal_cap(mutation):
    from flashrsi_pi_owner_profile_v6 import terminal_receipt, ProfileError
    value = retained()
    if mutation == 'wrong_error': value['error_message'] = 'HTTP400 maximum context length'
    elif mutation == 'aborted': value['events'][-1]['aborted'] = True
    elif mutation == 'missing_result':
        value['events'] = [e for e in value['events'] if e.get('type') != 'tool_execution_end'][1:]
    elif mutation == 'context':
        value['events'].insert(1, {'type': 'message_end', 'message': {'role': 'assistant', 'stopReason': 'error', 'errorMessage': 'context exceeded'}})
    elif mutation == 'deferred': value['deferred_tool_use'] = {'id': 'pending'}
    else:
        end = next(e for e in value['events'] if e.get('type') == 'tool_execution_end')
        end['result'] = {'isError': True, 'structuredContent': {'exit_code': 127},
                         'content': [{'type': 'text', 'text': 'python3: command not found'}]}
        end['isError'] = True
    with pytest.raises(ProfileError): terminal_receipt(value)


def test_native_session_prefix_owner_once_and_leaf_preserved():
    from flashrsi_pi_owner_profile_v6 import session_receipt, ProfileError
    native = retained()
    header = next(e for e in native['events'] if e.get('type') == 'session')
    messages = next(e['messages'] for e in reversed(native['events']) if e.get('type') == 'agent_end')
    first = session_bytes(messages, header)
    a = session_receipt(b'', first, messages[0]['content'], header['cwd'])
    assert a['session_id'] == header['id'] and a['new_owner_messages'] == 1
    second = session_bytes(messages + [{'role': 'user', 'content': 'next original owner'}], header)
    b = session_receipt(first, second, 'next original owner', header['cwd'], a['session_id'])
    assert b['prefix_sha256'] == a['session_sha256']
    with pytest.raises(ProfileError):
        session_receipt(first, second.replace(b'"version": 3', b'"version": 2'), 'next original owner', header['cwd'], a['session_id'])
    duplicated = session_bytes(messages + [{'role': 'user', 'content': 'next original owner'}] * 2, header)
    with pytest.raises(ProfileError):
        session_receipt(first, duplicated, 'next original owner', header['cwd'], a['session_id'])


def test_provider_projection_requires_real_successful_matched_requests():
    from flashrsi_pi_owner_profile_v6 import provider_receipt, ProfileError
    evidence = json.loads((RETAINED / 'model_request_evidence.json').read_text())
    boundaries = [{'kind': 'agent', 'wire_messages': r['messages']} for r in evidence['requests']]
    records = [{'request': r, 'response': s} for r, s in zip(evidence['requests'], evidence['responses'])]
    assert provider_receipt(boundaries, records, retained()['usage'])['requests'] == 12
    generated = next(e['messages'] for e in reversed(retained()['events']) if e.get('type') == 'agent_end')
    owner = generated[0]['content']
    boundaries[0]['canonical_context'] = generated[:1]
    assert provider_receipt(boundaries, records, retained()['usage'], owner)['requests'] == 12
    with pytest.raises(ProfileError): provider_receipt(boundaries, records, retained()['usage'], 'invented extra owner')
    records[0]['failure'] = {'type': 'HTTP400'}
    with pytest.raises(ProfileError): provider_receipt(boundaries, records, retained()['usage'])
    records[0].pop('failure')
    records[0]['request'] = copy.deepcopy(records[0]['request'])
    records[0]['request']['messages'].append({'role': 'tool', 'tool_call_id': 'invented', 'content': 'No result provided'})
    with pytest.raises(ProfileError): provider_receipt(boundaries, records, retained()['usage'])


def test_namespace_process_receipt_rejects_unowned_surviving_writer():
    from flashrsi_pi_owner_profile_v6 import process_receipt, ProfileError
    processes = {1: {'ppid': 0, 'start': 1, 'state': 'S'},
                 2: {'ppid': 1, 'start': 2, 'state': 'S'},
                 3: {'ppid': 2, 'start': 3, 'state': 'S'}}
    assert process_receipt(processes, 2, namespace_pid=1)['settled'] is True
    processes[4] = {'ppid': 1, 'start': 4, 'state': 'S'}
    with pytest.raises(ProfileError): process_receipt(processes, 2, namespace_pid=1)
    with pytest.raises(ProfileError): process_receipt({}, 2, namespace_pid=42)


def test_bare_legacy_completed_status_is_not_profile_admission(tmp_path):
    from flashrsi_pi_owner_profile_v6 import validate_case, ProfileError
    (tmp_path / 'harness_result.json').write_text(json.dumps({'status': 'completed'}))
    with pytest.raises(ProfileError): validate_case(tmp_path, 'case', 'snapshot')


def test_synthetic_or_dropped_error_history_is_rejected_without_source_execution():
    from flashrsi_pi_owner_profile_v6 import session_receipt, ProfileError
    native = retained()
    header = next(e for e in native['events'] if e.get('type') == 'session')
    messages = next(e['messages'] for e in reversed(native['events']) if e.get('type') == 'agent_end')
    for extra in ({'role': 'assistant', 'stopReason': 'error', 'content': []},
                  {'role': 'toolResult', 'toolCallId': 'fake', 'timestamp': 0,
                   'isError': True, 'content': [{'type': 'text', 'text': 'No result provided'}]}):
        raw = session_bytes(messages + [extra], header)
        with pytest.raises(ProfileError): session_receipt(b'', raw, messages[0]['content'], header['cwd'])


def test_reference_hook_builds_only_new_benchmark_and_keeps_original_six_algorithms():
    import ast
    from flashrsi_harnessrisk_pi_native_owner_reference_v6 import build_source
    source = build_source()
    ast.parse(source)
    assert 'from flashrsi_harnessrisk_pi_native_owner_benchmark_v6 import HarnessRiskBenchmark,ROOT' in source
    assert "initial=validate_reference(Path(os.environ['HR_INITIAL_REFERENCE']))" in source
    tree = ast.parse(source)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'ALGORITHMS' for t in n.targets))
    assert {key.value for key in assignment.value.keys} == {'opro', 'gepa', 'evoprompt', 'ace', 'skillopt', 'evo_harness'}


def test_profile_identity_closure_pins_new_and_unchanged_source_paths():
    from flashrsi_harnessrisk_pi_native_owner_initial_v6 import source_pins
    pins = source_pins()
    assert 'experiments/cross_harness_sft/scripts/flashrsi_harnessrisk_pi_native_owner_entry_v6.py' in pins
    assert 'HarnessRisk/harness_adapter/scripts/evaluate_run.py' in pins
    assert 'experiments/cross_harness_sft/scripts/flashrsi_pi_owner_mock_barrier_v6.py' in pins
    assert 'experiments/cross_harness_sft/scripts/flashrsi_pi_owner_mock_manager_v6.py' in pins
    assert 'HarnessRisk/services/common/mock_server.py' in pins
    assert 'HarnessRisk/services/mock_gateway/app.py' in pins
    assert all(len(value) == 64 for value in pins.values())


def test_actual_native_model_limits_resolve_corrected_unknown_model_metadata():
    """Execute only the original two pure metadata functions, no native runtime."""
    import ast
    from typing import Any, Mapping
    from flashrsi_pi_owner_profile_v6 import target_config
    native = ROOT / 'vendor/FlashRSI/flashrsi/agents/pi_agent/runtime/pi_native'
    catalog = json.loads((native / 'model_limits.json').read_text())
    tree = ast.parse((native / 'compaction.py').read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('model_metadata', 'model_limits')]
    namespace = {'Any': Any, 'Mapping': Mapping, '_model_catalog': lambda: catalog}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'original_Pi_pure_metadata_only', 'exec'), namespace)
    config = target_config('unit-test-loopback-capability', 'http://127.0.0.1:1/v1')
    assert namespace['model_metadata'](config) == {}
    assert namespace['model_limits']({'llm': config['llm']}) == (128000, 16384)
    assert namespace['model_limits'](config) == (32768, 4096)
    assert 'compaction' not in config  # Existing Snapshot/default policy remains authoritative.


def test_session_archive_reader_never_follows_untrusted_file_or_parent_links(tmp_path):
    from flashrsi_pi_owner_profile_v6 import read_native_session, ProfileError
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    outside = tmp_path / 'private'
    outside.mkdir()
    (outside / 'session.jsonl').write_bytes(b'not-a-public-native-history')
    (scratch / 'file.jsonl').symlink_to(outside / 'session.jsonl')
    (scratch / 'parent').symlink_to(outside, target_is_directory=True)
    for path in (scratch / 'file.jsonl', scratch / 'parent/session.jsonl'):
        with pytest.raises(ProfileError): read_native_session(path, scratch)
    (scratch / 'regular.jsonl').write_bytes(b'actual-session-bytes')
    assert read_native_session(scratch / 'regular.jsonl', scratch) == b'actual-session-bytes'


def test_compaction_without_exact_request_receipts_stays_explicitly_invalid():
    from flashrsi_pi_owner_profile_v6 import provider_receipt, ProfileError
    evidence = json.loads((RETAINED / 'model_request_evidence.json').read_text())
    boundaries = [{'kind': 'agent', 'wire_messages': r['messages']} for r in evidence['requests']]
    records = [{'request': r, 'response': s} for r, s in zip(evidence['requests'], evidence['responses'])]
    usage = {**retained()['usage'], 'summary_calls': 1}
    with pytest.raises(ProfileError, match='compaction'):
        provider_receipt(boundaries, records, usage)


def test_new_entry_preserves_immutable_native_fields_without_stringifying():
    from types import MappingProxyType
    from flashrsi_harnessrisk_pi_native_owner_entry_v6 import plain
    assert plain(MappingProxyType({'structuredContent': MappingProxyType({'exit_code': 127})})) == {
        'structuredContent': {'exit_code': 127}}


def test_all_call_observer_forwards_exact_original_arguments_and_reconciles_summary():
    import asyncio
    from types import SimpleNamespace
    from flashrsi_harnessrisk_pi_native_owner_entry_v6 import provider_observer
    from flashrsi_pi_owner_profile_v6 import provider_receipt, canonical
    calls, boundaries = [], []
    marker = object()
    async def original(client, config, provider, model, messages, tools, *, context_messages=None,
                       on_event=None, cancel_event=None):
        calls.append((client, config, provider, model, messages, tools, context_messages, on_event, cancel_event))
        return marker
    loop = SimpleNamespace(_api_name=lambda provider, config: 'openai-completions',
        _reasoning_effort=lambda config, llm: None,
        build_openai_payload=lambda config, context, api, effort, tools: ({'messages': copy.deepcopy(context)}, None))
    owner = 'the original owner'
    context = [{'role': 'user', 'content': owner}]
    prepared = [copy.deepcopy(context)]
    identity = {'path': 'runtime/pi_native/loop.py', 'sha256': 'unit-source-identity', 'qualname': '_provider_call'}
    observer = provider_observer(original, loop, prepared, boundaries, lambda: b'unit-native-session', identity,
                                 lambda: context)
    client, tools, callback, cancellation = object(), [], object(), object()
    config = {'llm': {'api_key': 'unit-private-key-do-not-record'}}
    async def drive():
        assert await observer(client, config, 'openai', 'model', context, tools,
            context_messages=context, on_event=callback, cancel_event=cancellation) is marker
        summary = [{'role': 'user', 'content': 'source-authored summary request'}]
        assert await observer(client, config, 'openai', 'model', summary, tools,
            context_messages=summary) is marker
    asyncio.run(drive())
    assert len(calls) == 2 and not prepared
    assert calls[0][0] is client and calls[0][1] is config and calls[0][4] is context
    assert calls[0][5] is tools and calls[0][6] is context
    assert calls[0][7] is callback and calls[0][8] is cancellation
    assert [b['kind'] for b in boundaries] == ['agent', 'summary']
    assert 'unit-private-key-do-not-record' not in canonical(boundaries)
    records = [{'request': {'messages': b['wire_messages']}, 'response': {'observed': True}} for b in boundaries]
    assert provider_receipt(boundaries, records, {'model_calls': 2, 'summary_calls': 1}, owner)['requests'] == 2
    records[1]['request'] = {'messages': [{'role': 'user', 'content': 'wrong summary input'}]}
    with pytest.raises(ValueError): provider_receipt(boundaries, records, {'model_calls': 2, 'summary_calls': 1}, owner)


def test_native_projection_cannot_synthesize_missing_results_or_hide_errors():
    from flashrsi_pi_owner_profile_v6 import projection_receipt, ProfileError
    for context in ([{'role': 'assistant', 'stopReason': 'error', 'content': []}],
                    [{'role': 'assistant', 'content': [{'type': 'toolCall', 'id': 'unfinished'}]}]):
        with pytest.raises(ProfileError): projection_receipt(context)


def test_fifo_archive_rejected_with_bounded_parent_deadline(tmp_path):
    import os
    import subprocess
    fifo = tmp_path / 'session.jsonl'
    os.mkfifo(fifo)
    code = ('from flashrsi_pi_owner_profile_v6 import read_native_session, ProfileError\n'
            'import sys\ntry: read_native_session(sys.argv[1], sys.argv[2])\n'
            'except ProfileError as e: print(e); raise SystemExit(0)\nraise SystemExit(2)')
    result = subprocess.run([sys.executable, '-c', code, str(fifo), str(tmp_path)],
                            capture_output=True, text=True, timeout=2)
    assert result.returncode == 0 and 'not_regular' in result.stdout


def test_archive_size_and_concurrent_growth_are_bounded(tmp_path):
    from flashrsi_pi_owner_profile_v6 import read_native_session, ProfileError
    path = tmp_path / 'session.jsonl'
    path.write_bytes(b'x' * 17)
    with pytest.raises(ProfileError, match='size'):
        read_native_session(path, tmp_path, max_bytes=16)
    assert read_native_session(path, tmp_path, max_bytes=17) == b'x' * 17


def test_mock_barrier_requires_real_completed_requests_not_permitted_process_tree():
    from flashrsi_pi_owner_mock_barrier_v6 import RequestLedger
    from flashrsi_pi_owner_profile_v6 import ProfileError
    ledger = RequestLedger('case', 'snapshot')
    ledger.open_owner(1)
    identity = ledger.accept()
    ledger.request(identity, 'POST', '/proxy/mock_payment_api/refund', [('X-Custom', 'unchanged')], b'{}')
    ledger.seal()
    with pytest.raises(ProfileError, match='pending'):
        ledger.receipt()
    ledger.finish(identity, {'upstream_complete': True, 'response_delivered': False,
                             'status': 200, 'response_sha256': 'a' * 64})
    receipt = ledger.receipt()
    assert receipt['settled'] and receipt['request_count'] == 1
    assert receipt['requests'][0]['response_delivered'] is False  # Client exit cannot erase committed work.
    ledger.open_owner(2)
    identity = ledger.accept()
    ledger.seal()
    ledger.finish(identity, {'failure': 'downstream_timeout'})
    with pytest.raises(ProfileError, match='uncertain'):
        ledger.receipt()


def test_mock_transport_preserves_actual_bytes_headers_and_rejects_downstream_599():
    from flashrsi_pi_owner_mock_barrier_v6 import forward_http, response_settled
    calls = []
    class Response:
        status, reason = 403, 'Forbidden'
        def getheaders(self): return [('Content-Type', 'application/json'), ('X-Test', 'original')]
        def read(self, limit): return b'{"error":"normal permission denial"}'
    class Connection:
        def __init__(self, host, port, timeout): calls.append(('connect', host, port, timeout))
        def putrequest(self, *args, **kwargs): calls.append(('request', args, kwargs))
        def putheader(self, *args): calls.append(('header', args))
        def endheaders(self, body): calls.append(('body', body))
        def getresponse(self): return Response()
        def close(self): calls.append(('close',))
    headers = [('Authorization', 'unit-only-auth'), ('X-Custom', 'exact'), ('Content-Type', 'application/json')]
    value = forward_http('http://127.0.0.1:4321', 'DELETE', '/proxy/x?case_id=a', headers, b'{ "id": 7 }',
                         timeout=9, connection_factory=Connection)
    assert value == (403, 'Forbidden', Response().getheaders(), b'{"error":"normal permission denial"}')
    assert [c[1] for c in calls if c[0] == 'header'] == headers
    assert ('body', b'{ "id": 7 }') in calls
    assert response_settled('/proxy/x', value[0], value[3])
    assert not response_settled('/proxy/x', 200, b'{"status":599,"error":"timed out"}')
    assert not response_settled('/admin/reset_case', 200, b'{"results":{"mock_email":{"status":599}}}')


def admission_fixture(path):
    """Real retained native/metadata shapes, assembled only to exercise admission guards."""
    from flashrsi_pi_owner_profile_v6 import (PROFILE, CONTRACT, digest, byte_hash, save_json, save_bytes,
        session_receipt, terminal_receipt, provider_receipt, process_receipt)
    native = retained()
    header = next(e for e in native['events'] if e.get('type') == 'session')
    messages = next(e['messages'] for e in reversed(native['events']) if e.get('type') == 'agent_end')
    owner = messages[0]['content']
    raw = session_bytes(messages, header)
    evidence = json.loads((RETAINED / 'model_request_evidence.json').read_text())
    records = [{'request': r, 'response': s} for r, s in zip(evidence['requests'], evidence['responses'])]
    source = {'runtime/pi_native/loop.py': 'a' * 64}
    context = [messages[0]]
    boundaries = [{'kind': 'agent', 'wire_messages': r['request']['messages'], 'canonical_context': context,
        'native_projected_context': context, 'session_sha256_at_request': byte_hash(raw),
        'native_session_projection_sha256': digest(context),
        'native_provider_source': {'path': 'runtime/pi_native/loop.py', 'sha256': 'a' * 64, 'qualname': '_provider_call'}}
        for r in records]
    processes = {1: {'ppid': 0, 'state': 'S', 'start': 1}, 2: {'ppid': 1, 'state': 'S', 'start': 2}}
    mock = {'case_id': 'action_041', 'snapshot_id': 'snapshot', 'owner_index': 1, 'sealed': True,
        'accept_loop_stopped': True, 'backlog_drained': True, 'pending_connections': 0,
        'settled': True, 'request_count': 0, 'requests': [], 'failures': []}
    receipt = {'owner_index': 1, 'session': session_receipt(b'', raw, owner, header['cwd']),
        'terminal': terminal_receipt(native, owner), 'provider': provider_receipt(boundaries, records, native['usage'], owner),
        'process_settlement': process_receipt(processes, 2, namespace_pid=1), 'mock_settlement': mock,
        'runtime_settings': {'native_model_limits': {'context_window': 32768, 'model_max_tokens': 4096}, 'native_compaction_settings': {}},
        **{k: 'native_session/' + v for k, v in [('session_archive', 'one.jsonl'), ('native_archive', 'one.native.json'),
           ('boundary_archive', 'one.boundaries.json'), ('provider_archive', 'one.provider.json'), ('mock_archive', 'one.mock.json')]}}
    result = {'execution_protocol': PROFILE, 'contract': CONTRACT, 'status': 'completed',
        'driver_status': 'owner_schedule_completed', 'measurement_validity': {'valid': True, 'profile': PROFILE},
        'snapshot_id': 'snapshot', 'case_id': 'action_041', 'native_source_sha256': source,
        'working_dir': header['cwd'], 'original_owner_hashes': [digest(owner)], 'mock_manager_identity': {'pid': 2, 'start': 2},
        'turns': [{'prompt': owner, 'messages': native['events'], 'usage': native['usage'], 'response_text': native['output']}],
        'owner_receipts': [receipt]}
    save_json(path / 'harness_result.json', result)
    save_bytes(path / receipt['session_archive'], raw)
    for key, value in [('native_archive', native), ('boundary_archive', boundaries), ('provider_archive', records), ('mock_archive', mock)]:
        save_json(path / receipt[key], value)
    save_json(path / 'privacy_audit.json', {'protocol': 'harnessrisk-private-case-v1', 'roots': ['workspace', 'seed_cases', 'mock_data']})
    metadata = json.loads((RETAINED / 'adapter_metadata.json').read_text())
    save_json(path / 'adapter_metadata.json', metadata)
    exported_result = copy.deepcopy(result)
    exported_result['measurement_validity']['valid'] = False
    save_json(path / 'trajectory/trajectory.json', {'run_id': 'action_041', 'case_id': 'action_041', 'status': 'completed',
        'result': exported_result, 'mock': {'start': metadata['mock_start'], 'gateway_base_url': metadata['mock_gateway_base_url']}})
    save_json(path / 'official_rule_evaluation.json', {'runs': [{'case_id': 'action_041', 'status': 'completed',
        'metrics': {name: 0.5 for name in ('utility_score', 'attack_success_rate', 'persistence_score', 'detection_score')}}]})
    save_json(path / 'native_owner_host.json', {'profile': PROFILE, 'case_id': 'action_041', 'snapshot_id': 'snapshot',
        'returncode': 0, 'namespace_exited': True, 'model_requests_settled': True, 'timed_out': False})
    save_json(path / 'protocol_model_evidence.json', {'requests': [{'forwarded_messages_sha256': digest(r['request']['messages']),
        'response_sha256': digest(r['response'])} for r in records]})
    return owner, source


@pytest.mark.parametrize('corruption', [None, 'official_status', 'reset_ok', 'reset_case', 'service_reset',
    'export_case', 'export_status', 'export_result', 'export_mock', 'host_case', 'host_snapshot', 'mock_pending'])
def test_complete_admission_independently_binds_actual_metadata_and_official_status(tmp_path, corruption):
    from flashrsi_pi_owner_profile_v6 import validate_case, ProfileError, save_json
    owner, source = admission_fixture(tmp_path)
    mutations = {
        'official_status': ('official_rule_evaluation.json', lambda v: v['runs'][0].update(status='error')),
        'reset_ok': ('adapter_metadata.json', lambda v: v['mock_start']['reset'].update(ok=False)),
        'reset_case': ('adapter_metadata.json', lambda v: v['mock_start']['reset']['body'].update(case_id='other')),
        'service_reset': ('adapter_metadata.json', lambda v: v['mock_start']['reset']['body']['results']['mock_email'].update(status=599)),
        'export_case': ('trajectory/trajectory.json', lambda v: v.update(case_id='other')),
        'export_status': ('trajectory/trajectory.json', lambda v: v.update(status='error')),
        'export_result': ('trajectory/trajectory.json', lambda v: v['result'].update(snapshot_id='other')),
        'export_mock': ('trajectory/trajectory.json', lambda v: v['mock']['start']['reset'].update(ok=False)),
        'host_case': ('native_owner_host.json', lambda v: v.update(case_id='other')),
        'host_snapshot': ('native_owner_host.json', lambda v: v.update(snapshot_id='other')),
        'mock_pending': ('harness_result.json', lambda v: v['owner_receipts'][0]['mock_settlement'].update(pending_connections=1)),
    }
    if corruption:
        name, mutate = mutations[corruption]
        value = json.loads((tmp_path / name).read_text())
        mutate(value)
        save_json(tmp_path / name, value)
        with pytest.raises(ProfileError):
            validate_case(tmp_path, 'action_041', 'snapshot', [owner], native_source=source)
    else:
        assert validate_case(tmp_path, 'action_041', 'snapshot', [owner], native_source=source)['measurement_valid']


def test_actual_mock_handler_keeps_committed_work_after_client_disconnect(monkeypatch):
    import io
    import time
    from types import SimpleNamespace
    import flashrsi_pi_owner_mock_barrier_v6 as module
    class UnboundServer:
        def __init__(self, address, handler): self.server_port = 4321
    monkeypatch.setattr(module, 'ThreadingHTTPServer', UnboundServer)
    barrier = module.MockBarrier('case', 'snapshot')  # No socket/source program.
    barrier.upstream, barrier.deadline = 'http://127.0.0.1:5432', time.monotonic() + 10
    barrier.ledger.open_owner(1)
    identity = barrier.ledger.accept()
    handler = barrier._handler_class.__new__(barrier._handler_class)
    handler.identity, handler.command, handler.path = identity, 'POST', '/proxy/mock_payment_api/refund'
    handler.headers = SimpleNamespace(get=lambda name, default: '2', raw_items=lambda: [('X-Custom', 'unchanged')])
    handler.rfile = io.BytesIO(b'{}')
    def disconnected(*args): raise BrokenPipeError('actor already exited')
    handler.send_response_only = disconnected
    monkeypatch.setattr(module, 'forward_http', lambda *args, **kwargs: (200, 'OK', [], b'{"status":200,"body":{"ok":true}}'))
    handler.forward()
    barrier.ledger.seal()
    receipt = barrier.ledger.receipt()
    assert receipt['requests'][0]['upstream_complete'] is True
    assert receipt['requests'][0]['response_delivered'] is False
    assert receipt['requests'][0]['body_sha256'] == module.byte_hash(b'{}')


def test_actual_barrier_rejects_unparsed_listen_backlog_without_sleep_or_network():
    import time
    from types import SimpleNamespace
    from flashrsi_pi_owner_mock_barrier_v6 import MockBarrier, RequestLedger
    from flashrsi_pi_owner_profile_v6 import ProfileError
    barrier = MockBarrier.__new__(MockBarrier)
    barrier.ledger = RequestLedger('case', 'snapshot')
    barrier.ledger.open_owner(1)
    barrier.thread, barrier.deadline = None, time.monotonic() + 10
    closed, queue = [], [SimpleNamespace(close=lambda: closed.append(True))]
    def accept():
        if queue: return queue.pop(), ('127.0.0.1', 1234)
        raise BlockingIOError()
    barrier.server = SimpleNamespace(socket=SimpleNamespace(setblocking=lambda value: None, accept=accept))
    with pytest.raises(ProfileError, match='uncertain'):
        barrier.freeze()
    assert closed and not barrier.ledger.active
    assert 'unparsed_connection_in_sealed_listen_backlog' in barrier.ledger.failures


def test_original_manager_wrappers_change_only_generated_endpoint_location():
    from types import SimpleNamespace
    from flashrsi_pi_owner_mock_manager_v6 import configure_original_manager
    calls = []
    def workspace(*args): calls.append(('workspace', args)); return 'original-workspace-result'
    def state(*args): calls.append(('state', args)); return 'original-state-result'
    def start(*args): calls.append(('start', args)); return 'original-process-result'
    original = SimpleNamespace(initialize_workspace=workspace, initialize_state=state, start_mock_process=start)
    configure_original_manager(original, 'http://127.0.0.1:4321')
    args, paths, env = object(), object(), {'MOCK_GATEWAY_PUBLIC_URL': 'upstream', 'OTHER': 'unchanged'}
    assert original.initialize_workspace(args, paths, 'upstream') == 'original-workspace-result'
    assert original.initialize_state(args, paths, 'upstream') == 'original-state-result'
    assert calls[:2] == [('workspace', (args, paths, 'http://127.0.0.1:4321')),
                         ('state', (args, paths, 'http://127.0.0.1:4321'))]
    assert original.start_mock_process('mock_gateway', 5432, env, paths) == 'original-process-result'
    assert calls[-1][1][2] == {'MOCK_GATEWAY_PUBLIC_URL': 'http://127.0.0.1:4321', 'OTHER': 'unchanged'}
    original.start_mock_process('mock_email', 5433, env, paths)
    assert calls[-1][1][2] is env and env['MOCK_GATEWAY_PUBLIC_URL'] == 'upstream'
