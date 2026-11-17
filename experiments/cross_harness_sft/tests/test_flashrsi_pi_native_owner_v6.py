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
