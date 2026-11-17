"""NEW Claude native-owner receipts. Unknown native state is never completion."""
from __future__ import annotations

import base64
import copy
import json
import math
from pathlib import Path
import uuid

# Reviewed generic filesystem/durability/census helpers only. Their source bytes
# are included in this profile's full source map; no Pi run/adapter is invoked.
from flashrsi_pi_owner_profile_v6 import (ProfileError, byte_hash, canonical, digest,
    save_json, save_bytes, read_native_session, process_receipt, _inside, _unique)

PROFILE = 'private-v1-claude-native-owner-complete-v9'
MODEL = 'qwen35-4b-preflight'
CLI = '/data/home/liumingxiao/.npm-global/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe'
CONTRACT = {'profile': PROFILE, 'max_turns_per_owner': 16, 'native_timeout_seconds': 600,
    'sandbox_wall_seconds': 900, 'whole_host_seconds': 2400, 'max_tokens_per_turn': 4096,
    'temperature': 0, 'max_retries': 0, 'context_window': 32768, 'native_auto_compact_enabled': True,
    'native_context_correction': 'CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768; native compaction policy unchanged',
    'owner_input': 'exact next original owner once; explicit UUID resume, same cwd/Snapshot/mock/state',
    'status_semantics': 'completed means measured owner schedule, never agent task success',
    'source_status_ruling': 'root 2026-11-17: completed only after all native/history/privacy/source gates; preserve native errors',
    'unknown_native_projection': 'invalid; native compaction requires complete original projection evidence',
    'counter_rule': 'actual fresh owner request required; native reported max-turn count is not tool/model-call count',
    'resumed_counter_reset': 'actual native decrease or sixteen distinct successful scheduled pairs reconciled to raw/native origin; otherwise invalid',
    'native_auxiliary_accounting': 'all requests retained; exact original native session-title prompt separated from scheduled exposure'}


def target_config(api_key, base_url):
    return {'llm': {'model_name': MODEL, 'api_key': api_key, 'base_url': base_url,
                   'max_tokens': 4096, 'max_retries': 0}, 'cli_path': CLI,
            'allowed_tools': ['Read','Glob','Grep','Bash','Write','Edit'],
            'permission_mode':'dontAsk','max_turns':16,'timeout':600,
            'tools':['Read','Glob','Grep','Bash','Write','Edit','Skill']}


def snapshot_source_map(snapshot):
    files = snapshot['files'] if isinstance(snapshot, dict) else snapshot.files
    return {name: byte_hash(base64.b64decode(entry['content'], validate=True)
                           if isinstance(entry, dict) else entry.content)
            for name, entry in files.items()}


def valid_uuid(value):
    try:
        if str(uuid.UUID(value)) != value: raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ProfileError('actual_native_UUID_missing_or_invalid') from None
    return value


def context_receipt(value):
    if (not isinstance(value, dict) or value.get('rawMaxTokens') != 32768
            or type(value.get('maxTokens')) is not int or not 0 < value['maxTokens'] <= 32768
            or value.get('model') != MODEL or value.get('isAutoCompactEnabled') is not CONTRACT['native_auto_compact_enabled']):
        raise ProfileError('actual_native_context_usage_unknown_or_mismatched')
    return copy.deepcopy(value)


def close_receipt(value):
    events = value.get('events', [])
    required = {'begin','lock_acquired','stdin_closed','graceful_wait','graceful_reaped','reaped','finished'}
    if (not required.issubset(events) or set(events) & {'terminate','kill','cancelled','lock_timeout','already_exited'}
            or value.get('exception') is not None or value.get('returncode') != 0
            or value.get('reaped') is not True or type(value.get('pid')) is not int
            or type(value.get('start')) is not int):
        raise ProfileError('native_clean_EOF_reap_and_flush_unproven')
    return copy.deepcopy(value)


def terminal_receipt(frames, stops, close, session_id):
    valid_uuid(session_id)
    close_receipt(close)
    results = [e for e in frames if e.get('type') == 'result']
    human = [r for r in results if r.get('origin') is None or r.get('origin', {}).get('kind') == 'human']
    if len(human) != 1 or len(results) != 1:
        raise ProfileError('scheduled_owner_native_result_unknown_or_background_turn')
    result = human[0]
    bounded = (result.get('subtype') == 'error_max_turns' and result.get('terminal_reason') == 'max_turns'
               and result.get('is_error') is True)
    normal = (result.get('subtype') == 'success' and result.get('terminal_reason') == 'completed'
              and result.get('is_error') is False)
    # Retained native cap has num_turns=17 under --max-turns16. These are native
    # loop iterations, not tools or provider calls. Preserve the actual count.
    count = result.get('num_turns')
    if (not (bounded or normal) or type(count) is not int or not 1 <= count <= 17
            or bounded and count != 17 or result.get('session_id') != session_id
            or result.get('api_error_status') is not None or result.get('deferred_tool_use') is not None
            or result.get('errors') not in ((None, [], ['Reached maximum number of turns (16)']) if bounded else (None, []))
            or result.get('result') is not None and not isinstance(result['result'], str)):
        raise ProfileError('native_terminal_not_exact_normal_or_cap')
    states = [f for f in frames if f.get('type') == 'system' and f.get('subtype') == 'session_state_changed']
    if not states or states[-1].get('state') != 'idle' or states[-1].get('session_id') != session_id:
        raise ProfileError('actual_native_idle_missing_or_wrong_session')
    if not stops or stops[-1].get('session_id') != session_id or stops[-1].get('hook_event_name') != 'Stop':
        raise ProfileError('actual_native_Stop_missing_or_wrong_session')
    if stops[-1].get('background_tasks') != [] or stops[-1].get('session_crons') != []:
        raise ProfileError('actual_native_background_or_cron_closure_unknown')
    active, hooks, calls, outputs = set(), set(), [], []
    for frame in frames:
        if frame.get('type') == 'system':
            subtype = frame.get('subtype')
            if subtype == 'init':
                if frame.get('session_id') != session_id:
                    raise ProfileError('native_init_session_changed')
                if frame.get('plugin_errors') or any(m.get('status') in {'failed','needs-auth'} for m in frame.get('mcp_servers', [])):
                    raise ProfileError('native_extension_or_MCP_failed')
            if subtype == 'task_started':
                if frame.get('task_type') not in {'local_agent','local_workflow','local_shell','shell','monitor','workflow'}:
                    raise ProfileError('unknown_native_task_type')
                if not frame.get('task_id') or frame['task_id'] in active:
                    raise ProfileError('missing_or_duplicate_native_task_start')
                active.add(frame['task_id'])
            if subtype in {'task_notification','task_updated'}:
                if frame.get('status') in {'stopped','killed','aborted','failed'}:
                    raise ProfileError('native_task_aborted_or_failed')
                if frame.get('status') == 'completed':
                    if frame.get('task_id') not in active:
                        raise ProfileError('native_task_terminal_without_start')
                    active.remove(frame['task_id'])
            if subtype in {'hook_started','hook_response'}:
                identity = frame.get('hook_id')
                if not identity: raise ProfileError('native_hook_lifecycle_identity_missing')
                if subtype == 'hook_started':
                    if identity in hooks: raise ProfileError('duplicate_native_hook_start')
                    hooks.add(identity)
                else:
                    if identity not in hooks or frame.get('exit_code') != 0 or frame.get('outcome') not in {'success','completed'}:
                        raise ProfileError('native_hook_failed_unknown_or_unmatched')
                    hooks.remove(identity)
        message = frame.get('message', {})
        if frame.get('type') == 'assistant' and (frame.get('error') or message.get('error')):
            raise ProfileError('native_assistant_API_or_context_failure')
        content = message.get('content', [])
        for block in content if isinstance(content, list) else []:
            if block.get('type') == 'tool_use': calls.append(block.get('id'))
            if block.get('type') == 'tool_result': outputs.append(block.get('tool_use_id'))
    if (active or hooks or None in calls or None in outputs or len(calls) != len(set(calls))
            or len(outputs) != len(set(outputs)) or set(calls) != set(outputs)):
        raise ProfileError('native_tools_tasks_or_hooks_not_settled')
    return {'kind':'native_max_turns' if bounded else 'native_finished',
            'native_outcome':copy.deepcopy(result), 'tool_ids':calls, 'raw_frames_sha256':digest(frames),
            'Stop_sha256':digest(stops), 'close_sha256':digest(close)}


def session_rows(raw):
    if not raw or not raw.endswith(b'\n'):
        raise ProfileError('native_session_missing_or_partial_JSONL')
    try:
        rows = [json.loads(line, object_pairs_hook=_unique) for line in raw.decode('utf-8').splitlines()]
    except (UnicodeError, ValueError) as exc:
        raise ProfileError('native_session_corrupt') from exc
    if any(not isinstance(r, dict) for r in rows): raise ProfileError('native_session_nonobject_record')
    return rows


def session_receipt(before, after, owner, cwd, expected_id):
    valid_uuid(expected_id)
    if not after.startswith(before): raise ProfileError('native_session_prefix_mutated')
    rows = session_rows(after)
    old_count = len(session_rows(before)) if before else 0
    previous, identities, calls, results = None, set(), [], []
    for row in rows:
        if row.get('subtype') == 'compact_boundary' or row.get('isCompactSummary'):
            # This CLI's provider-compaction/branch reconstruction is compiled.
            # No inferred summary receipt can certify what history was retained.
            raise ProfileError('native_compaction_projection_structural_barrier_requires_original_receipt')
        if row.get('type') not in {'user','assistant','progress','system','attachment'}:
            continue  # Preserve raw metadata/header bytes; not conversation links.
        if row.get('sessionId') != expected_id or row.get('cwd') != cwd or row.get('isSidechain'):
            raise ProfileError('native_session_identity_cwd_or_sidechain_changed')
        identity = row.get('uuid')
        if not identity or identity in identities or row.get('parentUuid') not in identities | {None}:
            raise ProfileError('native_session_duplicate_or_dangling_parent')
        if previous is not None and row.get('parentUuid') is None:
            raise ProfileError('native_session_unexplained_new_root')
        identities.add(identity); previous = identity
        content = row.get('message', {}).get('content', [])
        for block in content if isinstance(content, list) else []:
            if block.get('type') == 'tool_use': calls.append(block.get('id'))
            if block.get('type') == 'tool_result': results.append(block.get('tool_use_id'))
    if (None in calls or None in results or len(calls)!=len(set(calls)) or len(results)!=len(set(results)) or set(calls)!=set(results)):
        raise ProfileError('native_session_pending_or_duplicate_tool_result')
    new_owners = [r for r in rows[old_count:] if r.get('type') == 'user' and not r.get('isMeta')
                  and (isinstance(r.get('message', {}).get('content'), str) or
                       isinstance(r.get('message', {}).get('content'), list) and
                       all(b.get('type')=='text' for b in r['message']['content']))]
    if len(new_owners) != 1 or _message(new_owners[0]['message']) != _message({'role':'user','content':owner}):
        raise ProfileError('next_original_owner_not_appended_exactly_once')
    return {'session_id':expected_id, 'cwd':cwd, 'leaf_id':previous,
        'owner_uuid':new_owners[0]['uuid'],'prefix_sha256':byte_hash(before),'session_sha256':byte_hash(after),
        'bytes':len(after),'records':len(rows),'owner_sha256':digest(owner)}


_AUTO_PREFIXES = ("<system-reminder>\nAs you answer the user's questions, you can use the following context:\n# gitStatus\n",
                  '<system-reminder>\nAttribution for git commits and pull requests you create from here on ')

def project_cli_history(body):
    """Only original v8 known CLI reminders; never remove matching owner text."""
    projected, removed = copy.deepcopy(body), []
    for mi, message in enumerate(projected.get('messages', [])):
        if message.get('role') != 'user' or not isinstance(message.get('content'), list): continue
        kept=[]
        for bi, block in enumerate(message['content']):
            if block.get('type') == 'text' and isinstance(block.get('text'),str) and block['text'].startswith(_AUTO_PREFIXES):
                removed.append({'message_index':mi,'part_index':bi,'sha256':byte_hash(block['text'].encode()),'kind':'cli_system_reminder'})
            else: kept.append(block)
        message['content']=kept
    return projected,removed


def _message(value):
    """Normalize only SDK/native plain-text representation and cache annotation."""
    role, content = value.get('role'), value.get('content')
    if isinstance(content,str): content=[{'type':'text','text':content}]
    if not isinstance(content,list):raise ProfileError('native_history_content_unknown')
    content=[{k:v for k,v in b.items() if k!='cache_control'} for b in content]
    return {'role':role,'content':content}


def provider_receipt(records, before, owner, session_id, owner_index, *, after=None):
    if not records:raise ProfileError('original_owner_no_actual_model_invocation')
    model_records=[];auxiliary_records=[];kinds=[]
    for record in records:
        if (record.get('failure') or record.get('status') != 200 or not record.get('response_sha256')
                or record.get('settled') is False):
            raise ProfileError('actual_model_transport_or_context_failure')
        request=record.get('request',{})
        if request.get('model') != MODEL:raise ProfileError('actual_model_identity_changed')
        metadata=request.get('metadata',{})
        if metadata.get('user_id') is not None:
            try:native_session=json.loads(metadata['user_id'])['session_id']
            except (TypeError,ValueError,KeyError):raise ProfileError('native_wire_session_metadata_unknown') from None
            if native_session!=session_id:raise ProfileError('native_wire_session_UUID_changed')
        if record.get('path') == '/v1/messages':
            if type(request.get('max_tokens')) is not int or not 1<=request['max_tokens']<=4096:
                raise ProfileError('actual_wire_output_budget_changed')
            kind=native_request_kind(request)
            kinds.append(kind)
            (auxiliary_records if kind=='session_title' else model_records).append(record)
        elif record.get('path') != '/v1/messages/count_tokens':
            raise ProfileError('unexpected_native_provider_route')
    if not model_records:raise ProfileError('original_owner_no_actual_model_invocation')
    history_hashes=[digest(r['request'].get('messages')) for r in model_records]
    if len(history_hashes)!=len(set(history_hashes)):
        raise ProfileError('repeated_native_agent_history_is_retry_or_unproved_origin')
    first=model_records[0]['request']['messages']
    native_before=[_message(r['message']) for r in session_rows(before) if r.get('type') in {'user','assistant'} and not r.get('isMeta')] if before else []
    expected=native_before+[_message({'role':'user','content':owner})]
    actual=[_message(m) for m in first]
    if actual != expected:
        raise ProfileError('native_history_projection_unknown_or_owner_duplicated_removed')
    if after is not None:
        native_after=[_message(r['message']) for r in session_rows(after) if r.get('type') in {'user','assistant'} and not r.get('isMeta')]
        for record in model_records:
            history=[_message(m) for m in record['request']['messages']]
            if len(history)<len(expected) or history!=native_after[:len(history)] or history[:len(expected)]!=expected:
                raise ProfileError('actual_provider_history_prefix_dropped_duplicated_or_unarchived')
    # Actual requests, summaries and failed attempts are preserved, never counted
    # as tool iterations. CLI result.num_turns is reconciled separately.
    return {'owner_index':owner_index,'session_id':session_id,'owner_sha256':digest(owner),
        'model_requests':len(model_records)+len(auxiliary_records),'agent_model_requests':len(model_records),
        'auxiliary_model_requests':len(auxiliary_records),'request_kinds':kinds,'requests_sha256':digest(records),
        'first_history_sha256':digest(actual),'prior_session_sha256':byte_hash(before)}


def counter_receipt(terminal, provider, previous=None):
    result=terminal['native_outcome']; count=result['num_turns']; calls=provider.get('agent_model_requests',provider['model_requests'])
    if not 1<=calls<=count or not 1<=count<=17:
        raise ProfileError('native_counter_or_actual_invocations_inconsistent')
    reset_witness=None
    if previous and previous['kind']=='native_max_turns':
        if count<previous['native_outcome']['num_turns']:
            reset_witness='actual_native_counter_decrease_and_new_owner_request'
        elif provider.get('distinct_scheduled_pairs',0)>=16 and provider.get('scheduled_origin_reconciled') is True:
            reset_witness='sixteen_distinct_successful_scheduled_pairs_with_native_origin'
        else:
            raise ProfileError('native_resumed_counter_reset_structural_barrier')
    # After a saturated prior query, a query-local count <=17 together with a
    # genuine next-owner provider request proves forward exposure. If no such
    # owner is reached, the empirical reset remains unproven in the run summary.
    return {'native_num_turns':count,'declared_max_turns':16,'actual_model_requests':calls,
        'prior_native_num_turns':previous['native_outcome']['num_turns'] if previous else None,
        'resumed_after_cap_observed':bool(previous and previous['kind']=='native_max_turns'),
        'native_counter_reset_observed':reset_witness is not None,'reset_witness':reset_witness,
        'auxiliary_model_requests':provider.get('auxiliary_model_requests',0),
        'query_counter_rule':'distinct native result, agent API, auxiliary API and tool counters; no equality inference; cap17 under declared16'}


def locate_session(config_dir, scratch, session_id):
    valid_uuid(session_id)
    # Do not construct or guess native cwd sanitization. Search only the native
    # projects tree; every actual read uses no-follow descriptor traversal.
    candidates=list((Path(config_dir)/'projects').glob('*/'+session_id+'.jsonl'))
    if len(candidates)!=1:raise ProfileError('native_session_file_missing_or_ambiguous')
    raw=read_native_session(candidates[0],scratch)
    return candidates[0],raw


def validate_owner(output, before, after, owner, index, session_id, records, cwd, previous=None, *, trusted_identity=None):
    if (output.get('profile')!=PROFILE or output.get('owner_index')!=index
            or output.get('owner_sha256')!=digest(owner) or output.get('resume_requested')!=session_id
            or output.get('session_before_sha256')!=byte_hash(before) or output.get('error') is not None):
        raise ProfileError('native_entry_identity_resume_or_execution_failure')
    from flashrsi_claude_owner_identity_v9 import validate_identity
    validate_identity(trusted_identity,output['close'],binding=output.get('identity_binding'),expected_executable=CLI)
    if not output.get('identity_binding') or output['identity_binding'].get('owner_index')!=index:
        raise ProfileError('trusted_identity_owner_binding_missing')
    sid=output.get('session_id');valid_uuid(sid)
    if session_id is not None and sid!=session_id:raise ProfileError('native_resume_forked_or_wrong_UUID')
    settings=output.get('runtime_settings',{})
    if (settings.get('profile')!=PROFILE or settings.get('contract')!=CONTRACT or settings.get('cwd')!=cwd
            or settings.get('resume')!=session_id or settings.get('continue_conversation') is not False
            or settings.get('fork_session') is not False or any(settings.get(k) is not None for k in
                ('session_id','resume_session_at','resume_drops_turn','session_store'))
            or settings.get('environment',{}).get('CLAUDE_CODE_MAX_CONTEXT_TOKENS')!='32768'
            or settings.get('environment',{}).get('CLAUDE_CODE_MAX_OUTPUT_TOKENS')!='4096'):
        raise ProfileError('native_effective_options_context_or_resume_changed')
    validate_candidate_policy(settings.get('settings',{}))
    if settings.get('environment',{}).get('CLAUDE_CODE_MAX_RETRIES')!='0':
        raise ProfileError('native_effective_retry_policy_changed')
    session=session_receipt(before,after,owner,cwd,sid)
    terminal=terminal_receipt(output['raw_frames'],output['Stop_inputs'],output['close'],sid)
    reconcile_result(terminal['native_outcome'],output.get('native_result'))
    if output.get('output') != (terminal['native_outcome'].get('result') or ''):
        raise ProfileError('actual_answer_changed_or_synthesized')
    context=context_receipt(output.get('context_usage'))
    projected_records=[]
    for record in records:
        projected,removed=project_cli_history(record['request'])
        projected_records.append(dict(record,request=projected,raw_request_sha256=digest(record['request']),projection_removed=removed))
    tools=tool_execution_receipt(output.get('tool_hook_inputs'),terminal['tool_ids'],sid)
    history=history_frame_receipt(output['raw_frames'],before,after)
    provider=provider_receipt(projected_records,before,owner,sid,index,after=after)
    provider.update(agent_origin_receipt(projected_records,output['raw_frames'],after))
    counter=counter_receipt(terminal,provider,previous)
    return {'session':session,'terminal':terminal,'context':context,'provider':provider,'counter':counter,'history':history,'tools':tools}


def validate_case(path, case_id, snapshot_id, owners=None, source_sha256=None, *, require_host=True,
                  _pending=False, native_source=None):
    """Recompute from private raw receipts; source completed alone grants nothing."""
    path=Path(path)
    result=json.loads((path/'harness_result.json').read_text())
    if (result.get('execution_protocol')!=PROFILE or result.get('contract')!=CONTRACT
            or result.get('status')!='completed' or result.get('driver_status')!='owner_schedule_completed'
            or result.get('measurement_validity')!={'valid':not _pending,'profile':PROFILE}
            or result.get('case_id')!=case_id or result.get('snapshot_id')!=snapshot_id
            or not native_source or result.get('native_source_sha256')!=native_source):
        raise ProfileError('case_not_admitted_under_new_native_Claude_profile')
    if source_sha256 is not None and result.get('source_case_sha256')!=source_sha256:
        raise ProfileError('original_source_case_identity_changed')
    if not owners or result.get('original_owner_hashes')!=[digest(o) for o in owners]:
        raise ProfileError('original_owner_schedule_missing_or_changed')
    turns,receipts=result.get('turns',[]),result.get('owner_receipts',[])
    if len(turns)!=len(owners) or len(receipts)!=len(owners):raise ProfileError('original_owner_schedule_incomplete')
    before,sid,previous=b'',None,None
    expected_host=[]
    for index,(turn,receipt,owner) in enumerate(zip(turns,receipts,owners),1):
        output=json.loads(_inside(path,receipt['native_archive']).read_text())
        records=json.loads(_inside(path,receipt['provider_archive']).read_text())
        archive=_inside(path,receipt['session_archive']).read_bytes()
        identity=json.loads(_inside(path,receipt['identity_archive']).read_text())
        if identity.get('binding',{}).get('case_id')!=case_id or identity['binding'].get('snapshot_id')!=snapshot_id:
            raise ProfileError('trusted_identity_case_or_Snapshot_binding_changed')
        computed=validate_owner(output,before,archive,owner,index,sid,records,result['working_dir'],previous,trusted_identity=identity)
        if computed!=receipt.get('native_receipt') or receipt.get('owner_index')!=index:
            raise ProfileError('raw_owner_receipts_changed')
        if (turn.get('prompt')!=owner or turn.get('response_text')!=output['output']
                or turn.get('messages')!=output['messages'] or turn.get('usage')!=output['usage']
                or turn.get('native_result')!=output['native_result']):
            raise ProfileError('actual_turn_or_null_native_result_changed')
        proof=receipt['process_settlement'];manager=result['mock_manager_identity']
        if proof['mock_manager_pid']!=manager['pid'] or proof['processes'][str(manager['pid'])]['start']!=manager['start']:
            raise ProfileError('mock_manager_identity_changed')
        process_receipt({int(k):v for k,v in proof['processes'].items()},proof['mock_manager_pid'],namespace_pid=proof['namespace_pid'])
        from flashrsi_pi_owner_mock_barrier_v6 import validate_mock_receipt
        mock=json.loads(_inside(path,receipt['mock_archive']).read_text())
        validate_mock_receipt(mock,case_id,snapshot_id,index)
        if mock!=receipt['mock_settlement']:raise ProfileError('mock_barrier_archive_changed')
        model_transport=json.loads(_inside(path,receipt['model_transport_archive']).read_text())
        validate_model_transport(model_transport,index)
        if model_transport.get('connections')!=len(records):
            raise ProfileError('native_model_connection_without_complete_transport_record')
        if model_transport!=receipt['model_transport_settlement']:
            raise ProfileError('model_transport_barrier_archive_changed')
        if not records or any(r.get('settled') is not True for r in records):
            raise ProfileError('native_model_requests_still_active')
        expected_host.extend((r['path'],digest(r['request']),r['response_sha256']) for r in records)
        before,sid,previous=archive,computed['session']['session_id'],computed['terminal']
    outcomes=[r['native_receipt']['terminal']['native_outcome'] for r in receipts]
    bounded=sum(r['native_receipt']['terminal']['kind']=='native_max_turns' for r in receipts)
    if result.get('native_outcomes')!=outcomes or result.get('native_max_turns_count')!=bounded:
        raise ProfileError('native_error_outcomes_or_cap_count_changed')
    privacy = json.loads((path / 'privacy_audit.json').read_text())
    if privacy.get('protocol') != 'harnessrisk-private-case-v1' or privacy.get('roots') != ['workspace', 'seed_cases', 'mock_data']:
        raise ProfileError('privacy_audit_missing')
    grade = json.loads((path / 'official_rule_evaluation.json').read_text())['runs']
    if len(grade) != 1 or grade[0].get('case_id') != case_id or grade[0].get('status') != 'completed':
        raise ProfileError('official_grade_identity_or_coverage_missing')
    metadata = json.loads((path / 'adapter_metadata.json').read_text())
    mock_start = metadata.get('mock_start', {})
    reset = mock_start.get('reset', {})
    body = reset.get('body', {})
    services = {'mock_email', 'mock_slack', 'mock_browser', 'mock_filesystem', 'mock_skill_marketplace',
                'mock_payment_api', 'mock_github', 'mock_ci_cd'}
    resets = body.get('results', {})
    if (mock_start.get('started') is not True or mock_start.get('backend') != 'process'
            or reset.get('ok') is not True or reset.get('status') != 200
            or body.get('ok') is not True or body.get('case_id') != case_id or set(resets) != services):
        raise ProfileError('actual_mock_reset_missing_failed_or_wrong_case')
    for name, entry in resets.items():
        response = entry.get('body', {})
        if (entry.get('status') != 200 or response.get('ok') is not True
                or response.get('case_id') != case_id or response.get('service', name) != name):
            raise ProfileError('actual_mock_service_reset_missing_failed_or_wrong_case:' + name)
    exported = json.loads((path / 'trajectory/trajectory.json').read_text())
    # Source export occurs before independent admission; its pending validity
    # flag is preserved. All actual native outcomes and text must be identical.
    expected_result = dict(result, measurement_validity={'valid': False, 'profile': PROFILE})
    if (exported.get('case_id') != case_id or exported.get('run_id') != case_id
            or exported.get('status') != 'completed' or exported.get('result') != expected_result
            or exported.get('mock') != {'start': mock_start, 'gateway_base_url': metadata.get('mock_gateway_base_url')}):
        raise ProfileError('source_export_identity_status_result_or_mock_changed')
    metrics = grade[0]['metrics']
    for name in ('utility_score', 'attack_success_rate', 'persistence_score', 'detection_score'):
        value = metrics.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ProfileError('official_metric_missing_or_invalid:' + name)
    if require_host:
        host=json.loads((path/'native_owner_host.json').read_text())
        if (host.get('profile')!=PROFILE or host.get('case_id')!=case_id or host.get('snapshot_id')!=snapshot_id
                or host.get('returncode')!=0 or host.get('namespace_exited') is not True
                or host.get('model_requests_settled') is not True or host.get('timed_out') is not False):
            raise ProfileError('host_namespace_cleanup_or_outcome_unconfirmed')
        from flashrsi_owned_lifetime_v1 import validate_lifetime_receipt
        validate_lifetime_receipt(host.get('lifetime_receipt',{}),
            binding={'case_id':case_id,'snapshot_id':snapshot_id,'profile':PROFILE},require_namespace=True)
        host_records=json.loads((path/'protocol_model_evidence.json').read_text())['requests']
        if any(r.get('failure') or r.get('status')!=200 or r.get('settled') is not True for r in host_records):
            raise ProfileError('trusted_host_provider_failure_or_inflight')
        if [(r.get('path'),r.get('raw_request_sha256'),r.get('response_sha256')) for r in host_records]!=expected_host:
            raise ProfileError('trusted_host_provider_records_differ_from_owner_receipts')
        from flashrsi_harnessrisk_host_claude_native_owner_v9 import prepare_request
        for record in host_records:
            raw=record.get('raw_request')
            if not isinstance(raw,dict) or digest(raw)!=record.get('raw_request_sha256'):
                raise ProfileError('trusted_host_raw_request_missing_or_changed')
            forwarded,removed,conversions=prepare_request(record['path'],raw)
            if (record.get('forwarded_request')!=forwarded or record.get('forwarded_request_sha256')!=digest(forwarded)
                    or record.get('projection_removed')!=removed or record.get('reference_conversions')!=conversions):
                raise ProfileError('trusted_host_actual_wire_projection_changed')
            try:response=base64.b64decode(record['response_body_b64'],validate=True)
            except (KeyError,ValueError,TypeError):raise ProfileError('trusted_host_raw_response_missing') from None
            if byte_hash(response)!=record.get('response_sha256'):
                raise ProfileError('trusted_host_raw_response_changed')
    return {'profile':PROFILE,'owners':len(turns),'bounded_owners':bounded,'source_grade':grade[0],
        'snapshot_id':snapshot_id,'measurement_valid':True,
        'native_counter_reset_observed':any(r['native_receipt']['counter']['native_counter_reset_observed'] for r in receipts)}


def archive_native_tree(state_dir, scratch, destination):
    """Durable private byte archive, including failed/partial native JSONL.

    Native tree is data only: never import, interpret, resume or repair it here.
    All descriptor-relative opens reject symlinks/FIFOs, even after enumeration.
    """
    state_dir,destination=Path(state_dir),Path(destination)
    rows=[];total=0
    if not state_dir.is_dir():return {'files':[],'missing':True}
    for path in sorted(state_dir.rglob('*')):
        if path.is_symlink():raise ProfileError('native_tree_contains_symlink')
        if path.is_dir():continue
        raw=read_native_session(path,scratch)
        total+=len(raw)
        if total>512*1024*1024:raise ProfileError('native_tree_archive_total_size_exceeded')
        relative=path.relative_to(state_dir)
        save_bytes(destination/relative,raw)
        rows.append({'path':str(relative),'bytes':len(raw),'sha256':byte_hash(raw)})
    receipt={'files':rows,'bytes':total,'tree_sha256':digest(rows)}
    save_json(destination/'archive_manifest.json',receipt)
    return receipt


def history_frame_receipt(frames, before, after):
    rows=session_rows(after);by_uuid={r['uuid']:r for r in rows if r.get('uuid')}
    prior_ids={r.get('uuid') for r in session_rows(before)} if before else set()
    observed=[]
    for frame in frames:
        if frame.get('type') not in {'assistant','user'} or frame.get('parent_tool_use_id') is not None:continue
        identity=frame.get('uuid');row=by_uuid.get(identity)
        if not identity or not row:raise ProfileError('native_JSONL_flush_missing_emitted_message_UUID')
        if identity in prior_ids:raise ProfileError('native_old_history_reemitted_or_tool_replayed')
        message=dict(frame.get('message',{}));message.setdefault('role',frame['type'])
        if _message(row.get('message',{}))!=_message(message):
            raise ProfileError('native_JSONL_emitted_message_content_changed')
        observed.append(identity)
    if not observed:raise ProfileError('native_JSONL_final_assistant_flush_unknown')
    return {'message_uuids':observed,'messages_sha256':digest([by_uuid[k] for k in observed])}


def validate_model_transport(value,index):
    if (not isinstance(value,dict) or value.get('owner_index')!=index or value.get('settled') is not True
            or value.get('active')!=0 or value.get('accept_loop_stopped') is not True
            or value.get('backlog_drained') is not True or value.get('failures')!=[]
            or type(value.get('connections')) is not int or value['connections']<1):
        raise ProfileError('actual_native_model_transport_settlement_missing_or_uncertain')
    return value


def tool_execution_receipt(inputs, tool_ids, session_id):
    if not isinstance(inputs,list):raise ProfileError('actual_native_tool_callbacks_missing')
    starts=[];ends=[]
    for value in inputs:
        if value.get('session_id')!=session_id or not value.get('tool_use_id'):
            raise ProfileError('actual_native_tool_callback_identity_unknown')
        event=value.get('hook_event_name')
        if event=='PreToolUse':starts.append(value['tool_use_id'])
        elif event in {'PostToolUse','PostToolUseFailure'}:
            if value.get('is_interrupt') is True:raise ProfileError('actual_native_tool_interrupted')
            ends.append(value['tool_use_id'])
        else:raise ProfileError('unknown_native_tool_callback')
    if (len(starts)!=len(set(starts)) or len(ends)!=len(set(ends))
            or set(starts)!=set(ends) or set(starts)!=set(tool_ids)):
        raise ProfileError('native_tool_execution_callbacks_unsettled_or_synthetic_result')
    return {'tool_ids':starts,'inputs_sha256':digest(inputs)}


NATIVE_TITLE_SYSTEM_SHA256='765b5ba2fa0a315a3c749c7e54bf5cef450084a745eb01e66371b8b6359d4752'

def native_request_kind(request):
    """Only recognize the exact retained original native session-title prompt.

    Auxiliary calls are never dropped, replayed or counted as owner exposure.
    Unknown model calls still encounter the native history proof and fail closed.
    """
    system=request.get('system',[])
    text=system[-1].get('text') if isinstance(system,list) and system and isinstance(system[-1],dict) else None
    schema=request.get('output_config',{}).get('format',{})
    title_schema={'type':'json_schema','schema':{'type':'object','properties':{'title':{'type':'string'}},
                                              'required':['title'],'additionalProperties':False}}
    if (isinstance(text,str) and byte_hash(text.encode())==NATIVE_TITLE_SYSTEM_SHA256
            and request.get('tools')==[] and schema==title_schema):
        return 'session_title'
    return 'agent'


def agent_origin_receipt(records, frames, archive):
    """Reconcile successful scheduled API responses with raw/native message IDs."""
    frame_ids={f.get('message',{}).get('id') for f in frames if f.get('type')=='assistant'}
    session_ids={r.get('message',{}).get('id') for r in session_rows(archive) if r.get('type')=='assistant'}
    api_ids=[];pairs=[]
    for record in records:
        if record['path']!='/v1/messages' or native_request_kind(record['request'])!='agent':continue
        try:raw=base64.b64decode(record['response_body_b64'],validate=True)
        except (KeyError,ValueError,TypeError):raise ProfileError('scheduled_API_raw_response_missing') from None
        if byte_hash(raw)!=record.get('response_sha256'):raise ProfileError('scheduled_API_raw_response_changed')
        try:
            if record.get('content_type')=='application/json':messages=[json.loads(raw)]
            elif record.get('content_type')=='text/event-stream':
                events=[json.loads(line[5:].strip()) for line in raw.decode().splitlines() if line.startswith('data:') and line[5:].strip()!='[DONE]']
                messages=[e['message'] for e in events if e.get('type')=='message_start']
            else:raise ValueError()
        except (ValueError,UnicodeError,KeyError) as exc:raise ProfileError('scheduled_API_response_origin_unknown') from exc
        ids=[m.get('id') for m in messages if m.get('type')=='message' or m.get('role')=='assistant']
        if len(ids)!=1 or not ids[0] or ids[0] not in frame_ids or ids[0] not in session_ids:
            raise ProfileError('scheduled_API_response_not_reconciled_to_raw_frames_and_session')
        api_ids.extend(ids)
        pairs.append(digest({'request':record['request'],'response_sha256':record['response_sha256']}))
    if not api_ids or len(api_ids)!=len(set(api_ids)) or len(pairs)!=len(set(pairs)):
        raise ProfileError('scheduled_API_response_duplicate_retry_or_missing_origin')
    return {'scheduled_origin_reconciled':True,'distinct_scheduled_pairs':len(pairs),
            'scheduled_API_message_ids':api_ids,'scheduled_pairs_sha256':digest(pairs)}


PROTECTED_NATIVE_ENV=frozenset({
    'DISABLE_AUTO_COMPACT','CLAUDE_AUTOCOMPACT_PCT_OVERRIDE','CLAUDE_CODE_AUTO_COMPACT_WINDOW',
    'CLAUDE_CODE_MAX_CONTEXT_TOKENS','CLAUDE_CODE_MAX_OUTPUT_TOKENS','CLAUDE_CODE_MAX_RETRIES',
    'CLAUDE_CODE_DISABLE_AUTO_COMPACT','CLAUDE_CODE_DISABLE_AUTO_MEMORY','CLAUDE_CODE_SKIP_PROMPT_HISTORY',
    'CLAUDE_CONFIG_DIR','CLAUDE_SECURESTORAGE_CONFIG_DIR','CLAUDE_CODE_TMPDIR','CLAUDE_PLUGIN_ROOT',
    'CLAUDE_CODE_SDK_READS_SESSION_STATE','CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS'})

def validate_candidate_policy(settings):
    environment=settings.get('env',{})
    if any(name.upper() in PROTECTED_NATIVE_ENV or 'AUTOCOMPACT' in name.upper()
           or 'AUTO_COMPACT' in name.upper() for name in environment):
        raise ProfileError('candidate_protected_native_policy_override')
    # Settings may evolve instructions/tools within existing source rules, but
    # cannot alter runtime budget/compaction through direct native settings keys.
    protected={'maxTurns','maxBudgetUsd','maxRetries','maxOutputTokens','contextWindow',
               'autoCompactEnabled','autoCompactWindow','autoCompactThreshold'}
    if protected.intersection(settings):raise ProfileError('candidate_protected_native_settings_override')

PARSER_SHA256='febb1aee19e47e03433d89cd4b1f8c5636950e206df77e4f4ca2738b0c900393'

def reconcile_result(raw,parsed):
    import inspect
    from dataclasses import fields
    from claude_agent_sdk import ResultMessage
    from claude_agent_sdk._internal import message_parser
    if byte_hash(Path(inspect.getsourcefile(message_parser)).read_bytes())!=PARSER_SHA256:
        raise ProfileError('pinned_native_result_parser_changed')
    aliases={'modelUsage':'model_usage'}
    keys={field.name for field in fields(ResultMessage)}-{'model_usage'}
    known=keys|set(aliases)|{'type'}
    if not isinstance(parsed,dict) or set(raw)-known:
        raise ProfileError('unexplained_raw_native_result_field')
    if any(parsed.get(aliases.get(k,k))!=v for k,v in raw.items() if k!='type'):
        raise ProfileError('parsed_result_differs_from_actual_raw_native_result')
    return {'raw_sha256':digest(raw),'parsed_sha256':digest(parsed),'parser_sha256':PARSER_SHA256}
