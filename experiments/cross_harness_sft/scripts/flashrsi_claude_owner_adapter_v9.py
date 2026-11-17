"""Narrow native Claude runner retaining the original SDK/resource/auth options."""
from __future__ import annotations

import asyncio
import base64
import copy
import json
from pathlib import Path
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from contextlib import aclosing

from flashrsi_claude_owner_profile_v9 import (PROFILE, CONTRACT, ProfileError, valid_uuid,
    byte_hash, digest, target_config, terminal_receipt, context_receipt, locate_session, validate_candidate_policy)


def plain(value):
    if is_dataclass(value): return {f.name: plain(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value, Mapping):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,Path):return str(value)
    if value is None or type(value) in (str,int,float,bool):return value
    raise ProfileError('unserializable_actual_native_field:'+type(value).__name__)


def build_options(config, files_dir, task_dir, runtime, resume, stderr, stops, tool_inputs=None):
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
    from flashrsi.agents.claude_code import _authentication_env, _settings
    from flashrsi._records import _redact, _secrets
    if config != target_config(config['llm'].get('api_key'), config['llm'].get('base_url')):
        raise ProfileError('registered_Claude_native_options_changed')
    if resume is not None:valid_uuid(resume)
    llm=config['llm'];env=_authentication_env(llm)
    env.update({'CLAUDE_PLUGIN_ROOT':str(files_dir),'CLAUDE_CODE_DISABLE_AUTO_MEMORY':'1',
        'CLAUDE_CODE_SKIP_PROMPT_HISTORY':'','CLAUDE_CODE_MAX_CONTEXT_TOKENS':'32768',
        'CLAUDE_CONFIG_DIR':str(runtime/'state'),'CLAUDE_SECURESTORAGE_CONFIG_DIR':str(runtime/'state'),
        'CLAUDE_CODE_TMPDIR':str(runtime/'tmp')})
    (runtime/'tmp').mkdir(mode=0o700,exist_ok=True)
    instructions_path=files_dir/'CLAUDE.md'
    instructions=instructions_path.read_text() if instructions_path.is_file() else ''
    if config.get('_worker_system_prompt'):instructions+='\n\n'+config['_worker_system_prompt']
    settings=_settings(files_dir/'settings.json')
    settings_bytes=(files_dir/'settings.json').read_bytes() if settings else b''
    settings_value=json.loads(settings_bytes.decode('utf-8-sig')) if settings else {}
    validate_candidate_policy(settings_value)
    system_prompt={'type':'preset','preset':'claude_code'}
    if instructions.strip():system_prompt['append']=instructions
    async def observe_stop(value, _tool_use_id, _context):
        stops.append(plain(value))
        return {}  # Observation only: no permission/context/continue changes.
    async def observe_tool(value, _tool_use_id, _context):
        if tool_inputs is not None:tool_inputs.append(plain(value))
        return {}
    options=ClaudeAgentOptions(model=llm['model_name'],system_prompt=system_prompt,
        plugins=[{'type':'local','path':str(files_dir)}],skills='all',setting_sources=[],strict_mcp_config=True,
        mcp_servers=str(files_dir/'.mcp.json') if (files_dir/'.mcp.json').is_file() else {},
        settings=settings,forward_subagent_text=True,include_hook_events=True,
        cwd=task_dir,env=env,permission_mode=config.get('permission_mode','acceptEdits'),
        allowed_tools=config.get('allowed_tools',['Read','Glob','Grep']),disallowed_tools=config.get('disallowed_tools',[]),
        tools=config.get('tools'),max_turns=config.get('max_turns',10),max_budget_usd=config.get('max_budget_usd'),
        max_buffer_size=config.get('max_buffer_size',32*1024*1024),effort=llm.get('effort'),
        cli_path=Path(config['cli_path']).expanduser().resolve() if config.get('cli_path') else None,stderr=stderr,
        resume=resume,continue_conversation=False,session_id=None,fork_session=False,
        resume_session_at=None,resume_drops_turn=None,session_store=None,
        hooks={'Stop':[HookMatcher(hooks=[observe_stop])],
               **{name:[HookMatcher(hooks=[observe_tool])] for name in ('PreToolUse','PostToolUse','PostToolUseFailure')}})
    evidence={'profile':PROFILE,'settings_sha256':byte_hash(settings_bytes),
        'settings':settings_value,'environment':env,'resume':resume,'continue_conversation':False,
        'session_id':None,'fork_session':False,'resume_session_at':None,'resume_drops_turn':None,
        'session_store':None,'cwd':str(task_dir),'files_dir':str(files_dir),'contract':CONTRACT}
    return options,_redact(evidence,_secrets(config)+_secrets(env))


async def run(request):
    from claude_agent_sdk import ClaudeSDKClient, ResultMessage, SystemMessage
    from flashrsi.agents.claude_code import _disconnect, _record, _extension_error, _add_usage
    from flashrsi._records import _redact, _secrets
    from flashrsi_claude_owner_transport_v9 import ObservedTransport
    if request.get('profile')!=PROFILE or request.get('harness')!='claude_code':
        raise ProfileError('Claude_native_v9_only')
    home=Path(__import__('os').environ['HOME']).resolve(strict=True)
    runtime=Path(request['state_dir']);work=Path(request['working_dir']).resolve(strict=True)
    files=Path(request['workspace_dir']).resolve(strict=True)
    if not runtime.is_absolute() or not runtime.resolve().is_relative_to(home) or runtime.resolve().is_relative_to(work):
        raise ProfileError('persistent_native_state_must_be_existing_case_scratch')
    current=home
    for part in runtime.relative_to(home).parts:
        current/=part
        if current.is_symlink():raise ProfileError('persistent_native_state_symlink')
    runtime.mkdir(mode=0o700,parents=True,exist_ok=True)
    resume=request.get('resume');index=request['owner_index']
    if (index==1)!=(resume is None):raise ProfileError('actual_previous_native_UUID_resume_required')
    before=b''
    if resume is not None:
        _,before=locate_session(runtime/'state',home,resume)
    if byte_hash(before)!=request['expected_session_sha256']:
        raise ProfileError('native_state_prefix_changed_before_owner')
    frames,stops,messages,usage=[],[],[],{}
    tool_inputs=[]
    changed=asyncio.Event();stderr_lines=[]
    secrets=_secrets(request['config'])
    def stderr(line):stderr_lines.append(_redact(line,secrets))
    options,settings=build_options(request['config'],files,work,runtime,resume,stderr,stops,tool_inputs)
    # Same original bidirectional input shape; original transport builds argv.
    async def unused_prompt():
        if False:yield {}
    transport=ObservedTransport(prompt=unused_prompt(),options=options,frames=frames,changed=changed)
    from flashrsi_claude_owner_identity_v9 import request_identity
    async def observe_identity(pid):
        return await request_identity(pid,request['identity_request'],request['identity_response'],home,request['identity_binding'])
    transport.identity_observer=observe_identity
    client=ClaudeSDKClient(options=options,transport=transport)
    context=None;result=None;error=None
    try:
        try:
            async with asyncio.timeout(600):
                await client.connect()
                await client.query(request['question'])  # Exactly one original owner.
                async with aclosing(client.receive_messages()) as stream:
                    async for message in stream:
                        messages.append(_record(message))
                        if isinstance(message,SystemMessage):
                            extension_error=_extension_error(message)
                            if extension_error:raise ProfileError(extension_error)
                        if isinstance(message,ResultMessage):
                            result=message;_add_usage(usage,message)
                # An observation control request during this genuine owner run,
                # before waiting for possibly absent Stop/idle evidence.
                context=plain(await client.get_context_usage())
                # The public receive stream may close at an SDK run-end ceiling.
                # It is never sufficient proof. Wait for actual raw idle + Stop;
                # no prompt, interrupt, task-stop or wakeup is introduced.
                while True:
                    states=[f for f in frames if f.get('subtype')=='session_state_changed']
                    if result is not None and stops and states and states[-1].get('state')=='idle':break
                    changed.clear()
                    # Avoid a lost raw-frame wakeup between inspection and clear.
                    states=[f for f in frames if f.get('subtype')=='session_state_changed']
                    if result is not None and stops and states and states[-1].get('state')=='idle':break
                    await changed.wait()
        finally:
            await _disconnect(client)  # Original cancellation-shielded cleanup.
    except BaseException as exc:
        error={'type':type(exc).__name__,'detail':_redact(str(exc),secrets),
            'missing_evidence': ([name for name,present in (
                ('scheduled_owner_result',result is not None),
                ('native_Stop',bool(stops)),
                ('native_idle',any(f.get('subtype')=='session_state_changed' and f.get('state')=='idle' for f in frames)),
                ('native_context_usage',context is not None)) if not present])}
        if isinstance(exc,(KeyboardInterrupt,SystemExit,asyncio.CancelledError)):
            transport.close_evidence['events'].append('cancelled')
    raw_result=plain(result) if result is not None else None
    # Preserve actual nullable result independently from the source's string API.
    answer=(result.result or '') if result is not None else ''
    sid=result.session_id if result is not None else next((f.get('session_id') for f in frames if f.get('subtype')=='init'),resume)
    return {'profile':PROFILE,'owner_index':index,'owner_sha256':digest(request['question']),
        'identity_binding':request.get('identity_binding'),'session_before_sha256':byte_hash(before),'resume_requested':resume,'session_id':sid,
        'raw_stdout_b64':[base64.b64encode(b).decode() for b in transport.raw_stdout_bytes],
        'raw_frames':frames,'raw_writes':transport.raw_writes,'Stop_inputs':stops,'tool_hook_inputs':tool_inputs,'close':transport.close_evidence,
        'context_usage':context,'runtime_settings':settings,'native_result':raw_result,
        'output':_redact(answer,secrets),'messages':_redact(messages,secrets),'usage':_redact(usage,secrets),
        'error':error,'stderr_tail':'\n'.join(stderr_lines)[-8192:]}
