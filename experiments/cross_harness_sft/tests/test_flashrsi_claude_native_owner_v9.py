"""Pure filesystem/retained-frame tests. No CLI, services, source tasks or model."""
import asyncio
import copy
import importlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SID = '5600a401-b77e-4b3d-b82f-79ecfe37f623'

def profile():
    return importlib.import_module('flashrsi_claude_owner_profile_v9')

def test_new_profile_surface_exists():
    assert (ROOT / 'scripts/flashrsi_claude_owner_profile_v9.py').is_file(), 'new Claude profile is missing'

def frames():
    return [
        {'type':'system','subtype':'init','session_id':SID,'cwd':'/work'},
        {'type':'assistant','message':{'content':[{'type':'tool_use','id':'toolu_1','name':'Bash','input':{'command':'pwd'}}]}},
        {'type':'user','message':{'content':[{'type':'tool_result','tool_use_id':'toolu_1','content':'/work'}]}},
        {'type':'result','subtype':'error_max_turns','terminal_reason':'max_turns','is_error':True,
         'num_turns':17,'session_id':SID,'stop_reason':'tool_use','result':None,'errors':['Reached maximum number of turns (16)'],
         'api_error_status':None,'deferred_tool_use':None,'origin':None},
        {'type':'system','subtype':'session_state_changed','state':'idle','session_id':SID,'sdk_host_only':True}]

def stop():
    return [{'hook_event_name':'Stop','session_id':SID,'cwd':'/work','stop_hook_active':False,
             'background_tasks':[],'session_crons':[]}]

def close():
    return {'pid':44,'start':123,'events':['begin','lock_acquired','stdin_closed','graceful_wait','graceful_reaped','reaped','finished'],
            'returncode':0,'reaped':True,'exception':None}

def test_native_cap_keeps_raw_error_and_null_answer():
    p = profile()
    receipt = p.terminal_receipt(frames(), stop(), close(), SID)
    assert receipt['kind'] == 'native_max_turns'
    assert receipt['native_outcome'] == frames()[3]
    assert receipt['native_outcome']['result'] is None
    assert receipt['native_outcome']['num_turns'] == 17

@pytest.mark.parametrize('mutation', ['idle','stop','background','cron','pending_tool','deferred','aborted','api','wrong_session','no_result','notification','turns'])
def test_unknown_or_unsettled_native_boundaries_fail_closed(mutation):
    p = profile(); f,s = frames(),stop()
    if mutation == 'idle': f.pop()
    if mutation == 'stop': s=[]
    if mutation == 'background': s[0]['background_tasks']=[{'id':'bg','type':'shell','status':'running'}]
    if mutation == 'cron': del s[0]['session_crons']
    if mutation == 'pending_tool': f.pop(2)
    if mutation == 'deferred': f[3]['deferred_tool_use']={'tool_use_id':'toolu_1'}
    if mutation == 'aborted': f[3]['terminal_reason']='aborted_tools'
    if mutation == 'api': f[3]['api_error_status']=400
    if mutation == 'wrong_session': f[-1]['session_id']='different'
    if mutation == 'no_result': f.pop(3)
    if mutation == 'notification': f[3]['origin']={'kind':'task-notification'}
    if mutation == 'turns': f[3]['num_turns']=100
    with pytest.raises(p.ProfileError): p.terminal_receipt(f,s,close(),SID)

@pytest.mark.parametrize('event', ['terminate','kill','cancelled','lock_timeout'])
def test_close_escalation_and_uncertainty_invalid(event):
    p=profile(); c=close(); c['events'].append(event)
    with pytest.raises(p.ProfileError): p.close_receipt(c)

def test_missing_close_observation_is_unknown():
    p=profile()
    with pytest.raises(p.ProfileError): p.close_receipt({'returncode':0})

def test_sdk_options_keep_original_caps_and_explicit_resume(tmp_path):
    from flashrsi_claude_owner_profile_v9 import target_config
    from flashrsi_claude_owner_adapter_v9 import build_options
    root=tmp_path/'state';root.mkdir(); files=tmp_path/'snapshot';files.mkdir();work=tmp_path/'work';work.mkdir()
    opts,evidence=build_options(target_config('x'*40,'http://127.0.0.1:1234'),files,work,root,SID,lambda _:None,[])
    assert opts.resume == SID and opts.session_id is None and opts.continue_conversation is False
    assert opts.fork_session is False and opts.resume_session_at is None and opts.resume_drops_turn is None
    assert opts.session_store is None and opts.max_turns == 16
    assert opts.permission_mode == 'dontAsk' and opts.setting_sources == [] and opts.strict_mcp_config is True
    assert opts.env['CLAUDE_CODE_MAX_CONTEXT_TOKENS']=='32768'
    assert opts.env['CLAUDE_CODE_MAX_OUTPUT_TOKENS']=='4096'
    assert 'CLAUDE_CODE_AUTO_COMPACT_WINDOW' not in opts.env
    assert opts.env['CLAUDE_CONFIG_DIR']==str(root/'state')
    assert opts.env['CLAUDE_SECURESTORAGE_CONFIG_DIR']==str(root/'state')
    assert 'x'*40 not in json.dumps(evidence)

def transcript(owner='first',parent=None,uid='u1'):
    return {'type':'user','uuid':uid,'parentUuid':parent,'sessionId':SID,'cwd':'/work','message':{'role':'user','content':owner}}

def encoded(rows): return b''.join((json.dumps(r)+'\n').encode() for r in rows)

def test_native_history_preserves_prefix_and_exact_next_owner():
    p=profile(); old=[transcript(), {'type':'assistant','uuid':'a1','parentUuid':'u1','sessionId':SID,'cwd':'/work','message':{'role':'assistant','content':[{'type':'text','text':'ok'}]}}]
    after=old+[transcript('second','a1','u2'),{'type':'assistant','uuid':'a2','parentUuid':'u2','sessionId':SID,'cwd':'/work','message':{'role':'assistant','content':[{'type':'text','text':'done'}]}}]
    r=p.session_receipt(encoded(old),encoded(after),'second','/work',SID)
    assert r['session_id']==SID and r['owner_uuid']=='u2' and r['leaf_id']=='a2'

@pytest.mark.parametrize('mutation',['prefix','owner_duplicate','wrong_owner','wrong_cwd','wrong_session','dangling','duplicate_uuid','partial','compaction'])
def test_native_history_corruption_is_invalid(mutation):
    p=profile(); old=[transcript()]; new=transcript('second','u1','u2'); after=old+[new]
    if mutation=='prefix': old=[transcript('mutated')]
    if mutation=='owner_duplicate': after.append(transcript('second','u2','u3'))
    if mutation=='wrong_owner': new['message']['content']='other'
    if mutation=='wrong_cwd': new['cwd']='/else'
    if mutation=='wrong_session': new['sessionId']='other'
    if mutation=='dangling': new['parentUuid']='missing'
    if mutation=='duplicate_uuid': new['uuid']='u1'
    if mutation=='compaction': after.append({'type':'system','subtype':'compact_boundary','uuid':'c1','parentUuid':'u2','sessionId':SID,'cwd':'/work'})
    raw=encoded(after)
    if mutation=='partial': raw=raw[:-1]
    with pytest.raises(p.ProfileError): p.session_receipt(encoded(old),raw,'second','/work',SID)

def test_context_must_be_actual_native_observation():
    p=profile()
    assert p.context_receipt({'rawMaxTokens':32768,'maxTokens':32768,'model':'qwen35-4b-preflight','isAutoCompactEnabled':True})['rawMaxTokens']==32768
    for value in ({}, {'rawMaxTokens':200000}, {'rawMaxTokens':32768,'maxTokens':32768,'model':'wrong','isAutoCompactEnabled':True}):
        with pytest.raises(p.ProfileError): p.context_receipt(value)

def test_wire_owner_and_output_budget_and_real_invocation():
    p=profile(); records=[{'path':'/v1/messages','request':{'model':'qwen35-4b-preflight','max_tokens':4096,'messages':[{'role':'user','content':'first'}]},'status':200,'response_sha256':'a'}]
    p.provider_receipt(records,b'', 'first', SID, 1)
    for change in ('empty','max','failed','owner'):
        r=copy.deepcopy(records)
        if change=='empty':r=[]
        if change=='max':r[0]['request']['max_tokens']=8192
        if change=='failed':r[0]['status']=400
        if change=='owner':r[0]['request']['messages'][0]['content']='other'
        with pytest.raises(p.ProfileError):p.provider_receipt(r,b'','first',SID,1)

def test_projection_preserves_repeated_genuine_owner_text():
    p=profile(); request={'messages':[{'role':'user','content':[{'type':'text','text':'same'}]},{'role':'assistant','content':[{'type':'text','text':'ok'}]},{'role':'user','content':[{'type':'text','text':'same'}]}]}
    projected,removed=p.project_cli_history(request)
    assert projected==request and removed==[]

def test_reader_rejects_symlink_and_fifo_without_blocking(tmp_path):
    p=profile(); path=tmp_path/'session';path.symlink_to('/etc/hosts')
    with pytest.raises(p.ProfileError):p.read_native_session(path,tmp_path)
    path.unlink();import os;os.mkfifo(path)
    with pytest.raises(p.ProfileError):p.read_native_session(path,tmp_path)

@pytest.mark.parametrize('timeouts,expected',[(0,'clean'),(1,'terminate'),(2,'kill')])
def test_exact_sdk_close_branches_are_observed_without_processes(timeouts,expected):
    from flashrsi_claude_owner_transport_v9 import ObservedTransport
    class Stream:
        async def aclose(self):pass
    class Process:
        pid=44;returncode=None
        def __init__(self):self.waits=0;self.signals=[]
        async def wait(self):
            self.waits+=1
            if self.waits<=timeouts:raise TimeoutError()
            self.returncode=0
            return 0
        def terminate(self):self.signals.append('terminate')
        def kill(self):self.signals.append('kill')
    async def exercise():
        import anyio
        t=object.__new__(ObservedTransport)
        t.close_evidence={'pid':44,'start':123,'events':[],'exception':None,'returncode':None,'reaped':None}
        process=Process();t._process=process;t._write_lock=anyio.Lock()
        t._stderr_task=None;t._stdin_stream=Stream();t._stderr_stream=None;t._ready=True
        await t.close()
        return t,process
    t,process=asyncio.run(exercise())
    assert t._process is None and t.close_evidence['reaped'] is True
    assert process.waits==timeouts+1
    if expected=='clean':assert profile().close_receipt(t.close_evidence)
    else:
        assert expected in t.close_evidence['events']
        with pytest.raises(profile().ProfileError):profile().close_receipt(t.close_evidence)

def test_observational_copy_preserves_entire_pinned_close_ast():
    import ast,inspect,textwrap
    from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
    from flashrsi_claude_owner_transport_v9 import ObservedTransport
    original=ast.parse(textwrap.dedent(inspect.getsource(SubprocessCLITransport.close)))
    observed=ast.parse(textwrap.dedent(inspect.getsource(ObservedTransport._close_observed)))
    class StripObservations(ast.NodeTransformer):
        def visit_Expr(self,node):
            if 'self.close_evidence' in ast.unparse(node):return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if any('self.close_evidence' in ast.unparse(t) for t in node.targets):return None
            return self.generic_visit(node)
        def visit_If(self,node):
            node=self.generic_visit(node)
            return node if node.body else None
    observed=StripObservations().visit(observed)
    observed.body[0].name='close'
    assert ast.dump(observed,include_attributes=False)==ast.dump(original,include_attributes=False)

def test_raw_transport_captures_host_only_idle_before_sdk_filter(monkeypatch):
    from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
    from flashrsi_claude_owner_transport_v9 import ObservedTransport
    raw=frames()[-1]
    async def source(_self):yield raw
    monkeypatch.setattr(SubprocessCLITransport,'read_messages',source)
    async def exercise():
        t=object.__new__(ObservedTransport);t.frames=[];t.changed=asyncio.Event()
        received=[frame async for frame in t.read_messages()]
        raw['state']='running'
        return t,received
    t,received=asyncio.run(exercise())
    assert received[0] is raw and t.frames[0]['state']=='idle' and t.changed.is_set()

def test_retained_native_cap_shape_is_actual_saved_result():
    path=ROOT/'outputs/flashrsi_harnessrisk_4b_private_claude_cli_projection_v8_gpu1_20261115/claude_code_initial_full/cases/action_042/harness_result.json'
    if not path.is_file():pytest.skip('retained historical artifact unavailable; never fabricate it')
    result=json.loads(path.read_text())
    native=[m for turn in result['turns'] for m in turn['messages'] if m.get('type')=='ResultMessage'][-1]
    assert native['subtype']=='error_max_turns' and native['terminal_reason']=='max_turns'
    assert native['num_turns']==17 and native['is_error'] is True
    f=frames();f[3]={'type':'result',**{k:v for k,v in native.items() if k!='type'}}
    native_sid=native['session_id'];f[0]['session_id']=native_sid;f[-1]['session_id']=native_sid
    s=stop();s[0]['session_id']=native_sid
    admitted=profile().terminal_receipt(f,s,close(),native_sid)
    assert admitted['native_outcome']['result']==native['result']
    # This tests the native result shape only. Historical Stop/idle/flush were
    # absent and this fixture does not upgrade the old case to a valid run.

@pytest.mark.parametrize('failure',['stdin','cancel'])
def test_close_failure_retains_uncertainty_and_reaper_handle(failure):
    from flashrsi_claude_owner_transport_v9 import ObservedTransport
    class Stream:
        async def aclose(self):
            if failure=='cancel':raise asyncio.CancelledError()
            raise OSError('stdin close failed')
    class Process:
        pid=44;returncode=None
        async def wait(self):self.returncode=0
    async def exercise():
        import anyio
        t=object.__new__(ObservedTransport)
        t.close_evidence={'pid':44,'start':123,'events':[],'exception':None,'returncode':None,'reaped':None}
        t._process=Process();t._write_lock=anyio.Lock();t._stderr_task=None;t._stderr_stream=None;t._stdin_stream=Stream()
        try:await t.close()
        except asyncio.CancelledError:pass
        return t
    t=asyncio.run(exercise())
    with pytest.raises(profile().ProfileError):profile().close_receipt(t.close_evidence)
    if failure=='cancel':assert t._process is not None and t.close_evidence['exception']=='CancelledError'

def test_counter_requires_actual_owner_invocation_and_rejects_cumulative_values():
    p=profile();terminal=p.terminal_receipt(frames(),stop(),close(),SID)
    assert p.counter_receipt(terminal,{'model_requests':17,'distinct_scheduled_pairs':16,'scheduled_origin_reconciled':True},terminal)['resumed_after_cap_observed'] is True
    for calls in (0,18):
        with pytest.raises(p.ProfileError):p.counter_receipt(terminal,{'model_requests':calls})
    terminal['native_outcome']['num_turns']=34
    with pytest.raises(p.ProfileError):p.counter_receipt(terminal,{'model_requests':1})

def test_reference_preserves_all_six_original_algorithms_without_execution():
    import ast
    from flashrsi_harnessrisk_claude_native_owner_reference_v9 import build_source
    original=ast.parse((ROOT/'scripts/flashrsi_harnessrisk_algorithm.py').read_text())
    transformed=ast.parse(build_source())
    def binding(tree):
        return next(ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ALGORITHMS' for t in n.targets))
    assert binding(original)==binding(transformed)
    assert 'validate_reference' in build_source()

def test_source_completed_status_alone_is_not_native_admission(tmp_path):
    p=profile();(tmp_path/'harness_result.json').write_text(json.dumps({'status':'completed','case_id':'case','snapshot_id':'snap'}))
    with pytest.raises(p.ProfileError):p.validate_case(tmp_path,'case','snap',['owner'],'source',native_source={'CLAUDE.md':'hash'})

def test_model_host_keeps_one_reply_budget_and_genuine_repeated_owner():
    from flashrsi_harnessrisk_host_claude_native_owner_v9 import prepare_request
    body={'model':'qwen35-4b-preflight','max_tokens':4096,'messages':[{'role':'user','content':'same'},{'role':'assistant','content':[{'type':'text','text':'x'}]},{'role':'user','content':'same'}]}
    current,removed,conversions=prepare_request('/v1/messages',body)
    assert body.get('temperature') is None and current['temperature']==0
    assert current['messages']==body['messages'] and removed==[] and conversions==[]
    body['max_tokens']=8192
    with pytest.raises(profile().ProfileError):prepare_request('/v1/messages',body)

def test_model_transport_barrier_surface_exists():
    assert (ROOT/'scripts/flashrsi_claude_owner_model_barrier_v9.py').is_file(), 'native accepted/unparsed model barrier missing'

class MemorySocket:
    def __init__(self,pending=False):self.pending=pending;self.closed=False
    def setblocking(self,value):pass
    def accept(self):
        if not self.pending:raise BlockingIOError()
        self.pending=False
        class Connection:
            def close(self):pass
        return Connection(),('127.0.0.1',1)

class MemoryServer:
    server_port=1234
    def __init__(self,pending=False):self.socket=MemorySocket(pending);self.errors=[];self.closed=False;self.jobs=[]
    def serve_forever(self,**_kwargs):pass
    def shutdown(self):pass
    def server_close(self):self.closed=True
    def get_request(self):return object(),('127.0.0.1',1)
    def process_request_thread(self,request,address):self.jobs.append(request)

class MemoryThread:
    def __init__(self,**kwargs):self.kwargs=kwargs
    def start(self):pass
    def join(self,*_args,**_kwargs):pass

@pytest.mark.parametrize('mode',['settled','active','queued','handler_error'])
def test_model_barrier_accounts_connections_before_parse_without_socket(mode):
    from flashrsi_claude_owner_model_barrier_v9 import ModelBarrier
    server=MemoryServer(pending=mode=='queued')
    gate=ModelBarrier(lambda:server,thread_factory=MemoryThread)
    gate.open_owner(1)
    if mode in {'active','settled','handler_error'}:
        request,address=server.get_request()
        if mode!='active':server.process_request_thread(request,address)
    if mode=='handler_error':server.errors.append({'detail':'invalid HTTP framing'})
    if mode=='settled':
        receipt=gate.freeze()
        assert receipt['settled'] is True and receipt['connections']==1 and receipt['active']==0
    else:
        with pytest.raises(profile().ProfileError):gate.freeze()

def test_archive_preserves_failed_partial_session_bytes_privately(tmp_path):
    p=profile();native=tmp_path/'scratch'/'native';native.mkdir(parents=True)
    (native/'state').mkdir();raw=b'{"partial":'
    (native/'state'/'session.jsonl').write_bytes(raw)
    receipt=p.archive_native_tree(native,tmp_path/'scratch',tmp_path/'evidence')
    assert (tmp_path/'evidence/state/session.jsonl').read_bytes()==raw
    assert receipt['files'][0]['sha256']==p.byte_hash(raw)
    import shutil;shutil.rmtree(tmp_path/'scratch')
    assert (tmp_path/'evidence/archive_manifest.json').is_file()

def test_model_barrier_reuses_same_case_listener_across_original_owners():
    from flashrsi_claude_owner_model_barrier_v9 import ModelBarrier
    server=MemoryServer();gate=ModelBarrier(lambda:server,thread_factory=MemoryThread)
    for index in (1,2):
        gate.open_owner(index);receipt=gate.freeze()
        assert gate.server is server and receipt['owner_index']==index and server.closed is False
    gate.close();assert server.closed is True

def test_history_receipt_requires_final_native_assistant_and_no_old_tool_replay():
    p=profile()
    assert hasattr(p,'history_frame_receipt'), 'native JSONL flush must match emitted message UUIDs/content'
    owner=transcript();assistant={'type':'assistant','uuid':'a1','parentUuid':'u1','sessionId':SID,'cwd':'/work','message':{'role':'assistant','content':[{'type':'text','text':'done'}]}}
    f=[{'type':'assistant','uuid':'a1','message':assistant['message']}]
    raw=encoded([owner,assistant])
    assert p.history_frame_receipt(f,b'',raw)['message_uuids']==['a1']
    with pytest.raises(p.ProfileError):p.history_frame_receipt(f,b'',encoded([owner]))
    with pytest.raises(p.ProfileError):p.history_frame_receipt(f,raw,raw)
    wrong=copy.deepcopy(assistant);wrong['message']['content'][0]['text']='changed'
    with pytest.raises(p.ProfileError):p.history_frame_receipt(f,b'',encoded([owner,wrong]))

@pytest.mark.parametrize('mutation',['drop_prior','duplicate_owner','unarchived_assistant'])
def test_every_provider_request_retains_actual_native_history(mutation):
    p=profile()
    owner=transcript();assistant={'type':'assistant','uuid':'a1','parentUuid':'u1','sessionId':SID,'cwd':'/work','message':{'role':'assistant','content':[{'type':'text','text':'done'}]}}
    first={'model':p.MODEL,'max_tokens':4096,'messages':[{'role':'user','content':'first'}]}
    later=copy.deepcopy(first);later['messages'].append(copy.deepcopy(assistant['message']))
    if mutation=='drop_prior':later['messages'].pop(0)
    if mutation=='duplicate_owner':later['messages'].append({'role':'user','content':'first'})
    if mutation=='unarchived_assistant':later['messages'][-1]['content'][0]['text']='unarchived'
    records=[{'path':'/v1/messages','request':r,'status':200,'response_sha256':'hash'} for r in (first,later)]
    with pytest.raises(p.ProfileError):p.provider_receipt(records,b'','first',SID,1,after=encoded([owner,assistant]))

def test_original_tool_callbacks_close_actual_ids_without_permission_changes(tmp_path):
    p=profile()
    assert hasattr(p,'tool_execution_receipt'), 'tool execution must have actual native callback evidence'
    inputs=[{'hook_event_name':'PreToolUse','tool_use_id':'toolu_1','session_id':SID},
            {'hook_event_name':'PostToolUse','tool_use_id':'toolu_1','session_id':SID}]
    assert p.tool_execution_receipt(inputs,['toolu_1'],SID)['tool_ids']==['toolu_1']
    for values in ([],inputs[:1],inputs+inputs):
        with pytest.raises(p.ProfileError):p.tool_execution_receipt(values,['toolu_1'],SID)
    aborted=copy.deepcopy(inputs);aborted[-1].update(hook_event_name='PostToolUseFailure',is_interrupt=True)
    with pytest.raises(p.ProfileError):p.tool_execution_receipt(aborted,['toolu_1'],SID)
    from flashrsi_claude_owner_adapter_v9 import build_options
    runtime=tmp_path/'runtime';runtime.mkdir();files=tmp_path/'files';files.mkdir();work=tmp_path/'work';work.mkdir()
    observed=[];opts,_=build_options(p.target_config('x'*40,'http://127.0.0.1:1'),files,work,runtime,None,lambda _:None,[],observed)
    async def observe():
        callback=opts.hooks['PreToolUse'][0].hooks[0]
        value=await callback(inputs[0],None,{})
        assert value=={}
    asyncio.run(observe())
    assert observed==inputs[:1]

def test_native_owner_text_block_format_is_preserved():
    p=profile();owner=transcript();owner['message']['content']=[{'type':'text','text':'first'}]
    assert p.session_receipt(b'',encoded([owner]),'first','/work',SID)['owner_uuid']=='u1'

def complete_case_fixture(path):
    """Pure admission fixture; no runtime success is claimed by assembled data."""
    p=profile();owner='first';case='unit_case';snap='unit_snapshot';source={'settings.json':p.byte_hash(b'{}')}
    user=transcript();assistant={'type':'assistant','uuid':'a1','parentUuid':'u1','sessionId':SID,'cwd':'/work',
        'message':{'id':'api1','role':'assistant','content':[{'type':'text','text':'done'}]}}
    raw=encoded([user,assistant])
    f=[{'type':'system','subtype':'init','session_id':SID,'cwd':'/work'},
       {'type':'assistant','uuid':'a1','message':assistant['message']},
       {'type':'result','subtype':'success','terminal_reason':'completed','is_error':False,'num_turns':1,
        'session_id':SID,'result':'done','stop_reason':'end_turn','errors':[],'api_error_status':None,'deferred_tool_use':None},
       frames()[-1]]
    options={'profile':p.PROFILE,'contract':p.CONTRACT,'cwd':'/work','resume':None,'continue_conversation':False,
        'fork_session':False,'session_id':None,'resume_session_at':None,'resume_drops_turn':None,'session_store':None,
        'environment':{'CLAUDE_CODE_MAX_CONTEXT_TOKENS':'32768','CLAUDE_CODE_MAX_OUTPUT_TOKENS':'4096','CLAUDE_CODE_MAX_RETRIES':'0'}}
    output={'profile':p.PROFILE,'owner_index':1,'owner_sha256':p.digest(owner),'resume_requested':None,
        'session_before_sha256':p.byte_hash(b''),'error':None,'session_id':SID,'runtime_settings':options,
        'raw_frames':f,'Stop_inputs':stop(),'close':close(),'native_result':{k:v for k,v in f[2].items() if k!='type'},
        'context_usage':{'rawMaxTokens':32768,'maxTokens':19768,'model':p.MODEL,'isAutoCompactEnabled':True},
        'output':'done','messages':[],'usage':{},'tool_hook_inputs':[]}
    import base64
    response=json.dumps({'type':'message','id':'api1','role':'assistant','content':assistant['message']['content']}).encode()
    response_hash=p.byte_hash(response)
    records=[{'path':'/v1/messages','request':{'model':p.MODEL,'max_tokens':4096,'messages':[{'role':'user','content':owner}]},
              'status':200,'response_sha256':response_hash,'response_body_b64':base64.b64encode(response).decode(),'content_type':'application/json','settled':True}]
    identity={'observer':'trusted_namespace_supervisor_v9','entry_pid':43,'entry_start':122,'transport_pid':44,
              'transport_ppid':43,'transport_start':123,'observed_alive':True,'observed_monotonic_ns':1,
              'executable':{'path':p.CLI,'dev':1,'ino':1},'binding':{'case_id':case,'snapshot_id':snap,'owner_index':1,'nonce':'unit-only-nonce'}}
    output['identity_binding']=identity['binding']
    output['close']['identity']=identity
    native=p.validate_owner(output,b'',raw,owner,1,None,records,'/work',trusted_identity=identity)
    processes={1:{'ppid':0,'state':'S','start':1},2:{'ppid':1,'state':'S','start':2}}
    mock={'case_id':case,'snapshot_id':snap,'owner_index':1,'sealed':True,'accept_loop_stopped':True,'backlog_drained':True,
          'pending_connections':0,'settled':True,'request_count':0,'requests':[],'failures':[]}
    model={'owner_index':1,'connections':1,'active':0,'settled':True,'accept_loop_stopped':True,'backlog_drained':True,'failures':[]}
    receipt={'owner_index':1,'native_receipt':native,'process_settlement':p.process_receipt(processes,2,namespace_pid=1),
        'mock_settlement':mock,'model_transport_settlement':model,
        'session_archive':'native/session.jsonl','native_archive':'native/output.json','identity_archive':'native/identity.json',
        'provider_archive':'native/provider.json','mock_archive':'native/mock.json','model_transport_archive':'native/model.json'}
    result={'execution_protocol':p.PROFILE,'contract':p.CONTRACT,'status':'completed','driver_status':'owner_schedule_completed',
        'measurement_validity':{'valid':True,'profile':p.PROFILE},'case_id':case,'snapshot_id':snap,'native_source_sha256':source,
        'source_case_sha256':'source','original_owner_hashes':[p.digest(owner)],'working_dir':'/work','mock_manager_identity':{'pid':2,'start':2},
        'turns':[{'prompt':owner,'response_text':'done','messages':[],'usage':{},'native_result':output['native_result']}],
        'owner_receipts':[receipt],'native_outcomes':[f[2]],'native_max_turns_count':0}
    p.save_bytes(path/'native/session.jsonl',raw)
    for name,value in [('native/output.json',output),('native/provider.json',records),('native/mock.json',mock),('native/model.json',model),('harness_result.json',result),('native/identity.json',identity)]:p.save_json(path/name,value)
    services=['mock_email','mock_slack','mock_browser','mock_filesystem','mock_skill_marketplace','mock_payment_api','mock_github','mock_ci_cd']
    resets={name:{'status':200,'body':{'ok':True,'case_id':case,'service':name}} for name in services}
    metadata={'mock_start':{'started':True,'backend':'process','reset':{'ok':True,'status':200,'body':{'ok':True,'case_id':case,'results':resets}}},'mock_gateway_base_url':'http://127.0.0.1:1'}
    p.save_json(path/'adapter_metadata.json',metadata)
    p.save_json(path/'privacy_audit.json',{'protocol':'harnessrisk-private-case-v1','roots':['workspace','seed_cases','mock_data']})
    exported=copy.deepcopy(result);exported['measurement_validity']['valid']=False
    p.save_json(path/'trajectory/trajectory.json',{'case_id':case,'run_id':case,'status':'completed','result':exported,
        'mock':{'start':metadata['mock_start'],'gateway_base_url':metadata['mock_gateway_base_url']}})
    p.save_json(path/'official_rule_evaluation.json',{'runs':[{'case_id':case,'status':'completed',
        'metrics':{k:.5 for k in ('utility_score','attack_success_rate','persistence_score','detection_score')}}]})
    lifetime={'version':1,'host':{'pid':100,'start':1},'cleanup_confirmed':True,'launcher_reaped':True,'owned_children_reaped':True,
        'namespace_required':True,'namespace':{'pid':102,'start':2,'nspid':[102,1],'namespace_inode':1},'namespace_pid1_gone':True,
        'namespace_exited':True,'binding':{'case_id':case,'snapshot_id':snap,'profile':p.PROFILE}}
    p.save_json(path/'native_owner_host.json',{'profile':p.PROFILE,'case_id':case,'snapshot_id':snap,'returncode':0,'namespace_exited':True,'model_requests_settled':True,'timed_out':False,'lifetime_receipt':lifetime})
    from flashrsi_harnessrisk_host_claude_native_owner_v9 import prepare_request
    forwarded,removed,conversions=prepare_request('/v1/messages',records[0]['request'])
    p.save_json(path/'protocol_model_evidence.json',{'requests':[dict(records[0],raw_request=records[0]['request'],
        raw_request_sha256=p.digest(records[0]['request']),forwarded_request=forwarded,forwarded_request_sha256=p.digest(forwarded),
        projection_removed=removed,reference_conversions=conversions)]})
    return owner,source

@pytest.mark.parametrize('corruption',[None,'native_error','null_answer','no_model','wrong_resume','partial_history','source_status','reset','utility','host','provider_wire','mock_pending','model_active','caps','native_outcome'])
def test_complete_admission_recomputes_raw_native_source_and_host_receipts(tmp_path,corruption):
    p=profile();owner,source=complete_case_fixture(tmp_path)
    mutations={
        'native_error':('native/output.json',lambda v:v.update(error={'type':'TimeoutError'})),
        'null_answer':('native/output.json',lambda v:v.update(output='invented answer')),
        'no_model':('native/provider.json',lambda v:v.clear()),
        'wrong_resume':('native/output.json',lambda v:v.update(resume_requested=SID)),
        'source_status':('official_rule_evaluation.json',lambda v:v['runs'][0].update(status='error')),
        'reset':('adapter_metadata.json',lambda v:v['mock_start']['reset'].update(ok=False)),
        'utility':('official_rule_evaluation.json',lambda v:v['runs'][0]['metrics'].update(utility_score=None)),
        'host':('native_owner_host.json',lambda v:v.update(namespace_exited=False)),
        'provider_wire':('protocol_model_evidence.json',lambda v:v['requests'][0].update(raw_request_sha256='other')),
        'mock_pending':('native/mock.json',lambda v:v.update(pending_connections=1)),
        'model_active':('native/model.json',lambda v:v.update(active=1)),
        'caps':('harness_result.json',lambda v:v.update(native_max_turns_count=1)),
        'native_outcome':('harness_result.json',lambda v:v.update(native_outcomes=[])),
    }
    if corruption=='partial_history':(tmp_path/'native/session.jsonl').write_bytes(b'{"partial":')
    elif corruption:
        name,mutate=mutations[corruption];value=json.loads((tmp_path/name).read_text());mutate(value);p.save_json(tmp_path/name,value)
    if corruption:
        with pytest.raises(p.ProfileError):p.validate_case(tmp_path,'unit_case','unit_snapshot',[owner],'source',native_source=source)
    else:
        receipt=p.validate_case(tmp_path,'unit_case','unit_snapshot',[owner],'source',native_source=source)
        assert receipt['source_grade']['metrics']['utility_score']==.5 and receipt['owners']==1

def test_actual_entry_calls_one_owner_and_preserves_native_result_without_launch(monkeypatch,tmp_path):
    """Drive the real entry orchestration with only the process boundary replaced."""
    import claude_agent_sdk
    import flashrsi_claude_owner_transport_v9 as transport_module
    from flashrsi_claude_owner_adapter_v9 import run
    p=profile();home=tmp_path/'home';home.mkdir();work=tmp_path/'work';work.mkdir();files=tmp_path/'files';files.mkdir()
    monkeypatch.setenv('HOME',str(home));calls=[]
    class Transport:
        def __init__(self,prompt,options,frames,changed):
            self.options=options;self.frames=frames;self.changed=changed;self.raw_writes=[];self.raw_stdout_bytes=[];self.close_evidence=close()
    class Client:
        def __init__(self,options,transport):self.options=options;self.transport=transport
        async def connect(self):calls.append(('connect',self.options.resume))
        async def query(self,owner):
            calls.append(('query',owner));self.transport.raw_writes.append(json.dumps({'type':'user','message':{'content':owner}}))
        async def receive_messages(self):
            r=claude_agent_sdk.ResultMessage(subtype='success',duration_ms=1,duration_api_ms=1,is_error=False,
                num_turns=1,session_id=SID,result='',terminal_reason='completed',stop_reason='end_turn')
            self.transport.frames.extend([{'type':'system','subtype':'init','session_id':SID},
                {'type':'result','subtype':'success','session_id':SID,'terminal_reason':'completed','is_error':False,
                 'num_turns':1,'result':'','stop_reason':'end_turn'},frames()[-1]])
            await self.options.hooks['Stop'][0].hooks[0](stop()[0],None,{})
            self.transport.changed.set()
            yield r
        async def get_context_usage(self):
            calls.append(('context',))
            return {'rawMaxTokens':32768,'maxTokens':19768,'model':p.MODEL,'isAutoCompactEnabled':True}
        async def disconnect(self):calls.append(('disconnect',))
    monkeypatch.setattr(transport_module,'ObservedTransport',Transport)
    monkeypatch.setattr(claude_agent_sdk,'ClaudeSDKClient',Client)
    request={'profile':p.PROFILE,'harness':'claude_code','owner_index':1,'resume':None,'question':'original owner',
        'state_dir':str(home/'native'),'working_dir':str(work),'workspace_dir':str(files),
        'expected_session_sha256':p.byte_hash(b''),'config':p.target_config('x'*40,'http://127.0.0.1:1')}
    output=asyncio.run(run(request))
    assert calls==[('connect',None),('query','original owner'),('context',),('disconnect',)]
    assert output['error'] is None and output['output']=='' and output['native_result']['result']==''
    assert output['raw_writes'] and output['Stop_inputs']==stop() and output['raw_frames'][-1]['sdk_host_only'] is True

def test_runtime_source_map_binds_reused_helpers_and_exact_native_versions():
    from flashrsi_harnessrisk_claude_native_owner_initial_v9 import source_pins
    pins=source_pins()
    for name in ('flashrsi_pi_owner_profile_v6.py','flashrsi_pi_owner_mock_barrier_v6.py','flashrsi_pi_owner_mock_manager_v6.py',
                 'flashrsi_claude_owner_transport_v9.py','flashrsi_claude_owner_model_barrier_v9.py'):
        assert any(path.endswith('/'+name) for path in pins)
    assert pins[profile().CLI]=='dabafec82ca1319fa43d8be36a10fc61b1d11821cb98ad5df04ad95e199b8eed'

def test_counter_does_not_infer_reset_from_one_request_and_unchanged_native_count():
    p=profile();terminal=p.terminal_receipt(frames(),stop(),close(),SID)
    # One actual request does not prove how the native17 counter advanced.
    with pytest.raises(p.ProfileError,match='reset_structural_barrier'):p.counter_receipt(terminal,{'model_requests':1},terminal)
    receipt=p.counter_receipt(terminal,{'model_requests':17,'distinct_scheduled_pairs':16,'scheduled_origin_reconciled':True},terminal)
    assert receipt['resumed_after_cap_observed'] is True

def test_stdout_observer_preserves_actual_bytes_before_json_or_utf8_parse():
    import flashrsi_claude_owner_transport_v9 as transport
    assert hasattr(transport,'ObservedBytes'), 'raw bytes must survive native SDK parser failures'
    original_bytes=b'{bad-json}\n\xff'
    class Source:
        async def receive(self,*args,**kwargs):return original_bytes
        async def aclose(self):return 'closed'
    async def exercise():
        captured=[];stream=transport.ObservedBytes(Source(),captured)
        result=await stream.receive(123)
        assert result is original_bytes and captured==[original_bytes]
        assert await stream.aclose()=='closed'
    asyncio.run(exercise())

def test_retained_native_title_is_counted_separately_and_failure_stays_invalid():
    p=profile()
    path=ROOT/'outputs/flashrsi_harnessrisk_4b_private_claude_cli_projection_v8_gpu1_20261115/claude_code_initial_full/cases/action_041/native_model_request_evidence.json'
    if not path.is_file():pytest.skip('retained native title artifact unavailable')
    title=json.loads(path.read_text())['requests'][0]['request']
    assert hasattr(p,'native_request_kind'), 'auxiliary native model requests must remain accounted'
    assert p.native_request_kind(title)=='session_title'
    actual_sid=json.loads(title['metadata']['user_id'])['session_id']
    agent={'model':p.MODEL,'max_tokens':4096,'messages':[{'role':'user','content':'first'}]}
    records=[{'path':'/v1/messages','request':r,'status':200,'response_sha256':'hash'} for r in (title,agent)]
    receipt=p.provider_receipt(records,b'','first',actual_sid,1)
    assert receipt['agent_model_requests']==1 and receipt['auxiliary_model_requests']==1 and receipt['model_requests']==2
    records[0]['status']=500
    with pytest.raises(p.ProfileError):p.provider_receipt(records,b'','first',actual_sid,1)

@pytest.mark.parametrize('corruption',['connections','raw_forwarded','native_metadata'])
def test_more_native_host_and_model_binding_corruption_fails(tmp_path,corruption):
    p=profile();owner,source=complete_case_fixture(tmp_path)
    if corruption=='connections':
        value=json.loads((tmp_path/'native/model.json').read_text());value['connections']=2;p.save_json(tmp_path/'native/model.json',value)
    elif corruption=='raw_forwarded':
        value=json.loads((tmp_path/'protocol_model_evidence.json').read_text());value['requests'][0]['forwarded_request']={'max_tokens':8192};p.save_json(tmp_path/'protocol_model_evidence.json',value)
    else:
        value=json.loads((tmp_path/'native/provider.json').read_text());value[0]['request']['metadata']={'user_id':json.dumps({'session_id':'wrong'})};p.save_json(tmp_path/'native/provider.json',value)
    with pytest.raises(p.ProfileError):p.validate_case(tmp_path,'unit_case','unit_snapshot',[owner],'source',native_source=source)

def test_repeated_agent_history_is_not_counted_as_new_scheduled_exposure():
    p=profile();request={'model':p.MODEL,'max_tokens':4096,'messages':[{'role':'user','content':'first'}]}
    records=[{'path':'/v1/messages','request':copy.deepcopy(request),'status':200,'response_sha256':str(i)} for i in range(2)]
    with pytest.raises(p.ProfileError,match='retry'):p.provider_receipt(records,b'','first',SID,1)

@pytest.mark.parametrize('witness',['decrease','sixteen','unknown'])
def test_post_cap_reset_requires_the_root_authorized_empirical_witness(witness):
    p=profile();prior=p.terminal_receipt(frames(),stop(),close(),SID)
    next_terminal=copy.deepcopy(prior);provider={'model_requests':1,'agent_model_requests':1,'auxiliary_model_requests':0}
    if witness=='decrease':next_terminal['native_outcome']['num_turns']=1
    if witness=='sixteen':provider.update(model_requests=17,agent_model_requests=16,auxiliary_model_requests=1,distinct_scheduled_pairs=16,scheduled_origin_reconciled=True)
    if witness=='unknown':
        with pytest.raises(p.ProfileError,match='native_resumed_counter_reset_structural_barrier'):p.counter_receipt(next_terminal,provider,prior)
    else:assert p.counter_receipt(next_terminal,provider,prior)['native_counter_reset_observed'] is True

@pytest.mark.parametrize('key',['DISABLE_AUTO_COMPACT','CLAUDE_AUTOCOMPACT_PCT_OVERRIDE','CLAUDE_CODE_AUTO_COMPACT_WINDOW','CLAUDE_CODE_MAX_RETRIES','CLAUDE_CODE_MAX_OUTPUT_TOKENS','CLAUDE_CODE_MAX_CONTEXT_TOKENS'])
def test_review_protected_candidate_env_rejected_without_rewrite(tmp_path,key):
    p=profile();from flashrsi_claude_owner_adapter_v9 import build_options
    files=tmp_path/'files';files.mkdir();runtime=tmp_path/'runtime';runtime.mkdir()
    raw=json.dumps({'env':{key:'1'}}).encode();(files/'settings.json').write_bytes(raw)
    with pytest.raises(p.ProfileError,match='protected'):build_options(p.target_config('x'*40,'http://127.0.0.1:1'),files,tmp_path,runtime,None,lambda _:None,[])
    assert (files/'settings.json').read_bytes()==raw

def test_review_native_compaction_enabled_must_match_frozen_policy():
    p=profile()
    with pytest.raises(p.ProfileError):p.context_receipt({'rawMaxTokens':32768,'maxTokens':19768,'model':p.MODEL,'isAutoCompactEnabled':False})

def test_review_actual_pinned_parser_alias_reconciles_without_dropping_raw_fields(tmp_path):
    from claude_agent_sdk._internal.message_parser import parse_message
    from flashrsi_claude_owner_adapter_v9 import plain
    p=profile();owner,source=complete_case_fixture(tmp_path)
    output=json.loads((tmp_path/'native/output.json').read_text());raw=output['raw_frames'][2]
    raw.update(duration_ms=1,duration_api_ms=1,modelUsage={p.MODEL:{'inputTokens':4,'outputTokens':1,'contextWindow':32768}})
    parsed=plain(parse_message(raw));assert parsed['model_usage']==raw['modelUsage']
    output['native_result']=parsed;p.save_json(tmp_path/'native/output.json',output)
    # Build receipts again using the actual wire/parser pair, not a pseudo-wire
    # reconstructed from the parsed snake_case dataclass.
    records=json.loads((tmp_path/'native/provider.json').read_text());archive=(tmp_path/'native/session.jsonl').read_bytes()
    receipt=p.validate_owner(output,b'',archive,owner,1,None,records,'/work',trusted_identity=output['close']['identity'])
    assert receipt['terminal']['native_outcome']['modelUsage']==raw['modelUsage']
    raw['unexplained_future_field']=123
    with pytest.raises(p.ProfileError):p.validate_owner(output,b'',archive,owner,1,None,records,'/work',trusted_identity=output['close']['identity'])

def test_review_identity_observer_requires_live_direct_entry_child(tmp_path):
    import importlib,os,subprocess,sys
    module=importlib.import_module('flashrsi_claude_owner_identity_v9')
    # Only authorized CPU sleeper fixtures; no sandbox/native/source/API launch.
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],close_fds=True)
    try:
        receipt=module.observe_process_identity(child.pid,os.getpid())
        assert receipt['transport_pid']==child.pid and receipt['transport_ppid']==os.getpid()
        assert isinstance(receipt['transport_start'],int)
        with pytest.raises(ValueError):module.observe_process_identity(child.pid,1)
    finally:child.kill();child.wait(timeout=5)

def test_review_transport_does_not_read_proc_in_sandbox():
    import inspect
    from flashrsi_claude_owner_transport_v9 import ObservedTransport
    source=inspect.getsource(ObservedTransport.connect)
    assert "Path('/proc'" not in source
    assert 'identity_observer' in source

@pytest.mark.parametrize('corruption',['pid','start','entry','owner','case','snapshot','nonce','exe'])
def test_review_private_identity_rejects_tampered_or_replayed_scratch_reply(tmp_path,corruption):
    p=profile();owner,source=complete_case_fixture(tmp_path)
    output=json.loads((tmp_path/'native/output.json').read_text());reply=output['close']['identity']
    if corruption=='pid':reply['transport_pid']+=1
    if corruption=='start':reply['transport_start']+=1
    if corruption=='entry':reply['entry_pid']+=1
    if corruption in ('owner','case','snapshot','nonce'):
        key={'owner':'owner_index','case':'case_id','snapshot':'snapshot_id','nonce':'nonce'}[corruption]
        reply['binding'][key]='other'
    if corruption=='exe':reply['executable']['path']='/wrong'
    p.save_json(tmp_path/'native/output.json',output)
    with pytest.raises(p.ProfileError):p.validate_case(tmp_path,'unit_case','unit_snapshot',[owner],'source',native_source=source)

def test_review_identity_response_write_uses_held_directory_fd_after_parent_swap(tmp_path):
    import os
    from flashrsi_claude_owner_identity_v9 import atomic_response
    scratch=tmp_path/'scratch';scratch.mkdir();outside=tmp_path/'outside';outside.mkdir()
    fd=os.open(scratch,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        (scratch/'reply.json').symlink_to(outside/'victim')
        old=tmp_path/'old';scratch.rename(old);scratch.symlink_to(outside,target_is_directory=True)
        atomic_response(fd,'reply.json',{'actual':'receipt'})
        assert json.loads((old/'reply.json').read_text())=={'actual':'receipt'}
        assert not (outside/'victim').exists() and not (outside/'reply.json').exists()
    finally:os.close(fd)

def test_review_identity_ipc_reader_rejects_symlink_and_fifo(tmp_path):
    import os
    from flashrsi_claude_owner_identity_v9 import request_identity
    scratch=tmp_path/'scratch';scratch.mkdir();outside=tmp_path/'outside';outside.write_text('{}')
    reply=scratch/'reply';request=scratch/'request'
    for kind in ('symlink','fifo'):
        if kind=='symlink':reply.symlink_to(outside)
        else:os.mkfifo(reply)
        with pytest.raises(ValueError):asyncio.run(request_identity(2,request,reply,scratch,{'nonce':'n'}))
        reply.unlink()

def test_review_original_h0_settings_policy_is_compatible_without_export():
    import base64
    p=profile();path=ROOT/'outputs/flashrsi_harnessrisk_4b_private_claude_cli_projection_v8_gpu1_20261115/claude_code_initial_full/initial_snapshot.json'
    snapshot=json.loads(path.read_text());raw=base64.b64decode(snapshot['files']['settings.json']['content'])
    settings=json.loads(raw.decode('utf-8-sig'));p.validate_candidate_policy(settings)
    assert settings=={} and snapshot['id']=='26a1fb7c061e3aec310e5fb266d9083c5b5ab8a754dec4b4ae292a1792357f3b'

def test_review_complete_file_ipc_observation_binds_real_owned_child(tmp_path):
    import os,subprocess,sys,time
    from flashrsi_claude_owner_identity_v9 import IdentityObserver,request_identity,validate_identity
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],close_fds=True)
    binding={'case_id':'unit','snapshot_id':'snapshot','owner_index':1,'nonce':'unique-unit-nonce'}
    observer=IdentityObserver(tmp_path/'request',tmp_path/'response',tmp_path,os.getpid(),time.monotonic()+5,
        expected_executable=sys.executable,binding=binding)
    try:
        reply=asyncio.run(request_identity(child.pid,tmp_path/'request',tmp_path/'response',tmp_path,binding))
        trusted=observer.finish()
        assert reply==trusted
        validate_identity(trusted,{'pid':child.pid,'start':trusted['transport_start'],'identity':reply},
            binding=binding,expected_executable=sys.executable)
    finally:
        child.kill();child.wait(timeout=5)
