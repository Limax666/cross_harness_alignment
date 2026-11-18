"""Run a FlashRSI MiniAgent snapshot in one confined HarnessRisk case."""
import json, os, secrets, subprocess, sys, tempfile, threading, time, urllib.parse
from pathlib import Path
from flashrsi.agents.miniagent import _ENTRY
from flashrsi.core import Snapshot
from flashrsi_gateway import create_gateway
from flashrsi_model_relay import relay_request
from flashrsi_native_protocol_relay import create_native_gateway
from flashrsi_harnessrisk_privacy import PROTOCOL, assert_no_oracle, audit_case_inputs

ROOT=Path(__file__).resolve().parents[1]; PROJECT=ROOT.parents[1]
BENCH=PROJECT/'HarnessRisk'; ADAPTER=BENCH/'harness_adapter/harness_adapter.py'
DATA=BENCH/'data/HarnessRisk'; MODEL='qwen35-4b-preflight'

class Proxy:
    def __init__(self):self.requests=[];self.responses=[]
    def forward(self,request):
        assert_no_oracle(request)
        self.requests.append(request)
        result=relay_request(os.environ['DEV_RELAY_PATH'],os.environ['DEV_RELAY_TOKEN'],request)
        self.responses.append(result);return result
class Gate:
    def abort(self):pass
class Dispatcher:
    def dispatch(self,*args):raise ValueError('unexpected tool gateway')

def main():
    if os.getpid()!=1:raise RuntimeError('requires dedicated PID namespace')
    case=os.environ['HR_CASE_ID'];harness=os.environ.get('HR_HARNESS','miniagent');root=Path(os.environ['HR_RUN_ROOT']);root.mkdir(parents=True,exist_ok=True)
    if harness not in ('miniagent','opencode','openclaw','hermes','pi','claude_code','codex'):raise ValueError('unsupported harness')
    data=json.loads((DATA/(case+'.json')).read_text());task=data['task'];owner=task.get('user_messages') or [task['user_message']]
    snapshot=Snapshot.load(os.environ['HR_SNAPSHOT'])
    if snapshot.id!=os.environ['HR_EXPECTED_ID']:raise ValueError('snapshot id mismatch')
    path=root/case
    if path.exists():raise FileExistsError(f'case path exists: {path}')
    proxy=Proxy();token=secrets.token_urlsafe(32);server=create_gateway(proxy,Dispatcher(),Gate(),token)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    gateway=f'http://127.0.0.1:{server.server_port}'
    native_server=None;native_thread=None
    if harness in ('claude_code','codex'):
        native_server=create_native_gateway(os.environ['NATIVE_RELAY_PATH'],os.environ['NATIVE_RELAY_TOKEN'],token,MODEL,completion_budget=4096)
        native_thread=threading.Thread(target=native_server.serve_forever,daemon=True);native_thread.start()
        gateway=f'http://127.0.0.1:{native_server.server_port}'
    with tempfile.TemporaryDirectory(prefix='hr-supervisor-') as temporary:
        tmp=Path(temporary);manager_out=tmp/'manager.out';manager_err=tmp/'manager.err'
        with manager_out.open('w') as out,manager_err.open('w') as err:
            manager=subprocess.Popen([sys.executable,str(ADAPTER),'serve-mocks','--mock-backend','process','--run-root',str(root),
                '--run-id',case,'--data-dir',str(DATA),'--case-id',case,'--harness',harness],stdout=out,stderr=err,start_new_session=True,
                env={**os.environ,'HARNESSRISK_PRIVATE_CASE':'1'})
        try:
            deadline=time.monotonic()+90;info=None
            while time.monotonic()<deadline:
                if manager.poll() is not None:raise RuntimeError('mock manager exited: '+manager_err.read_text()[-1200:])
                try:info=json.loads(manager_out.read_text());break
                except (json.JSONDecodeError,FileNotFoundError):time.sleep(.25)
            if info is None:raise TimeoutError('mock manager startup timed out')
            work=Path(info['workspace']).resolve();mock_port=urllib.parse.urlsplit(info['gateway']).port
            if work != (path/'workspace').resolve():raise RuntimeError('unexpected workspace path')
            audit=audit_case_inputs(path)
            (path/'privacy_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
            scratch=tmp/'scratch';scratch.mkdir();trusted=tmp/'trusted';trusted.mkdir();snapshot_file=trusted/'snapshot.json';snapshot.save(snapshot_file)
            turns=[];history=[];status='completed'
            with snapshot._export(full=True) as exported:
                for index,message in enumerate(owner):
                    question=message if not history else ('Prior owner turns and assistant responses:\n'+
                       '\n\n'.join('Owner: '+u+'\nAssistant: '+a for u,a in history)+'\n\nCurrent owner turn:\n'+message)
                    entry=(str(Path(_ENTRY).resolve()) if harness=='miniagent' else str(ROOT/'scripts'/('flashrsi_harnessrisk_claude_entry.py' if harness in ('claude_code','codex') else 'flashrsi_harnessrisk_port_entry.py')));python=sys.executable
                    ro=sorted(set(['/usr','/lib','/lib64','/etc/ssl/certs',str(Path(sys.prefix).resolve()),
                                   str(Path(sys.base_prefix).resolve()),str(Path(exported).resolve()),str(ROOT/'scripts'),
                                   str(ROOT/'vendor/FlashRSI')]))
                    if harness=='claude_code':ro.extend([str(trusted),'/data/home/liumingxiao/.npm-global/lib/node_modules/@anthropic-ai/claude-code/bin'])
                    if harness=='codex':ro.append(str(trusted))
                    env={'PATH':str(work/'bin')+':/usr/bin:/bin','LANG':'C.UTF-8','HOME':str(scratch),
                         'TMPDIR':str(scratch),'MOCK_GATEWAY_BASE_URL':info['gateway']}
                    config={'python':python,'entry':entry,'readonly':ro,'writable':[str(work),str(scratch)],
                            'ports':[native_server.server_port if native_server else server.server_port,mock_port],'environment':env}
                    launcher_env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','HARNESSRISK_SANDBOX_CONFIG':json.dumps(config)}
                    request={'question':question,'llm_config':{'model_name':MODEL,'api_key':token,'base_url':gateway+'/v1',
                              'max_tokens':4096,'temperature':0,'seed':0,'extra_body':{'chat_template_kwargs':{'enable_thinking':False}}},
                             'web_config':{},'debug_llm_io':False,'max_turns':12,'workspace_dir':str(exported),
                             'runs_root':str(scratch/'runs'),'working_dir':str(work)}
                    if harness in ('claude_code','codex'):
                        if harness=='claude_code':
                            config_native={'llm':{'model_name':MODEL,'api_key':token,'base_url':gateway,'max_tokens':4096,'max_retries':0},
                              'cli_path':'/data/home/liumingxiao/.npm-global/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe',
                              'allowed_tools':['Read','Glob','Grep','Bash','Write','Edit'],'permission_mode':'dontAsk','max_turns':12,'timeout':360}
                        else:
                            config_native={'llm':{'model_name':MODEL,'api_key':token,'base_url':gateway+'/v1','provider_name':'FlashRSI local'},
                                           'sandbox':'workspace-write','timeout':360}
                        request={'harness':harness,'question':question,'workspace_dir':str(exported),'working_dir':str(work),
                                 'snapshot_json':str(snapshot_file),'config':config_native}
                    elif harness!='miniagent':
                        if harness=='pi':request['llm_config']['provider_name']='openai'
                        request['llm_config']['stream']=False
                        request={'harness':harness,'question':question,'workspace_dir':str(exported),'working_dir':str(work),
                                 'config':{'llm':request['llm_config'],'max_turns':12,'timeout':360}}
                    child=subprocess.run([str(ROOT/'scripts/flashrsi_harnessrisk_sandbox.py'),'-I','-B',entry],
                         input=json.dumps(request),text=True,capture_output=True,env=launcher_env,timeout=360)
                    if child.returncode:raise RuntimeError(f'model sandbox exited {child.returncode}: '+child.stderr[-2000:])
                    result=json.loads(child.stdout);answer=result.get('answer','');history.append((message,answer))
                    turns.append({'owner_turn':index+1,'prompt':message,'response_text':answer,'stop_reason':result.get('stop_reason'),
                                  'messages':result.get('messages',[]),'usage':result.get('usage')})
                    if result.get('stop_reason')!='completed':status='error';break
            (path/'trajectory'/'hermes_turns.jsonl').write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in turns))
            (path/'harness_result.json').write_text(json.dumps({'status':status,'process_exit_code':0 if status=='completed' else 1,
                'harness':'flashrsi_'+harness,'snapshot_id':snapshot.id,'turns':turns,
                'privacy_protocol':PROTOCOL,'multiturn_adapter':'prior owner and assistant text replay; tool state preserved'},ensure_ascii=False,indent=2)+'\n')
            (path/'model_request_evidence.json').write_text(json.dumps({'requests':proxy.requests,'responses':proxy.responses},ensure_ascii=False)+'\n')
            result=subprocess.run([sys.executable,str(ADAPTER),'export',str(path),'--case-id',case,'--harness',harness],capture_output=True,text=True,timeout=60)
            if result.returncode:raise RuntimeError('trajectory export failed: '+result.stderr[-1000:])
            evaluated=subprocess.run([sys.executable,str(BENCH/'harness_adapter/scripts/evaluate_run.py'),str(path),'--data-dir',str(DATA),
                      '--output',str(path/'official_rule_evaluation.json'),'--pretty'],capture_output=True,text=True,timeout=60)
            if evaluated.returncode:raise RuntimeError('official evaluator failed: '+evaluated.stderr[-1000:])
            print(json.dumps({'case_id':case,'status':status,'snapshot_id':snapshot.id,'official_rules':str(path/'official_rule_evaluation.json')}),flush=True)
            return 0 if status=='completed' else 2
        finally:
            manager.terminate()
            try:manager.wait(timeout=30)
            except subprocess.TimeoutExpired:manager.kill();manager.wait()
            server.shutdown();thread.join(timeout=5);server.server_close()
            if native_server is not None:native_server.shutdown();native_thread.join(timeout=5);native_server.server_close()
if __name__=='__main__':raise SystemExit(main())
