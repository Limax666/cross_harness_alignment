"""Hermes-only private-case supervisor using a copied visible workspace.

Nested mount namespaces are unavailable on this host.  This variant copies the
already audited case workspace into a per-case temporary directory admitted by
the existing sandbox policy, then synchronizes the confined work product back
to the trusted HarnessRisk workspace before export and official scoring.
"""
import json, os, secrets, shutil, subprocess, sys, tempfile, threading, time, urllib.parse
from pathlib import Path

from flashrsi.agents.miniagent import _ENTRY
from flashrsi.core import Snapshot
from flashrsi_gateway import create_gateway
from flashrsi_harnessrisk_privacy import PROTOCOL, assert_no_oracle, audit_case_inputs
from flashrsi_model_relay import relay_request

ROOT=Path(__file__).resolve().parents[1]; PROJECT=ROOT.parents[1]
BENCH=PROJECT/'HarnessRisk'; ADAPTER=BENCH/'harness_adapter/harness_adapter.py'; DATA=BENCH/'data/HarnessRisk'; MODEL='qwen35-4b-preflight'


def prepare_visible_workspace(source, visible):
    """Make a non-symlink private copy for the sandbox's writable root."""
    source,visible=Path(source),Path(visible)
    if visible.exists(): raise FileExistsError(visible)
    shutil.copytree(source,visible,symlinks=False)


def synchronize_workspace(visible, destination):
    """Export only the confined copy's resulting workspace to HarnessRisk."""
    shutil.copytree(Path(visible),Path(destination),dirs_exist_ok=True,symlinks=False)


class Proxy:
    def __init__(self): self.requests=[]; self.responses=[]
    def forward(self,request):
        assert_no_oracle(request); self.requests.append(request)
        result=relay_request(os.environ['DEV_RELAY_PATH'],os.environ['DEV_RELAY_TOKEN'],request)
        self.responses.append(result); return result


class Gate:
    def abort(self): pass
class Dispatcher:
    def dispatch(self,*args): raise ValueError('unexpected tool gateway')


def main():
    if os.getpid()!=1: raise RuntimeError('requires dedicated PID namespace')
    case=os.environ['HR_CASE_ID']; root=Path(os.environ['HR_RUN_ROOT']); root.mkdir(parents=True,exist_ok=True)
    if os.environ.get('HR_HARNESS')!='hermes': raise ValueError('workspace-v9 supervisor is restricted to Hermes')
    task=json.loads((DATA/(case+'.json')).read_text())['task']; owner=task.get('user_messages') or [task['user_message']]
    snapshot=Snapshot.load(os.environ['HR_SNAPSHOT'])
    if snapshot.id!=os.environ['HR_EXPECTED_ID']: raise ValueError('snapshot id mismatch')
    path=root/case
    if path.exists(): raise FileExistsError(f'case path exists: {path}')
    proxy=Proxy(); token=secrets.token_urlsafe(32); server=create_gateway(proxy,Dispatcher(),Gate(),token)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start(); gateway=f'http://127.0.0.1:{server.server_port}'
    with tempfile.TemporaryDirectory(prefix='hr-supervisor-') as temporary:
        tmp=Path(temporary); manager_out=tmp/'manager.out'; manager_err=tmp/'manager.err'
        with manager_out.open('w') as out,manager_err.open('w') as err:
            manager=subprocess.Popen([sys.executable,str(ADAPTER),'serve-mocks','--mock-backend','process','--run-root',str(root),'--run-id',case,'--data-dir',str(DATA),'--case-id',case,'--harness','hermes'],stdout=out,stderr=err,start_new_session=True,env={**os.environ,'HARNESSRISK_PRIVATE_CASE':'1'})
        try:
            info=None; deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                if manager.poll() is not None: raise RuntimeError('mock manager exited: '+manager_err.read_text()[-1200:])
                try: info=json.loads(manager_out.read_text()); break
                except (json.JSONDecodeError,FileNotFoundError): time.sleep(.25)
            if info is None: raise TimeoutError('mock manager startup timed out')
            work=Path(info['workspace']).resolve(); mock_port=urllib.parse.urlsplit(info['gateway']).port
            if work != (path/'workspace').resolve(): raise RuntimeError('unexpected workspace path')
            (path/'privacy_audit.json').write_text(json.dumps(audit_case_inputs(path),indent=2)+'\n')
            scratch=tmp/'scratch'; scratch.mkdir(); visible=tmp/'workspace'; prepare_visible_workspace(work,visible)
            trusted=tmp/'trusted'; trusted.mkdir(); snapshot_file=trusted/'snapshot.json'; snapshot.save(snapshot_file)
            turns=[]; history=[]; status='completed'
            with snapshot._export(full=True) as exported:
                for index,message in enumerate(owner):
                    question=message if not history else 'Prior owner turns and assistant responses:\n'+'\n\n'.join('Owner: '+u+'\nAssistant: '+a for u,a in history)+'\n\nCurrent owner turn:\n'+message
                    config={'python':sys.executable,'entry':str(ROOT/'scripts/flashrsi_harnessrisk_port_entry.py'),'readonly':sorted(set(['/usr','/lib','/lib64','/etc/ssl/certs',str(Path(sys.prefix).resolve()),str(Path(sys.base_prefix).resolve()),str(Path(exported).resolve()),str(ROOT/'scripts'),str(ROOT/'vendor/FlashRSI')])), 'writable':[str(visible),str(scratch)],'ports':[server.server_port,mock_port],'environment':{'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','HOME':str(scratch),'TMPDIR':str(scratch),'MOCK_GATEWAY_BASE_URL':info['gateway']}}
                    request={'harness':'hermes','question':question,'workspace_dir':str(exported),'working_dir':str(visible),'config':{'context_length':16384,'llm':{'model_name':MODEL,'api_key':token,'base_url':gateway+'/v1','context_length':16384,'max_tokens':4096,'temperature':0,'seed':0,'extra_body':{'chat_template_kwargs':{'enable_thinking':False}},'stream':False},'max_turns':12,'timeout':360}}
                    child=subprocess.run([sys.executable,str(ROOT/'scripts/flashrsi_harnessrisk_sandbox.py'),'-I','-B',str(ROOT/'scripts/flashrsi_harnessrisk_port_entry.py')],input=json.dumps(request),text=True,capture_output=True,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','HARNESSRISK_SANDBOX_CONFIG':json.dumps(config)},timeout=360)
                    if child.returncode: raise RuntimeError(f'model sandbox exited {child.returncode}: '+child.stderr[-2000:])
                    result=json.loads(child.stdout); answer=result.get('answer',''); history.append((message,answer)); turns.append({'owner_turn':index+1,'prompt':message,'response_text':answer,'stop_reason':result.get('stop_reason'),'messages':result.get('messages',[]),'usage':result.get('usage')})
                    if result.get('stop_reason')!='completed': status='error'; break
            synchronize_workspace(visible,work)
            (path/'trajectory'/'hermes_turns.jsonl').write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in turns))
            (path/'harness_result.json').write_text(json.dumps({'status':status,'process_exit_code':0 if status=='completed' else 1,'harness':'flashrsi_hermes','snapshot_id':snapshot.id,'turns':turns,'privacy_protocol':PROTOCOL,'multiturn_adapter':'prior owner and assistant text replay; tool state preserved'},ensure_ascii=False,indent=2)+'\n')
            (path/'model_request_evidence.json').write_text(json.dumps({'requests':proxy.requests,'responses':proxy.responses},ensure_ascii=False)+'\n')
            result=subprocess.run([sys.executable,str(ADAPTER),'export',str(path),'--case-id',case,'--harness','hermes'],capture_output=True,text=True,timeout=60)
            if result.returncode: raise RuntimeError('trajectory export failed: '+result.stderr[-1000:])
            evaluated=subprocess.run([sys.executable,str(BENCH/'harness_adapter/scripts/evaluate_run.py'),str(path),'--data-dir',str(DATA),'--output',str(path/'official_rule_evaluation.json'),'--pretty'],capture_output=True,text=True,timeout=60)
            if evaluated.returncode: raise RuntimeError('official evaluator failed: '+evaluated.stderr[-1000:])
            print(json.dumps({'case_id':case,'status':status,'snapshot_id':snapshot.id,'official_rules':str(path/'official_rule_evaluation.json')}),flush=True); return 0 if status=='completed' else 2
        finally:
            manager.terminate()
            try: manager.wait(timeout=30)
            except subprocess.TimeoutExpired: manager.kill(); manager.wait()
            server.shutdown(); thread.join(timeout=5); server.server_close()


if __name__=='__main__': raise SystemExit(main())
