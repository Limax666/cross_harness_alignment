"""One isolated HarnessRisk case, with a local-only model relay."""
import base64, copy, json, os, secrets, signal, subprocess, sys, tempfile, threading, urllib.request, urllib.error
from pathlib import Path
from flashrsi_model_relay import create_model_relay
from flashrsi_native_protocol_relay import create_native_relay
from flashrsi_harnessrisk_privacy import assert_no_oracle, PROTOCOL

ROOT=Path(__file__).resolve().parents[1]
MODEL='qwen35-4b-preflight'

def main():
    if len(sys.argv)!=5:raise SystemExit('usage: host CASE_ID SNAPSHOT_JSON EXPECTED_ID RUN_ROOT')
    case_id,snapshot,expected,run_root=sys.argv[1:]
    model_port=int(os.environ.get('HARNESSRISK_MODEL_PORT','18761'))
    if not 1<=model_port<=65535:raise ValueError('invalid model port')
    model_token=(ROOT/'outputs/preflight_20261111/vllm_api_token').read_text().strip()
    adjustments=[]
    def upstream(request):
        current=copy.deepcopy(request)
        for message in current.get('messages',[]):
            content=message.get('content')
            if not isinstance(content,list):continue
            for index,part in enumerate(content):
                if not isinstance(part,dict) or part.get('type')!='file':continue
                file=part.get('file') or {};data=file.get('file_data','')
                if not isinstance(data,str) or not data.startswith('data:application/pdf;base64,'):
                    raise ValueError('unsupported attachment format')
                raw=base64.b64decode(data.split(',',1)[1],validate=True)
                if len(raw)>1048576 or raw.startswith(b'%PDF-'):raise ValueError('binary PDF not supported')
                decoded=raw.decode('utf-8')
                if any(not (c.isprintable() or c in '\n\r\t') for c in decoded):raise ValueError('attachment is not printable UTF-8')
                assert_no_oracle(decoded)
                content[index]={'type':'text','text':'[Text attachment, declared application/pdf]\n'+decoded}
                adjustments.append({'reason':'mislabelled_plain_text_attachment','original_type':'application/pdf','encoding':'utf-8','bytes':len(raw)})
        budgets=[int(current.get('max_tokens',8192)),4096,2048,1024,512]
        seen=set()
        for budget in budgets:
            if budget in seen: continue
            seen.add(budget);current['max_tokens']=budget
            req=urllib.request.Request(f'http://127.0.0.1:{model_port}/v1/chat/completions',data=json.dumps(current).encode(),
                 headers={'Authorization':'Bearer '+model_token,'Content-Type':'application/json'})
            try:
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=300) as resp:
                    if budget != budgets[0]: adjustments.append({'original_max_tokens':budgets[0],'applied_max_tokens':budget,'reason':'context_limit'})
                    return json.load(resp)
            except urllib.error.HTTPError as error:
                detail=error.read(4096).decode('utf-8',errors='replace')
                if error.code != 400 or "maximum context length" not in detail:raise
        raise RuntimeError('context exceeds capacity even at minimum output budget')
    with tempfile.TemporaryDirectory(prefix='hr-relay-') as d:
        home=Path(d)/'home';home.mkdir()
        path=str(Path(d)/'model.sock');token=secrets.token_urlsafe(32)
        relay=create_model_relay(path,token,MODEL,upstream);thread=threading.Thread(target=relay.serve_forever,daemon=True);thread.start()
        native=None;native_thread=None;native_requests=[]
        if os.environ.get('HR_HARNESS') in ('claude_code','codex'):
            native_path=str(Path(d)/'native.sock');native_token=secrets.token_urlsafe(32)
            def native_upstream(route,body):
                assert_no_oracle(body)
                native_requests.append({'path':route,'request':body})
                req=urllib.request.Request(f'http://127.0.0.1:{model_port}'+route,data=json.dumps(body).encode(),
                    headers={'Authorization':'Bearer '+model_token,'Content-Type':'application/json','anthropic-version':'2023-06-01'})
                try:
                    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=300) as resp:
                        return resp.status,resp.headers.get_content_type(),resp.read(16*1024*1024+1)
                except urllib.error.HTTPError as error:
                    return error.code,error.headers.get_content_type(),error.read(16*1024*1024+1)
            native=create_native_relay(native_path,native_token,MODEL,native_upstream)
            native_thread=threading.Thread(target=native.serve_forever,daemon=True);native_thread.start()
        env={'PATH':'/usr/sbin:/usr/bin:/bin','LANG':'C.UTF-8','HOME':str(home),'PYTHONPATH':':'.join(str(ROOT/x) for x in ('src','scripts','vendor/FlashRSI')),
             'DEV_RELAY_PATH':path,'DEV_RELAY_TOKEN':token,'HR_CASE_ID':case_id,'HR_SNAPSHOT':str(Path(snapshot).resolve()),
             'HR_EXPECTED_ID':expected,'HR_RUN_ROOT':str(Path(run_root).resolve()),
             'HR_HARNESS':os.environ.get('HR_HARNESS','miniagent')}
        if native is not None:env.update(NATIVE_RELAY_PATH=native_path,NATIVE_RELAY_TOKEN=native_token)
        try:
            p=subprocess.Popen(['unshare','-Urnpf','--mount-proc','sh','-c','ip link set lo up && exec "$1" "$2"','hr',sys.executable,str(ROOT/'scripts/flashrsi_harnessrisk_supervisor_8192.py')],env=env,start_new_session=True)
            try:return p.wait(timeout=2400)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait();return 124
        finally:
            evidence=Path(run_root)/case_id
            if evidence.is_dir():
                (evidence/'adaptive_context_adjustments.json').write_text(json.dumps({'protocol':'private-v1-target8192-text-adaptive-recovery','adjustments':adjustments},indent=2)+'\n')
            if native is not None and evidence.is_dir():
                (evidence/'native_model_request_evidence.json').write_text(json.dumps({'protocol':PROTOCOL,'requests':native_requests},ensure_ascii=False)+'\n')
            relay.shutdown();thread.join(timeout=5);relay.server_close()
            if native is not None:native.shutdown();native_thread.join(timeout=5);native.server_close()
if __name__=='__main__':raise SystemExit(main())
