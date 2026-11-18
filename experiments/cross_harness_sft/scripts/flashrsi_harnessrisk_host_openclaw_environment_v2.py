"""One isolated HarnessRisk case, with a local-only model relay."""
import hashlib, json, os, secrets, signal, subprocess, sys, tempfile, threading, urllib.request, urllib.error
from pathlib import Path
from flashrsi_model_relay import create_model_relay
from flashrsi_native_protocol_relay import create_native_relay
from flashrsi_harnessrisk_privacy import assert_no_oracle, PROTOCOL

ROOT=Path(__file__).resolve().parents[1]
MODEL='qwen35-4b-preflight'

def main():
    if len(sys.argv)!=5:raise SystemExit('usage: host CASE_ID SNAPSHOT_JSON EXPECTED_ID RUN_ROOT')
    case_id,snapshot,expected,run_root=sys.argv[1:]
    if os.environ.get('HR_HARNESS')!='openclaw':raise ValueError('OpenClaw-only environment-session-v2 host')
    model_port=int(os.environ.get('HARNESSRISK_MODEL_PORT','18761'))
    if not 1<=model_port<=65535:raise ValueError('invalid model port')
    model_token=(ROOT/'outputs/preflight_20261111/vllm_api_token').read_text().strip()
    records=[]
    def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    def upstream(request):
        assert_no_oracle(request)
        record={'sequence':len(records)+1,'forwarded_request_sha256':digest(request)}
        records.append(record)
        req=urllib.request.Request(f'http://127.0.0.1:{model_port}/v1/chat/completions',data=json.dumps(request).encode(),
             headers={'Authorization':'Bearer '+model_token,'Content-Type':'application/json'})
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=300) as resp:result=json.load(resp)
        except urllib.error.HTTPError as error:
            raw=error.read(8192)
            record['failure']={'http_status':error.code,'body_sha256':hashlib.sha256(raw).hexdigest(),
                'context_capacity_error':b'maximum context length' in raw or b'maximum model length' in raw}
            raise RuntimeError('native upstream rejected; see trusted protocol_model_evidence.json') from None
        record.update(response_sha256=digest(result),usage=result.get('usage',{}),
                      finish_reason=result.get('choices',[{}])[0].get('finish_reason'))
        return result
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
            p=subprocess.Popen(['unshare','-Urnpf','--mount-proc','sh','-c','ip link set lo up && exec "$1" "$2"','hr',sys.executable,str(ROOT/'scripts/flashrsi_harnessrisk_openclaw_environment_supervisor_v2.py')],env=env,start_new_session=True)
            try:return p.wait(timeout=2400)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait();return 124
        finally:
            evidence=Path(run_root)/case_id
            if evidence.is_dir():
                (evidence/'protocol_model_evidence.json').write_text(json.dumps({'protocol':'private-v1-openclaw-environment-session-v2','requests':records},indent=2)+'\n')
            if native is not None and evidence.is_dir():
                (evidence/'native_model_request_evidence.json').write_text(json.dumps({'protocol':PROTOCOL,'requests':native_requests},ensure_ascii=False)+'\n')
            relay.shutdown();thread.join(timeout=5);relay.server_close()
            if native is not None:native.shutdown();native_thread.join(timeout=5);native.server_close()
if __name__=='__main__':raise SystemExit(main())
