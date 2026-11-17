"""NEW Claude native profile; one GPU1 route, one attempt, original confinement."""
from __future__ import annotations

import base64
import copy
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request

from flashrsi_claude_owner_profile_v9 import PROFILE, MODEL, ProfileError, digest, byte_hash, save_json, project_cli_history
ROOT=Path(__file__).resolve().parents[1]


def prepare_request(route,body):
    """Frozen native reference projection; preserve genuine repeated owners."""
    from flashrsi_harnessrisk_privacy import assert_no_oracle
    assert_no_oracle(body)
    projected,removed=project_cli_history(body) if route=='/v1/messages' else (copy.deepcopy(body),[])
    conversions=[]
    def convert(value):
        if isinstance(value,dict):
            if value.get('type')=='tool_reference':
                name=value.get('tool_name')
                if not isinstance(name,str) or not name:raise ProfileError('invalid_native_tool_reference')
                conversions.append({'reason':'native_tool_reference_to_text','tool_name':name})
                return {'type':'text','text':'[Tool reference: '+name+']'}
            return {k:convert(v) for k,v in value.items()}
        if isinstance(value,list):return [convert(v) for v in value]
        return value
    current=convert(projected)
    if route=='/v1/messages':
        if type(current.get('max_tokens')) is not int or not 1<=current['max_tokens']<=4096:
            raise ProfileError('native_wire_output_budget_changed')
        current['temperature']=0
    return current,removed,conversions


def main():
    if len(sys.argv)!=5 or os.environ.get('HR_HARNESS')!='claude_code':
        raise ProfileError('usage: Claude-only host CASE SNAPSHOT EXPECTED_ID RUN_ROOT')
    if os.environ.get('HARNESSRISK_MODEL_PORT')!='18764':raise ProfileError('GPU1_18764_only')
    from flashrsi_native_protocol_relay import create_native_relay
    case,snapshot,expected,run_root=sys.argv[1:]
    model_token=(ROOT/'outputs/preflight_20261111/vllm_api_token').read_text().strip()
    records=[];lock=threading.Lock()
    def upstream(route,body):
        with lock:
            record={'sequence':len(records)+1,'path':route,'raw_request':copy.deepcopy(body),
                    'raw_request_sha256':digest(body),'settled':False}
            records.append(record)
        try:
            current,removed,conversions=prepare_request(route,body)
            record.update(forwarded_request=current,forwarded_request_sha256=digest(current),
                          projection_removed=removed,reference_conversions=conversions)
            request=urllib.request.Request('http://127.0.0.1:18764'+route,data=json.dumps(current).encode(),
                headers={'Authorization':'Bearer '+model_token,'Content-Type':'application/json','anthropic-version':'2023-06-01'})
            try:
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=300) as response:
                    status,ctype,payload=response.status,response.headers.get_content_type(),response.read(16*1024*1024+1)
            except urllib.error.HTTPError as error:
                status,ctype,payload=error.code,error.headers.get_content_type(),error.read(16*1024*1024+1)
            record.update(status=status,content_type=ctype,response_sha256=byte_hash(payload),
                response_body_b64=base64.b64encode(payload).decode(),
                context_capacity_error=b'maximum context length' in payload or b'maximum model length' in payload)
            return status,ctype,payload  # One actual attempt, no smaller-reply/context retry.
        except Exception as exc:
            record['failure']={'type':type(exc).__name__,'detail':str(exc)[:500]}
            raise
        finally:
            with lock:record['settled']=True
    with tempfile.TemporaryDirectory(prefix='hr-relay-') as temporary:
        tmp=Path(temporary);home=tmp/'home';home.mkdir()
        relay_path,relay_token=str(tmp/'native.sock'),secrets.token_urlsafe(32)
        relay=create_native_relay(relay_path,relay_token,MODEL,upstream)
        thread=threading.Thread(target=relay.serve_forever,daemon=True);thread.start()
        env={'PATH':'/usr/sbin:/usr/bin:/bin','LANG':'C.UTF-8','HOME':str(home),
            'PYTHONPATH':':'.join(str(ROOT/p) for p in ('src','scripts','vendor/FlashRSI')),
            'NATIVE_RELAY_PATH':relay_path,'NATIVE_RELAY_TOKEN':relay_token,
            'HR_CASE_ID':case,'HR_SNAPSHOT':str(Path(snapshot).resolve()),'HR_EXPECTED_ID':expected,
            'HR_RUN_ROOT':str(Path(run_root).resolve()),'HR_HARNESS':'claude_code'}
        proof={'profile':PROFILE,'case_id':case,'snapshot_id':expected,'whole_host_seconds':2400,
               'timed_out':False,'namespace_exited':False,'namespace_parent_death_signal':'SIGKILL'}
        process=None
        try:
            process=subprocess.Popen(['unshare','-Urnpf','--kill-child=SIGKILL','--mount-proc','sh','-c',
                'ip link set lo up && exec "$1" "$2"','hr',sys.executable,
                str(ROOT/'scripts/flashrsi_harnessrisk_claude_native_owner_supervisor_v9.py')],env=env,start_new_session=True)
            proof['namespace_launcher_pid']=process.pid
            try:code=process.wait(timeout=2400)
            except subprocess.TimeoutExpired:
                proof['timed_out']=True;os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=30);code=124
            proof.update(returncode=code,namespace_exited=process.poll() is not None)
            return code
        finally:
            if process is not None and process.poll() is None:
                try:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=30)
                except (OSError,subprocess.TimeoutExpired):proof['cleanup_confirmed']=False
            if process is not None:proof['namespace_exited']=process.poll() is not None
            relay.shutdown();thread.join(timeout=5);relay.server_close()
            with lock:
                proof['model_requests_settled']=all(r.get('settled') is True for r in records)
                retained=copy.deepcopy(records)
            path=Path(run_root)/case
            if path.is_dir():
                save_json(path/'protocol_model_evidence.json',{'profile':PROFILE,'requests':retained})
                save_json(path/'native_owner_host.json',proof)

if __name__=='__main__':raise SystemExit(main())
