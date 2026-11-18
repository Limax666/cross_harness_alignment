"""Local-only model relay and PID-namespace watchdog for native probes."""
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path
from flashrsi_native_protocol_relay import create_native_relay

ROOT=Path(__file__).resolve().parents[1]
MODEL='qwen35-4b-preflight'
token=(ROOT/'outputs/preflight_20261111/vllm_api_token').read_text().strip()


def upstream(route,body):
    request=urllib.request.Request('http://127.0.0.1:18761'+route,data=json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json','anthropic-version':'2023-06-01'})
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=300) as response:
            return response.status,response.headers.get_content_type(),response.read(16*1024*1024+1)
    except urllib.error.HTTPError as error:
        return error.code,error.headers.get_content_type(),error.read(16*1024*1024+1)


def main():
    with tempfile.TemporaryDirectory(prefix='rsi-native-relay-') as directory:
        path=str(Path(directory)/'model.sock');relay_token=secrets.token_urlsafe(32)
        relay=create_native_relay(path,relay_token,MODEL,upstream)
        thread=threading.Thread(target=relay.serve_forever,daemon=True);thread.start()
        env={'PATH':'/usr/sbin:/usr/bin:/bin','LANG':'C.UTF-8',
             'PYTHONPATH':':'.join(str(ROOT/x) for x in ('src','scripts','vendor/FlashRSI')),
             'DEV_RELAY_PATH':path,'DEV_RELAY_TOKEN':relay_token,'NATIVE_HARNESS':os.environ['NATIVE_HARNESS'],
             'B0_TASK_ID':os.environ['B0_TASK_ID'],'B0_SEED':os.environ['B0_SEED'],'B0_CASE_OUT':os.environ['B0_CASE_OUT'],
             'FLASHRSI_COMPLETION_BUDGET':os.environ.get('FLASHRSI_COMPLETION_BUDGET','4096')}
        for key in ('B1_SNAPSHOT_JSON','B1_EXPECTED_SNAPSHOT_ID'):
            if key in os.environ:env[key]=os.environ[key]
        try:
            process=subprocess.Popen(['unshare','-Urnpf','--mount-proc','sh','-c',
                'ip link set lo up && exec "$1" "$2"','rsi',sys.executable,str(ROOT/'scripts/flashrsi_native_supervisor.py')],
                env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
            try:stdout,stderr=process.communicate(timeout=360)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);stdout,stderr=process.communicate()
                raise TimeoutError('native PID namespace exceeded deadline')
            print(json.dumps({'supervisor_exit':process.returncode,'stdout':stdout[-3000:],
                              'stderr':stderr[-3000:].replace(relay_token,'[redacted]').replace(token,'[redacted]')}))
            return process.returncode
        finally:
            relay.shutdown();thread.join(timeout=5);relay.server_close()


if __name__=='__main__':raise SystemExit(main())
