import json,os,secrets,subprocess,sys,tempfile,threading,urllib.request
from pathlib import Path
from flashrsi_model_relay import create_model_relay
root=Path(__file__).resolve().parents[1]
model_token=(root/'outputs/preflight_20261111/vllm_api_token').read_text().strip()
def upstream(request):
 req=urllib.request.Request('http://127.0.0.1:18761/v1/chat/completions',data=json.dumps(request).encode(),headers={'Authorization':'Bearer '+model_token,'Content-Type':'application/json'})
 with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=120) as response: return json.load(response)
with tempfile.TemporaryDirectory(prefix='rsi-relay-') as path:
 sock=str(Path(path)/'model.sock'); token=secrets.token_urlsafe(32)
 server=create_model_relay(sock,token,'qwen35-4b-preflight',upstream)
 thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
 env={'PATH':'/usr/sbin:/usr/bin:/bin','LANG':'C.UTF-8','PYTHONPATH':':'.join(str(root/x) for x in ('src','scripts','vendor/FlashRSI')),'DEV_RELAY_PATH':sock,'DEV_RELAY_TOKEN':token,'B0_TASK_ID':os.environ['B0_TASK_ID'],'B0_SEED':os.environ['B0_SEED'],'B0_CASE_OUT':os.environ['B0_CASE_OUT'],'FLASHRSI_COMPLETION_BUDGET':os.environ.get('FLASHRSI_COMPLETION_BUDGET','512')}
 for key in ('B1_SNAPSHOT_JSON','B1_EXPECTED_SNAPSHOT_ID'):
  if key in os.environ: env[key]=os.environ[key]
 try:
  result=subprocess.run(['unshare','-Urn','sh','-c','ip link set lo up && exec "$1" "$2"','rsi',sys.executable,str(root/'scripts/flashrsi_openclaw_full_supervisor.py')],env=env,timeout=360,text=True,capture_output=True)
  print(json.dumps({'supervisor_exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr[-3000:]}))
 finally:
  server.shutdown(); thread.join(timeout=5); server.server_close()
