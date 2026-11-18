import json
import os
from pathlib import Path
import subprocess
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer


def test_stdio_mcp_exposes_fixed_tools_and_forwards_calls(tmp_path):
    token='x'*40;seen=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            assert self.path=='/tool' and self.headers['Authorization']=='Bearer '+token
            seen.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            data=b'{"observation":"today"}';self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    schemas=tmp_path/'schemas.json';schemas.write_text(json.dumps([{'function':{'name':'get_day','parameters':{'type':'object'}}}]))
    script=Path(__file__).resolve().parents[1]/'scripts/flashrsi_native_mcp_stdio.py'
    env={**os.environ,'FLASHRSI_GATEWAY_URL':f'http://127.0.0.1:{server.server_port}','FLASHRSI_GATEWAY_TOKEN':token,'FLASHRSI_NATIVE_SCHEMAS':str(schemas)}
    lines=[{'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-03-26'}},
           {'jsonrpc':'2.0','id':2,'method':'tools/list'},
           {'jsonrpc':'2.0','id':3,'method':'tools/call','params':{'name':'get_day','arguments':{}}}]
    try:
        result=subprocess.run(['/usr/bin/python3',str(script)],input=''.join(json.dumps(x)+'\n' for x in lines),env=env,text=True,capture_output=True,check=True,timeout=10)
        replies=[json.loads(line) for line in result.stdout.splitlines()]
        assert replies[1]['result']['tools'][0]['name']=='get_day'
        assert replies[2]['result']['content'][0]['text']=='today'
        assert seen==[{'name':'get_day','arguments':{}}]
    finally:server.shutdown();thread.join();server.server_close()
