"""Bounded Responses/Messages relay for isolated native-harness development.

This transport alone does not attest tool calls or qualify an evaluation.
"""
import base64
import hmac
import json
import os
from pathlib import Path
import socket
import socketserver
import stat
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALLOWED=frozenset({'/v1/responses','/v1/messages','/v1/messages/count_tokens'})
MAX_REQUEST=4*1024*1024
MAX_RESPONSE=16*1024*1024


def _token_ok(token):
    return isinstance(token,str) and len(token)>=32 and token.isascii() and not any(c.isspace() for c in token)


def create_native_relay(path,token,model,upstream):
    if not _token_ok(token) or not callable(upstream) or not isinstance(model,str) or not model:
        raise ValueError('invalid relay configuration')
    if not isinstance(path,str) or not os.path.isabs(path) or len(os.fsencode(path))>100 or os.path.lexists(path):
        raise ValueError('invalid relay path')
    parent=Path(path).parent
    if os.path.realpath(parent)!=str(parent):raise ValueError('noncanonical relay directory')
    info=parent.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.geteuid() or stat.S_IMODE(info.st_mode)&0o077:
        raise ValueError('relay directory must be private')
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.request.settimeout(300)
            try:
                line=self.rfile.readline(MAX_REQUEST+1)
                if len(line)>MAX_REQUEST or not line.endswith(b'\n'):raise ValueError('bad frame')
                payload=json.loads(line)
                if set(payload)!={'token','path','request'} or not isinstance(payload['token'],str) or not hmac.compare_digest(payload['token'],token):raise ValueError('bad capability')
                if payload['path'] not in ALLOWED or not isinstance(payload['request'],dict) or payload['request'].get('model')!=model:raise ValueError('bad request')
                status,content_type,body=upstream(payload['path'],payload['request'])
                if type(status) is not int or not 100<=status<=599 or content_type not in ('application/json','text/event-stream') or not isinstance(body,bytes) or len(body)>MAX_RESPONSE:raise ValueError('bad response')
                response={'ok':True,'status':status,'content_type':content_type,'body':base64.b64encode(body).decode('ascii')}
            except Exception:
                response={'ok':False}
            self.wfile.write(json.dumps(response,separators=(',',':')).encode()+b'\n')
    class Server(socketserver.ThreadingUnixStreamServer):daemon_threads=True
    server=Server(path,Handler);os.chmod(path,0o600);return server


def native_relay_request(path,token,route,request):
    if not _token_ok(token) or route not in ALLOWED or not isinstance(request,dict):raise ValueError('bad relay request')
    raw=json.dumps({'token':token,'path':route,'request':request},ensure_ascii=True,allow_nan=False).encode()+b'\n'
    if len(raw)>MAX_REQUEST:raise ValueError('request too large')
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as conn:
        conn.settimeout(300);conn.connect(path);conn.sendall(raw)
        with conn.makefile('rb') as stream:line=stream.readline(MAX_RESPONSE*2+1)
    if not line.endswith(b'\n') or len(line)>MAX_RESPONSE*2:raise ValueError('bad relay response')
    result=json.loads(line)
    if result.get('ok') is not True:raise ValueError('relay rejected request')
    body=base64.b64decode(result['body'],validate=True)
    if len(body)>MAX_RESPONSE:raise ValueError('response too large')
    return result['status'],result['content_type'],body


def create_native_gateway(relay_path,relay_token,child_token,model,*,gate=None,dispatch=None,completion_budget=None):
    if not _token_ok(child_token):raise ValueError('invalid gateway capability')
    if (gate is None)!=(dispatch is None):raise ValueError('gate and tool dispatcher must be supplied together')
    if completion_budget is not None and completion_budget not in (512,1024,2048,4096):raise ValueError('invalid completion budget')
    errors=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            auth=self.headers.get_all('Authorization',[]);keys=self.headers.get_all('x-api-key',[])
            allowed=(len(auth)==1 and hmac.compare_digest(auth[0],'Bearer '+child_token)) or (len(keys)==1 and hmac.compare_digest(keys[0],child_token))
            if not allowed:self.send_error(401);return
            try:
                parsed=urllib.parse.urlsplit(self.path);route=parsed.path
                if parsed.query not in ('','beta=true') or parsed.fragment or route not in ALLOWED|{'/tool'} or self.headers.get_content_type()!='application/json' or self.headers.get('Transfer-Encoding') is not None:raise ValueError('invalid request')
                lengths=self.headers.get_all('Content-Length',[])
                if len(lengths)!=1 or not lengths[0].isdigit() or not 0<int(lengths[0])<=MAX_REQUEST:raise ValueError('invalid length')
                body=self.rfile.read(int(lengths[0]));request=json.loads(body)
                if not isinstance(request,dict):raise ValueError('invalid request object')
                if route=='/tool':
                    if gate is None or set(request)!={'name','arguments'} or not isinstance(request['name'],str) or not isinstance(request['arguments'],dict):raise ValueError('invalid tool request')
                    output=gate.dispatch(request['name'],request['arguments'],dispatch)
                    if not isinstance(output,dict) or not isinstance(output.get('observation'),str):raise ValueError('invalid tool result')
                    status,content_type,result=200,'application/json',json.dumps({'observation':output['observation']},ensure_ascii=True).encode()
                else:
                    if request.get('model')!=model:raise ValueError('unexpected model')
                    if route=='/v1/responses' and request.get('tools'):request['parallel_tool_calls']=False
                    if route=='/v1/messages' and request.get('tools'):
                        choice=request.get('tool_choice',{'type':'auto'})
                        if not isinstance(choice,dict):raise ValueError('unsupported tool choice')
                        request['tool_choice']={**choice,'disable_parallel_tool_use':True}
                    if completion_budget is not None and route=='/v1/responses':request['max_output_tokens']=completion_budget
                    if completion_budget is not None and route=='/v1/messages':request['max_tokens']=completion_budget
                    status,content_type,result=native_relay_request(relay_path,relay_token,route,request)
                    if status==200 and gate is not None and route!='/v1/messages/count_tokens':gate.register(route,content_type,result,request)
                self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(result)));self.end_headers();self.wfile.write(result)
            except Exception as error:
                errors.append({'type':type(error).__name__,'detail':str(error)[:400]})
                self.send_error(400,'request rejected')
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True;server.errors=errors;return server
