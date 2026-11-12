"""Authenticated gateway adapting a captured non-stream completion to SSE."""
import copy,hmac,json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from flashrsi_tool_attestation import _invalid_constant,_unique_object


def completion_to_sse(response):
    if not isinstance(response,dict) or len(response.get('choices',[]))!=1: raise ValueError('expected one choice')
    choice=response['choices'][0]; message=choice['message']; delta={'role':'assistant'}
    if message.get('content') is not None: delta['content']=message['content']
    calls=message.get('tool_calls') or []
    if calls: delta['tool_calls']=[{'index':i,**copy.deepcopy(call)} for i,call in enumerate(calls)]
    common={'id':response.get('id','stream'),'object':'chat.completion.chunk','model':response.get('model','')}
    events=[{**common,'choices':[{'index':0,'delta':delta,'finish_reason':None}]},
            {**common,'choices':[{'index':0,'delta':{},'finish_reason':choice.get('finish_reason')}]}]
    if response.get('usage') is not None: events.append({**common,'choices':[],'usage':response['usage']})
    return (''.join('data: '+json.dumps(e,ensure_ascii=False)+'\n\n' for e in events)+'data: [DONE]\n\n').encode()


def create_stream_gateway(proxy,dispatcher,gate,token):
    if not isinstance(token,str) or len(token)<32 or not token.isascii() or any(c.isspace() for c in token): raise ValueError('invalid token')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,format,*args): pass
        def reply(self,status,value,content_type='application/json'):
            body=json.dumps(value,ensure_ascii=True,allow_nan=False).encode() if content_type=='application/json' else value
            self.send_response(status); self.send_header('Content-Type',content_type); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            headers=self.headers.get_all('Authorization',[])
            if len(headers)!=1 or not hmac.compare_digest(headers[0],'Bearer '+token): return self.reply(401,{'error':'unauthorized'})
            try:
                lengths=self.headers.get_all('Content-Length',[])
                if len(lengths)!=1 or self.headers.get('Transfer-Encoding') is not None or not lengths[0].isdigit() or not 0<int(lengths[0])<=2*1024*1024: raise ValueError('invalid length')
                if self.headers.get_content_type()!='application/json': raise ValueError('invalid content type')
                raw=self.rfile.read(int(lengths[0])); data=json.loads(raw,object_pairs_hook=_unique_object,parse_constant=_invalid_constant)
                if not isinstance(data,dict): raise ValueError('invalid payload')
                if self.path=='/v1/chat/completions':
                    stream=data.pop('stream',False); data.pop('stream_options',None)
                    if type(stream) is not bool: raise ValueError('invalid stream')
                    result=proxy.forward(data)
                    if stream: return self.reply(200,completion_to_sse(result),'text/event-stream')
                elif self.path=='/tool':
                    if set(data)!={'name','arguments'} or not isinstance(data['name'],str) or not isinstance(data['arguments'],dict): raise ValueError('invalid tool')
                    native=dispatcher.dispatch(data['name'],data['arguments'])
                    if not isinstance(native,dict) or not isinstance(native.get('observation'),str): raise ValueError('invalid observation')
                    result={'observation':native['observation']}
                else: raise ValueError('unknown route')
                self.reply(200,result)
            except Exception:
                gate.abort(); self.reply(400,{'error':'request rejected; episode invalid'})
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler); server.daemon_threads=True
    return server
