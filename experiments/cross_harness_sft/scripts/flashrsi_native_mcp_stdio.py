"""Minimal per-episode MCP stdio adapter to a loopback-only trusted gateway."""
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

MAX_LINE=2*1024*1024


def main():
    address=os.environ['FLASHRSI_GATEWAY_URL'];token=os.environ['FLASHRSI_GATEWAY_TOKEN']
    parsed=urllib.parse.urlsplit(address)
    if parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or parsed.port is None or parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError('invalid gateway address')
    if len(token)<32 or not token.isascii() or any(c.isspace() for c in token):raise ValueError('invalid gateway token')
    schemas=json.loads(Path(os.environ['FLASHRSI_NATIVE_SCHEMAS']).read_text())
    tools=[];names=set()
    for item in schemas:
        fn=item['function'];name=fn['name']
        if not isinstance(name,str) or not name or name in names:raise ValueError('invalid tool list')
        names.add(name);tools.append({'name':name,'description':fn.get('description',''),'inputSchema':fn['parameters']})
    while True:
        raw=sys.stdin.buffer.readline(MAX_LINE+1)
        if not raw:return
        if len(raw)>MAX_LINE or not raw.endswith(b'\n'):raise ValueError('invalid MCP frame')
        request=json.loads(raw);method=request.get('method');identifier=request.get('id')
        if identifier is None:continue
        try:
            if method=='initialize':
                result={'protocolVersion':request.get('params',{}).get('protocolVersion','2024-11-05'),
                        'capabilities':{'tools':{'listChanged':False}},'serverInfo':{'name':'agentdojo-benchmark','version':'1.0'}}
            elif method=='ping':result={}
            elif method=='tools/list':result={'tools':tools}
            elif method=='tools/call':
                params=request.get('params',{});name=params.get('name');arguments=params.get('arguments',{})
                if name not in names or not isinstance(arguments,dict):raise ValueError('invalid tool request')
                body=json.dumps({'name':name,'arguments':arguments},allow_nan=False).encode()
                req=urllib.request.Request(address+'/tool',data=body,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=60) as response:
                    value=json.load(response)
                if set(value)!={'observation'} or not isinstance(value['observation'],str):raise ValueError('invalid gateway result')
                result={'content':[{'type':'text','text':value['observation']}],'isError':False}
            else:raise ValueError('unsupported MCP method')
            reply={'jsonrpc':'2.0','id':identifier,'result':result}
        except Exception:
            reply={'jsonrpc':'2.0','id':identifier,'error':{'code':-32603,'message':'MCP request rejected'}}
        sys.stdout.write(json.dumps(reply,ensure_ascii=True,allow_nan=False)+'\n');sys.stdout.flush()


if __name__=='__main__':main()
