import json
import threading
import urllib.error
import urllib.request
import pytest
from flashrsi_native_protocol_relay import create_native_gateway,create_native_relay
from flashrsi_native_tool_gate import NativeToolGate


def test_native_gateway_forwards_only_pinned_protocol_and_model(tmp_path):
    relay_token='r'*40;child_token='c'*40;seen=[]
    relay=create_native_relay(str(tmp_path/'model.sock'),relay_token,'qwen4b',
        lambda path,request:(seen.append((path,request)) or (200,'text/event-stream',b'data: example\n\n')))
    relay_thread=threading.Thread(target=relay.serve_forever,daemon=True);relay_thread.start()
    gateway=create_native_gateway(str(tmp_path/'model.sock'),relay_token,child_token,'qwen4b')
    gateway_thread=threading.Thread(target=gateway.serve_forever,daemon=True);gateway_thread.start()
    try:
        url=f'http://127.0.0.1:{gateway.server_port}'
        def post(path,model='qwen4b',token=child_token):
            data=json.dumps({'model':model,'stream':True}).encode()
            request=urllib.request.Request(url+path,data=data,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            return urllib.request.urlopen(request,timeout=5)
        with post('/v1/responses') as response:assert response.read()==b'data: example\n\n'
        with post('/v1/messages') as response:assert response.status==200
        for route,model,token in [('/v1/other','qwen4b',child_token),('/v1/messages','other',child_token),('/v1/messages','qwen4b','bad')]:
            with pytest.raises(urllib.error.HTTPError):post(route,model,token)
        assert [x[0] for x in seen]==['/v1/responses','/v1/messages']
    finally:
        gateway.shutdown();gateway_thread.join();gateway.server_close()
        relay.shutdown();relay_thread.join();relay.server_close()


def test_gateway_attests_mcp_dispatch_against_model_response(tmp_path):
    rt='r'*40;ct='c'*40
    message={'output':[{'type':'function_call','call_id':'c1','name':'get_current_day','namespace':'mcp__benchmark','arguments':'{}'}]}
    relay=create_native_relay(str(tmp_path/'model.sock'),rt,'qwen4b',lambda path,request:(200,'application/json',json.dumps(message).encode()))
    thread=threading.Thread(target=relay.serve_forever,daemon=True);thread.start()
    gate=NativeToolGate({'get_current_day'});gateway=create_native_gateway(str(tmp_path/'model.sock'),rt,ct,'qwen4b',gate=gate,dispatch=lambda n,a:{'observation':'today'})
    gt=threading.Thread(target=gateway.serve_forever,daemon=True);gt.start()
    try:
        url=f'http://127.0.0.1:{gateway.server_port}'
        def post(path,body):
            request=urllib.request.Request(url+path,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+ct,'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=5) as response:return json.load(response)
        post('/v1/responses',{'model':'qwen4b'})
        assert post('/tool',{'name':'get_current_day','arguments':{}})=={'observation':'today'}
        assert gate.attested()
    finally:
        gateway.shutdown();gt.join();gateway.server_close();relay.shutdown();thread.join();relay.server_close()
