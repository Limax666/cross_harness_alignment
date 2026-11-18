"""Match native protocol MCP tool calls with the trusted AgentDojo dispatcher."""
import json
import threading


def _events(body):
    for block in body.decode('utf-8').replace('\r\n','\n').split('\n\n'):
        data='\n'.join(line[5:].lstrip() for line in block.splitlines() if line.startswith('data:'))
        if data and data!='[DONE]':
            yield json.loads(data)


def _mcp_name(name,namespace):
    if namespace=='mcp__benchmark' and isinstance(name,str):return name
    prefix='mcp__benchmark__'
    if isinstance(name,str) and name.startswith(prefix):return name[len(prefix):]
    return None


def extract_calls(route,content_type,body):
    """Extract completed benchmark calls, failing if a stream is incomplete."""
    calls=[]
    if content_type=='text/event-stream':
        events=list(_events(body))
        if route=='/v1/responses':
            if not any(e.get('type')=='response.completed' for e in events):raise ValueError('incomplete Responses stream')
            items=[e.get('item',{}) for e in events if e.get('type')=='response.output_item.done']
        elif route=='/v1/messages':
            if not any(e.get('type')=='message_stop' for e in events):raise ValueError('incomplete Messages stream')
            blocks={}
            for event in events:
                index=event.get('index');kind=event.get('type')
                if kind=='content_block_start' and event.get('content_block',{}).get('type')=='tool_use':
                    blocks[index]=dict(event['content_block']);blocks[index]['_parts']=[]
                elif kind=='content_block_delta' and index in blocks and event.get('delta',{}).get('type')=='input_json_delta':
                    blocks[index]['_parts'].append(event['delta'].get('partial_json',''))
            items=[]
            for block in blocks.values():
                parts=''.join(block.pop('_parts'))
                if parts:block['input']=json.loads(parts)
                items.append(block)
        else:return []
    else:
        value=json.loads(body)
        items=value.get('output',[]) if route=='/v1/responses' else value.get('content',[]) if route=='/v1/messages' else []
    for item in items:
        if not isinstance(item,dict) or item.get('type') not in ('function_call','tool_use'):continue
        name=_mcp_name(item.get('name'),item.get('namespace'))
        if name is None:continue
        call_id=item.get('call_id') or item.get('id')
        args=item.get('arguments',item.get('input',{}))
        if isinstance(args,str):args=json.loads(args)
        if not isinstance(call_id,str) or not call_id or not isinstance(args,dict):raise ValueError('invalid MCP call')
        calls.append({'id':call_id,'name':name,'arguments':args})
    return calls


class NativeToolGate:
    def __init__(self,allowed):
        self.allowed=frozenset(allowed);self.pending=[];self.observed=[];self.executed=[];self.failed=False;self.responses=[];self.requests=[]
        self.lock=threading.Lock()

    def register(self,route,content_type,body,request=None):
        with self.lock:
            try:
                if self.failed or self.pending:raise ValueError('pending MCP call')
                calls=extract_calls(route,content_type,body)
                seen={x['id'] for x in self.observed}
                if any(x['name'] not in self.allowed or x['id'] in seen for x in calls) or len({x['id'] for x in calls})!=len(calls):
                    raise ValueError('unauthorized or duplicate MCP call')
                self.pending.extend(calls);self.observed.extend(calls)
                self.responses.append({'route':route,'content_type':content_type,'body':body.decode('utf-8')})
                if request is not None:self.requests.append(json.loads(json.dumps(request,allow_nan=False)))
            except Exception:
                self.failed=True;raise

    def dispatch(self,name,arguments,execute):
        with self.lock:
            try:
                if self.failed or not self.pending:raise ValueError('unexpected MCP dispatch')
                expected=self.pending[0]
                if expected['name']!=name or json.dumps(expected['arguments'],sort_keys=True,allow_nan=False)!=json.dumps(arguments,sort_keys=True,allow_nan=False):
                    raise ValueError('MCP dispatch mismatch')
                result=execute(name,arguments)
                self.pending.pop(0);self.executed.append(expected)
                return result
            except Exception:
                self.failed=True;raise

    def attested(self):
        with self.lock:return bool(self.responses) and not self.failed and not self.pending and self.observed==self.executed
