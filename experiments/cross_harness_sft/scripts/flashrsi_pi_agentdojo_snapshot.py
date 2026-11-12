"""Bind FlashRSI Pi's Python runtime to the fixed AgentDojo tools."""
import json,re
_NAME=re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,63}\Z')
_BRIDGE='''import json,os,urllib.request
from pathlib import Path
from .tools import ToolResult

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl): raise ValueError("redirect refused")
def tool_schemas(enabled=None):
    return json.loads((Path(__file__).parent/"benchmark_schemas.json").read_text())
def prepare_tool_arguments(name,arguments,arguments_error=None):
    if arguments_error or not isinstance(arguments,dict) or name not in {x["function"]["name"] for x in tool_schemas()}:
        raise ValueError("invalid tool call")
    return arguments
def execute_tool(name,arguments,cwd,**kwargs):
    prepare_tool_arguments(name,arguments)
    body=json.dumps({"name":name,"arguments":arguments},allow_nan=False).encode()
    req=urllib.request.Request(os.environ["FLASHRSI_GATEWAY_URL"]+"/tool",data=body,headers={"Authorization":"Bearer "+os.environ["FLASHRSI_GATEWAY_TOKEN"],"Content-Type":"application/json"})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),_NoRedirect())
    with opener.open(req,timeout=30) as response: raw=response.read(2*1024*1024+1)
    if len(raw)>2*1024*1024: raise ValueError("observation too large")
    value=json.loads(raw)
    if set(value)!={"observation"} or not isinstance(value["observation"],str): raise ValueError("invalid observation")
    return ToolResult(value["observation"])
'''


def make_pi_snapshot(agent,tools,safety_policy):
    if not isinstance(safety_policy,str) or not safety_policy.strip() or not isinstance(tools,list) or not tools: raise ValueError('policy and tools required')
    names=[]
    for item in tools:
        if not isinstance(item,dict) or item.get('type')!='function' or not isinstance(item.get('function'),dict): raise ValueError('invalid tool')
        fn=item['function']; name=fn.get('name')
        if not isinstance(name,str) or not _NAME.fullmatch(name) or not isinstance(fn.get('parameters'),dict): raise ValueError('invalid schema')
        names.append(name)
    if len(names)!=len(set(names)): raise ValueError('duplicate tool')
    root=agent.snapshot
    with root.check_out() as directory:
        runtime=directory/'runtime/pi_native'; loop=runtime/'loop.py'; source=loop.read_text(); old='from .tools import ToolResult, _resolve, execute_tool, prepare_tool_arguments'
        if source.count(old)!=1: raise ValueError('unexpected Pi registry')
        loop.write_text(source.replace(old,'from .tools import ToolResult, _resolve\nfrom .benchmark_bridge import execute_tool, prepare_tool_arguments'))
        resources=runtime/'resources.py'; source=resources.read_text(); old='from .tools import tool_schemas'
        if source.count(old)!=1: raise ValueError('unexpected Pi resources')
        resources.write_text(source.replace(old,'from .benchmark_bridge import tool_schemas'))
        (runtime/'benchmark_bridge.py').write_text(_BRIDGE)
        (runtime/'benchmark_schemas.json').write_text(json.dumps(tools,ensure_ascii=False,allow_nan=False))
        prompt=directory/'AGENTS.md'; original=prompt.read_text() if prompt.exists() else ''
        prompt.write_text(original.rstrip()+'\n\n## Trusted safety policy\n'+safety_policy.strip()+'\n')
        return root.commit(directory)
