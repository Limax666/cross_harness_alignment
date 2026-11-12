"""Bind the FlashRSI Hermes Python port to fixed AgentDojo tool schemas."""
import json,re
_NAME=re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,63}\Z')
_BRIDGE='''import json,os,urllib.request
from pathlib import Path

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("redirect refused")

class HermesTools:
    def __init__(self, cwd, snapshot_root, enabled_tools=None):
        self.cwd = Path(cwd).resolve()
        self.schemas = json.loads((Path(__file__).parent / "benchmark_schemas.json").read_text())
        self._allowed = {item["function"]["name"] for item in self.schemas}
        self.todos = None
    def execute(self, name, args):
        if name not in self._allowed or not isinstance(args, dict):
            raise ValueError("unregistered tool")
        body = json.dumps({"name":name,"arguments":args},allow_nan=False).encode()
        req = urllib.request.Request(os.environ["FLASHRSI_GATEWAY_URL"] + "/tool",data=body,headers={"Authorization":"Bearer "+os.environ["FLASHRSI_GATEWAY_TOKEN"],"Content-Type":"application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),_NoRedirect())
        with opener.open(req,timeout=30) as response:
            raw=response.read(2*1024*1024+1)
        if len(raw)>2*1024*1024: raise ValueError("observation too large")
        value=json.loads(raw)
        if set(value)!={"observation"} or not isinstance(value["observation"],str): raise ValueError("invalid observation")
        return value["observation"]
    def close(self): pass
'''


def make_hermes_snapshot(agent, tools, safety_policy):
    if not isinstance(safety_policy,str) or not safety_policy.strip() or not isinstance(tools,list) or not tools:
        raise ValueError('policy and tools are required')
    names=[]
    for item in tools:
        if not isinstance(item,dict) or item.get('type')!='function' or not isinstance(item.get('function'),dict): raise ValueError('invalid tool')
        fn=item['function']; name=fn.get('name')
        if not isinstance(name,str) or not _NAME.fullmatch(name) or not isinstance(fn.get('parameters'),dict): raise ValueError('invalid tool schema')
        names.append(name)
    if len(set(names))!=len(names): raise ValueError('duplicate tool')
    root=agent.snapshot
    with root.check_out() as directory:
        runtime=directory/'runtime/hermes_native'; loop=runtime/'loop.py'; source=loop.read_text(); old='from .tools import HermesTools'
        if source.count(old)!=1: raise ValueError('unexpected Hermes tool registry')
        loop.write_text(source.replace(old,'from .benchmark_bridge import HermesTools'))
        (runtime/'benchmark_bridge.py').write_text(_BRIDGE)
        (runtime/'benchmark_schemas.json').write_text(json.dumps(tools,ensure_ascii=False,allow_nan=False))
        resources=runtime/'resources.py'; content=resources.read_text(); anchor='TOOLSETS = {'
        if content.count(anchor)!=1: raise ValueError('unexpected Hermes resource registry')
        content=content.replace(anchor,'DEFAULT_TOOLS.extend(item["function"]["name"] for item in json.loads((Path(__file__).parent / "benchmark_schemas.json").read_text()) if item["function"]["name"] not in DEFAULT_TOOLS)\n'+anchor)
        resources.write_text(content)
        prompt=directory/'AGENTS.md'; original=prompt.read_text() if prompt.exists() else ''
        prompt.write_text(original.rstrip()+'\n\n## Trusted safety policy\n'+safety_policy.strip()+'\n')
        return root.commit(directory)
