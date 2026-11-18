"""Run a pinned Codex or Claude Code Snapshot inside the native sandbox."""
import asyncio
import json
import os
import stat
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'vendor/FlashRSI'),str(ROOT/'scripts')]
from flashrsi.core import Snapshot
from flashrsi.agents.codex import Codex
from flashrsi.agents.claude_code import ClaudeCode

request=json.load(sys.stdin)
if set(request)!={'harness','snapshot_json','prompt','workdir','claude_cli'}:
    raise ValueError('invalid native entry request')
token=os.environ['FLASHRSI_GATEWAY_TOKEN'];base=os.environ['FLASHRSI_GATEWAY_URL']
budget=int(os.environ['FLASHRSI_COMPLETION_BUDGET'])
if request['harness']=='codex':
    agent=Codex({'llm':{'model_name':'qwen35-4b-preflight','api_key':token,'base_url':base+'/v1',
                        'provider_name':'FlashRSI local'},'sandbox':'read-only','timeout':300})
elif request['harness']=='claude_code':
    agent=ClaudeCode({'llm':{'model_name':'qwen35-4b-preflight','api_key':token,'base_url':base,
                             'max_tokens':budget,'max_retries':0},'cli_path':request['claude_cli'],
                      'permission_mode':'dontAsk','allowed_tools':['mcp__benchmark__*'],
                      'tools':[],'max_turns':12,'timeout':300})
else:raise ValueError('unknown native harness')
snapshot=Snapshot.load(request['snapshot_json']);agent.update(snapshot)
os.umask(0o022)
# FlashRSI reapplies the mode to each newly exported file. The native sandbox
# blocks chmod, so permit only the no-op case, inside this run's checkout.
_chmod=Path.chmod
_open=Path.open
def _checkout_open(path, mode='r', *args, **kwargs):
    if mode=='xb':
        resolved=path.resolve()
        root=Path(request['workdir']).resolve().parent
        checkout=next((p for p in resolved.parents if p.name.startswith('checkout-') and root in p.parents),None)
        if checkout is not None:
            relative=resolved.relative_to(checkout).as_posix()
            entry=snapshot.files.get(relative)
            if entry is None:raise ValueError('unexpected exported snapshot path')
            fd=os.open(resolved,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o755 if entry.executable else 0o644)
            return os.fdopen(fd,'wb')
    return _open(path,mode,*args,**kwargs)
Path.open=_checkout_open
def _checkout_chmod(path, mode, *args, **kwargs):
    resolved=path.resolve()
    root=Path(request['workdir']).resolve().parent
    if root not in resolved.parents:
        return _chmod(path, mode, *args, **kwargs)
    if stat.S_IMODE(path.stat().st_mode)!=mode:
        raise PermissionError(f'sandbox cannot change exported snapshot mode: {path.name} {stat.S_IMODE(path.stat().st_mode):o} -> {mode:o}')
Path.chmod=_checkout_chmod
result=asyncio.run(agent.run(request['prompt'],working_dir=Path(request['workdir'])))
print(json.dumps({'status':result.status,'output':result.output,'error_message':result.error_message,
                  'messages':[dict(x) for x in result.messages],'usage':dict(result.usage)},ensure_ascii=False,default=str))
