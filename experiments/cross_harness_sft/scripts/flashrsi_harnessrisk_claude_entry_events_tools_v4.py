"""Execute Claude Code with the supplied FlashRSI snapshot in the sandbox (events-tools v4).

events-tools-v4 (per diagnosis_claude_native_events_tools_v4_20261114.json):
serialize SDK event objects as real JSON structures instead of `default=str`
strings, so downstream dict()/Mapping consumers receive mappings, not
stringified pseudo-dicts. Tool advertisement restriction lives in the
supervisor's config (`tools` key), not here. Codex must use the original
entry; this variant is registered for claude_code only.
"""
import asyncio,hashlib,json,sys,os,shutil,stat
from collections.abc import Mapping
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'vendor/FlashRSI')]
from flashrsi.agents.claude_code import ClaudeCode
from flashrsi.core import Snapshot

def _plain(value):
    """Recursively convert SDK immutable views to JSON-native structures."""
    if isinstance(value,Mapping):
        return {k if isinstance(k,(str,int,float,bool)) or k is None else str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_plain(v) for v in value]
    if isinstance(value,(str,int,float,bool)) or value is None:
        return value
    return str(value)

async def main():
    request=json.load(sys.stdin)
    if request.get('harness')!='claude_code':
        raise ValueError('events-tools-v4 entry is registered for claude_code only')
    snapshot=Snapshot.load(request['snapshot_json']);config=request['config']
    agent=ClaudeCode(config)
    result=await agent._execute(request['question'],Path(request['working_dir']),snapshot,Path(request['workspace_dir']))
    payload={'answer':result.output or '',
             'messages':_plain(list(result.messages)),
             'stop_reason':'completed' if result.status=='finished' else 'runtime_error: '+str(result.error_message),
             'usage':_plain(dict(result.usage))}
    print(json.dumps(payload,ensure_ascii=False))
if __name__=='__main__':asyncio.run(main())
