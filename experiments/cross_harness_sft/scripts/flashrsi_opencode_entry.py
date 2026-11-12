"""Run the OpenCode snapshot inside the restricted one-episode process."""
import asyncio,json,os,sys
from pathlib import Path

ROOT=Path('/output/cross_harness_alignment/experiments/cross_harness_sft')
sys.path[:0]=[str(ROOT/'vendor/FlashRSI'),str(ROOT/'scripts')]
from flashrsi.agents.opencode import OpenCode
from flashrsi_opencode_agentdojo_snapshot import make_opencode_snapshot

request=json.load(sys.stdin)
token=os.environ['FLASHRSI_GATEWAY_TOKEN']; base=os.environ['FLASHRSI_GATEWAY_URL']
config={'llm':{'model_name':'qwen35-4b-preflight','api_key':token,'base_url':base+'/v1','max_tokens':512,'temperature':0,'stream':False},'max_turns':12,'timeout':300,'session':{'persist':False},'compaction':{'enabled':False}}
agent=OpenCode(config); root_id=agent.snapshot.id
snapshot=make_opencode_snapshot(agent,request['tools'],request['policy']); agent.update(snapshot)
result=asyncio.run(agent.run(request['prompt']))
print(json.dumps({'root_snapshot':root_id,'adapted_snapshot':snapshot.id,'result':{'status':result.status,'output':result.output,'error_message':result.error_message,'messages':[dict(x) for x in result.messages],'usage':dict(result.usage)}},ensure_ascii=False,default=str))
