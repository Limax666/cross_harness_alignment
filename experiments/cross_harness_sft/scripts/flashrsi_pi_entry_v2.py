"""Run the Pi snapshot inside the restricted one-episode process."""
import asyncio,json,os,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'vendor/FlashRSI'),str(ROOT/'scripts')]
from flashrsi.agents.pi_agent import Pi
from flashrsi.core.agent import _Execution
from flashrsi_completion_budget import completion_budget

request=json.load(sys.stdin)
token=os.environ['FLASHRSI_GATEWAY_TOKEN']; base=os.environ['FLASHRSI_GATEWAY_URL']
budget=completion_budget({'FLASHRSI_COMPLETION_BUDGET':os.environ.get('FLASHRSI_COMPLETION_BUDGET','512')})
config={'llm':{'model_name':'qwen35-4b-preflight','provider_name':'local','model_api':'openai-completions','api_key':token,'base_url':base+'/v1','max_tokens':budget,'temperature':0,'stream':False},'max_turns':12,'timeout':300,'compaction':{'enabled':False},'thinking':'off'}
agent=Pi(config)
execution=agent._execution.set(_Execution(None, Path(request['snapshot_dir']), agent._config))
try:
    result=asyncio.run(agent._run(request['prompt']))
finally:
    agent._execution.reset(execution)
print(json.dumps({'result':{'status':result.status,'output':result.output,'error_message':result.error_message,'messages':[dict(x) for x in result.messages],'usage':dict(result.usage)}},ensure_ascii=False,default=str))
