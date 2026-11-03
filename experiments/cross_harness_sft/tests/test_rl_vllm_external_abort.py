import asyncio
from types import SimpleNamespace
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

def test_abort_resolves_external_request_id():
 events=[]
 class Engine:
  def __init__(self):self.output_processor=SimpleNamespace(request_states={'req-random':object()},external_req_ids={'req':['req-random']})
  async def abort(self,request_id):
   events.append(request_id);self.output_processor.request_states.pop('req-random');self.output_processor.external_req_ids.pop(request_id)
 e=Engine();s=object.__new__(vLLMHttpServer);s.node_rank=0;s.engine=e
 result=asyncio.run(s.abort_request('req',reset_prefix_cache=False))
 assert result['aborted'] is True and events==['req'] and not e.output_processor.request_states


def test_abort_cancels_request_before_engine_registration():
 async def exercise():
  processor=SimpleNamespace(request_states={},external_req_ids={})
  server=object.__new__(vLLMHttpServer);server.node_rank=0;server.engine=SimpleNamespace(output_processor=processor)
  entered=asyncio.Event()
  async def waiting():
   entered.set();await asyncio.sleep(60)
  task=asyncio.create_task(waiting());server._generation_tasks={'pending':task};await entered.wait()
  result=await server.abort_request('pending',reset_prefix_cache=False)
  assert result['aborted'] is True and task.done()
 asyncio.run(exercise())
