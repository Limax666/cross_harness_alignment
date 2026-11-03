"""Deadline must abort the actual backend ID and release the load balancer slot."""
import asyncio
from types import SimpleNamespace
from omegaconf import OmegaConf
from verl.workers.rollout.llm_server import LLMServerClient

def test_timeout_aborts_actual_backend_id(monkeypatch):
    monkeypatch.setenv('VERL_GENERATION_TIMEOUT_SECONDS','0.01')
    events=[]
    async def generate(**kwargs):
        events.append(('generate',kwargs['request_id']))
        await asyncio.sleep(10)
    async def abort(request_id,reset_prefix_cache):
        events.append(('abort',request_id,reset_prefix_cache));return {'aborted':True}
    server=SimpleNamespace(generate=SimpleNamespace(remote=generate),abort_request=SimpleNamespace(remote=abort))
    config=OmegaConf.create({'actor_rollout_ref':{'rollout':{'name':'vllm','full_determinism':False}}})
    client=LLMServerClient(config=config,load_balancer_handle=None)
    async def acquire(*args,**kwargs):return 'server-1',server
    client._acquire_server=acquire
    client._release_server=lambda sid,request_id=None:events.append(('release',sid,request_id))
    client._vllm_request_id=lambda _: 'actual-backend-id'
    async def go():
        try: await client.generate('sticky-id',prompt_ids=[1],sampling_params={})
        except TimeoutError:pass
        else:raise AssertionError('expected timeout')
    asyncio.run(go())
    assert events==[('generate','actual-backend-id'),('abort','actual-backend-id',False),('release','server-1','sticky-id')]


def test_deadline_rejects_unsupported_backend_before_dispatch(monkeypatch):
    monkeypatch.setenv('VERL_GENERATION_TIMEOUT_SECONDS','1')
    events=[]
    server=SimpleNamespace(generate=SimpleNamespace(remote=lambda **kwargs:events.append('dispatch')))
    config=OmegaConf.create({'actor_rollout_ref':{'rollout':{'name':'sglang','full_determinism':False}}})
    client=LLMServerClient(config=config,load_balancer_handle=None)
    async def acquire(*args,**kwargs):return 'server',server
    client._acquire_server=acquire
    client._release_server=lambda sid,request_id=None:events.append('released')
    async def go():
        try:await client.generate('req',prompt_ids=[1],sampling_params={})
        except ValueError:pass
        else:raise AssertionError('unsupported backend not rejected')
    asyncio.run(go());assert events==['released']
