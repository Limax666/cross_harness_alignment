from types import SimpleNamespace
from flashrsi_evo_dispatch import EvoRecorder
from flashrsi_evo_skill import READ_SCHEMA
from flashrsi_tool_attestation import ToolCallGate, GuardedChatProxy, GuardedToolDispatcher, attest_tool_calls


def test_skill_reader_uses_only_snapshot_and_rejects_traversal():
    snapshot=SimpleNamespace(files={'skills/allowed/SKILL.md':SimpleNamespace(content=b'---\ndescription: x\n---\nvisible lesson')})
    recorder=EvoRecorder(None,snapshot)
    assert recorder.tool('read_skill',{'skill':'allowed'})['observation']=='visible lesson'
    assert recorder.tool('read_skill',{'skill':'../../secret'})['invalid_call'] is True
    assert recorder.tool('read_skill',{'skill':'allowed','extra':'denied'})['invalid_call'] is True
    assert recorder.tool('read_skill',{'skill':'missing'})['observation']=='Skill not found.'


def test_skill_read_is_bound_to_model_call_and_attested():
    snapshot=SimpleNamespace(files={'skills/allowed/SKILL.md':SimpleNamespace(content=b'---\ndescription: x\n---\nvisible lesson')})
    tools=[{'type':'function','function':READ_SCHEMA}]
    gate=ToolCallGate({'read_skill'})
    message={'role':'assistant','content':None,'tool_calls':[{'id':'one','type':'function','function':{'name':'read_skill','arguments':'{"skill":"allowed"}'}}]}
    responses=[{'choices':[{'message':message,'finish_reason':'tool_calls'}]},
               {'choices':[{'message':{'role':'assistant','content':'done'},'finish_reason':'stop'}]}]
    proxy=GuardedChatProxy(gate,'qwen4b',tools,lambda _:responses.pop(0))
    dispatcher=GuardedToolDispatcher(gate,EvoRecorder(None,snapshot))
    proxy.forward({'model':'qwen4b','messages':[],'tools':tools})
    output=dispatcher.dispatch('read_skill',{'skill':'allowed'})
    assert output['observation']=='visible lesson'
    proxy.forward({'model':'qwen4b','messages':[message,{'role':'tool','tool_call_id':'one','content':output['observation']}],'tools':tools})
    gate.finish()
    assert attest_tool_calls(proxy.raw_messages,dispatcher.executed)
