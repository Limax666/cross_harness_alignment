import pytest
from flashrsi_tool_attestation import ToolCallGate
from flashrsi_final_turn_proxy import FinalTurnChatProxy

TOOLS=[{'type':'function','function':{'name':'read','parameters':{'type':'object','properties':{}}}}]


def test_final_turn_without_tools_is_allowed():
    gate=ToolCallGate({'read'})
    proxy=FinalTurnChatProxy(gate,'model',TOOLS,lambda request:{'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'done'}}]})
    assert proxy.forward({'model':'model','messages':[],'tool_choice':'none'})['choices'][0]['message']['content']=='done'
    gate.finish()


def test_final_turn_cannot_return_a_tool_call():
    gate=ToolCallGate({'read'})
    call={'id':'x','type':'function','function':{'name':'read','arguments':'{}'}}
    proxy=FinalTurnChatProxy(gate,'model',TOOLS,lambda request:{'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','tool_calls':[call]}}]})
    with pytest.raises(ValueError): proxy.forward({'model':'model','messages':[],'tool_choice':'none'})
    with pytest.raises(ValueError): gate.claim('read',{})


def test_tools_may_not_disappear_without_explicit_none():
    gate=ToolCallGate({'read'})
    proxy=FinalTurnChatProxy(gate,'model',TOOLS,lambda request:{})
    with pytest.raises(ValueError): proxy.forward({'model':'model','messages':[]})
