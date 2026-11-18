import json
import pytest
from flashrsi_native_tool_gate import NativeToolGate,extract_calls


def events(*rows):
    return ''.join('data: '+json.dumps(x)+'\n\n' for x in rows).encode()


def test_responses_stream_matches_exact_mcp_dispatch():
    body=events({'type':'response.output_item.done','item':{'type':'function_call','name':'get_current_day','namespace':'mcp__benchmark','call_id':'c1','arguments':'{}'}}, {'type':'response.completed'})
    gate=NativeToolGate({'get_current_day'});gate.register('/v1/responses','text/event-stream',body)
    assert gate.dispatch('get_current_day',{},lambda name,args:'today')=='today'
    assert gate.attested()


def test_messages_stream_reassembles_tool_json_and_denies_mismatch():
    body=events({'type':'content_block_start','index':1,'content_block':{'type':'tool_use','id':'c2','name':'mcp__benchmark__lookup','input':{}}},
                {'type':'content_block_delta','index':1,'delta':{'type':'input_json_delta','partial_json':'{"x":1}'}},
                {'type':'content_block_stop','index':1},{'type':'message_stop'})
    assert extract_calls('/v1/messages','text/event-stream',body)==[{'id':'c2','name':'lookup','arguments':{'x':1}}]
    gate=NativeToolGate({'lookup'});gate.register('/v1/messages','text/event-stream',body)
    with pytest.raises(ValueError):gate.dispatch('lookup',{'x':True},lambda name,args:None)
    assert not gate.attested()


def test_incomplete_stream_fails_closed():
    with pytest.raises(ValueError,match='incomplete'):
        extract_calls('/v1/responses','text/event-stream',events({'type':'response.created'}))
