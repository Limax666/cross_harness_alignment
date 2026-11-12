import json
from flashrsi_stream_gateway import completion_to_sse


def test_stream_envelope_preserves_content_and_tool_call():
    response={'id':'abc','model':'model','choices':[{'index':0,'message':{'role':'assistant','content':'ok','tool_calls':[{'id':'c1','type':'function','function':{'name':'read','arguments':'{"x":1}'}}]},'finish_reason':'tool_calls'}]}
    events=completion_to_sse(response).decode().split('\n\n')
    decoded=[json.loads(e[6:]) for e in events if e.startswith('data: {')]
    delta=decoded[0]['choices'][0]['delta']
    assert delta['content']=='ok'
    assert delta['tool_calls'][0]['function']['arguments']=='{"x":1}'
    assert decoded[-1]['choices'][0]['finish_reason']=='tool_calls'
    assert 'data: [DONE]' in events
