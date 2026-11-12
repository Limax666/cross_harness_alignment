import json
import threading
import urllib.error
import urllib.request

import pytest

from flashrsi_gateway import create_gateway
from flashrsi_tool_attestation import ToolCallGate, GuardedChatProxy, GuardedToolDispatcher


def post(url, token, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


@pytest.fixture
def gateway():
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    response = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [
        {"id": "one", "type": "function", "function": {"name": "read", "arguments": '{"id":1}'}}
    ]}}]}
    gate = ToolCallGate({"read"})
    proxy = GuardedChatProxy(gate, "qwen4b", tools, lambda _: response)

    class Recorder:
        calls = []

        def tool(self, name, arguments):
            self.calls.append((name, arguments))
            return {"observation": "fixture", "invalid_call": False}

    recorder = Recorder()
    dispatcher = GuardedToolDispatcher(gate, recorder)
    server = create_gateway(proxy, dispatcher, gate, "test-token-long-enough-32-characters")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", tools, gate, proxy, dispatcher, recorder
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_real_http_model_to_guarded_tool(gateway):
    url, tools, gate, proxy, dispatcher, recorder = gateway
    token = "test-token-long-enough-32-characters"
    post(url + "/v1/chat/completions", token, {"model": "qwen4b", "messages": [], "tools": tools})
    result = post(url + "/tool", token, {"name": "read", "arguments": {"id": 1}})
    assert result["observation"] == "fixture"
    assert recorder.calls == [("read", {"id": 1})]
    assert dispatcher.executed == [{"id": "one", "name": "read", "arguments": {"id": 1}}]
    assert proxy.raw_messages[0]["tool_calls"][0]["id"] == "one"
    gate.finish()


def test_http_rejects_unauthorized_requests_and_unmatched_dispatch(gateway):
    url, tools, gate, proxy, dispatcher, recorder = gateway
    with pytest.raises(urllib.error.HTTPError) as error:
        post(url + "/tool", "wrong", {"name": "read", "arguments": {}})
    assert error.value.code == 401
    post(url + "/v1/chat/completions", "test-token-long-enough-32-characters",
         {"model": "qwen4b", "messages": [], "tools": tools})
    with pytest.raises(urllib.error.HTTPError):
        post(url + "/tool", "test-token-long-enough-32-characters", {"name": "read", "arguments": {"id": 2}})
    assert recorder.calls == []
    with pytest.raises(ValueError):
        gate.finish()
