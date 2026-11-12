import pytest

from flashrsi_tool_attestation import (attest_tool_calls, ToolCallGate, GuardedToolDispatcher,
                                       GuardedChatProxy, attest_tool_observations)


def test_exact_ordered_calls_match_raw_api_response():
    messages = [{"role": "assistant", "tool_calls": [
        {"id": "call-1", "type": "function", "function": {"name": "get_day", "arguments": '{"day":"2024-05-19"}'}},
        {"id": "call-2", "type": "function", "function": {"name": "lookup", "arguments": '{}'}},
    ]}]
    executed = [{"id": "call-1", "name": "get_day", "arguments": {"day": "2024-05-19"}},
                {"id": "call-2", "name": "lookup", "arguments": {}}]
    assert attest_tool_calls(messages, executed) is True
    assert attest_tool_calls(messages, list(reversed(executed))) is False
    assert attest_tool_calls(messages, executed[:1]) is False


@pytest.mark.parametrize("arguments", ['{"x":1,"x":2}', '{"x":NaN}', '[]', 'not-json'])
def test_rejects_ambiguous_or_non_object_arguments(arguments):
    messages = [{"role": "assistant", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "tool", "arguments": arguments}}
    ]}]
    assert not attest_tool_calls(messages, [{"id": "a", "name": "tool", "arguments": {"x": 2}}])


def test_rejects_duplicate_ids_and_boolean_number_mismatch():
    call = {"id": "a", "type": "function", "function": {"name": "tool", "arguments": '{"x":true}'}}
    item = {"id": "a", "name": "tool", "arguments": {"x": 1}}
    assert not attest_tool_calls([{"role": "assistant", "tool_calls": [call]}], [item])
    assert not attest_tool_calls([{"role": "assistant", "tool_calls": [call, call]}], [item, item])


def test_rejects_calls_from_non_assistant_and_malformed_records():
    assert not attest_tool_calls([{"role": "tool", "tool_calls": [{}]}], [])
    assert not attest_tool_calls([{"role": "assistant", "tool_calls": [{}]}], [])
    assert not attest_tool_calls([], [{"id": "x", "name": "tool", "arguments": {}}])
    assert not attest_tool_calls([], [])
    assert attest_tool_calls([{"role": "assistant", "content": "done"}], [])


def test_rejects_non_json_dispatch_keys_even_when_json_coerces_them():
    messages = [{"role": "assistant", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "tool", "arguments": '{"nested":{"1":"x"}}'}}
    ]}]
    assert not attest_tool_calls(messages, [{"id": "a", "name": "tool", "arguments": {"nested": {1: "x"}}}])


def test_observation_attestation_independently_rejects_duplicate_ids():
    call = {"id": "x", "type": "function", "function": {"name": "read", "arguments": "{}"}}
    assistant = {"role": "assistant", "tool_calls": [call, call]}
    requests = [{"messages": []}, {"messages": [assistant, {"role": "tool", "tool_call_id": "x", "content": "ok"}]}]
    assert not attest_tool_observations([assistant, {"role": "assistant", "content": "done"}], requests, {"x": "ok"})


def test_gate_only_authorizes_raw_api_calls_once_in_order():
    gate = ToolCallGate({"get_day", "lookup"})
    gate.accept_response({"role": "assistant", "tool_calls": [
        {"id": "one", "type": "function", "function": {"name": "get_day", "arguments": '{"day":"today"}'}},
        {"id": "two", "type": "function", "function": {"name": "lookup", "arguments": '{}'}},
    ]})
    assert gate.claim("get_day", {"day": "today"}) == "one"
    gate.complete("one")
    assert gate.claim("lookup", {}) == "two"
    gate.complete("two")
    gate.finish()
    with pytest.raises(ValueError):
        gate.claim("lookup", {})


def test_gate_refuses_new_response_while_dispatches_are_pending():
    gate = ToolCallGate({"read"})
    gate.accept_response({"role": "assistant", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "read", "arguments": '{}'}}
    ]})
    with pytest.raises(ValueError):
        gate.accept_response({"role": "assistant", "content": "too early"})
    with pytest.raises(ValueError):
        gate.claim("read", {})


def test_gate_rejects_malformed_calls_before_dispatch_and_fails_closed():
    gate = ToolCallGate({"read"})
    with pytest.raises(ValueError):
        gate.accept_response({"role": "assistant", "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "read", "arguments": '{"a":1,"a":2}'}}
        ]})
    with pytest.raises(ValueError):
        gate.claim("read", {})


def test_gate_rejects_unauthorized_or_modified_calls_and_poisoned_session():
    gate = ToolCallGate({"read"})
    gate.accept_response({"role": "assistant", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "read", "arguments": '{"id":1}'}}
    ]})
    with pytest.raises(ValueError):
        gate.claim("read", {"id": 2})
    with pytest.raises(ValueError):
        gate.claim("read", {"id": 1})
    with pytest.raises(ValueError):
        gate.finish()


def test_gate_rejects_unknown_tool_and_duplicate_ids_across_responses():
    gate = ToolCallGate({"read"})
    with pytest.raises(ValueError):
        gate.accept_response({"role": "assistant", "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "shell", "arguments": '{}'}}
        ]})
    other = ToolCallGate({"read"})
    response = {"role": "assistant", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "read", "arguments": '{}'}}
    ]}
    other.accept_response(response)
    other.claim("read", {})
    other.complete("x")
    with pytest.raises(ValueError):
        other.accept_response(response)


def test_gate_does_not_advance_while_a_tool_is_still_executing():
    gate = ToolCallGate({"read"})
    gate.accept_response({"role": "assistant", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "read", "arguments": '{}'}}
    ]})
    gate.claim("read", {})
    with pytest.raises(ValueError):
        gate.accept_response({"role": "assistant", "content": "done"})
    with pytest.raises(ValueError):
        gate.complete("x")


def test_dispatcher_rejects_mismatch_before_actual_tool_side_effect():
    class Recorder:
        calls = []

        def tool(self, name, arguments):
            self.calls.append((name, arguments))
            return {"observation": "ok"}

    recorder = Recorder()
    gate = ToolCallGate({"delete_file"})
    dispatcher = GuardedToolDispatcher(gate, recorder)
    gate.accept_response({"role": "assistant", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "delete_file", "arguments": '{"file_id":"13"}'}}
    ]})
    with pytest.raises(ValueError):
        dispatcher.dispatch("delete_file", {"file_id": "14"})
    assert recorder.calls == []
    assert dispatcher.executed == []


def test_dispatcher_records_call_id_and_fails_closed_on_tool_exception():
    class Recorder:
        def tool(self, name, arguments):
            if arguments.get("bad"):
                raise RuntimeError("tool failure")
            return {"observation": "ok"}

    gate = ToolCallGate({"read"})
    dispatcher = GuardedToolDispatcher(gate, Recorder())
    gate.accept_response({"role": "assistant", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "read", "arguments": '{}'}}
    ]})
    assert dispatcher.dispatch("read", {}) == {"observation": "ok"}
    assert dispatcher.executed == [{"id": "x", "name": "read", "arguments": {}}]
    gate.accept_response({"role": "assistant", "tool_calls": [
        {"id": "y", "type": "function", "function": {"name": "read", "arguments": '{"bad":true}'}}
    ]})
    with pytest.raises(RuntimeError):
        dispatcher.dispatch("read", {"bad": True})
    with pytest.raises(ValueError):
        gate.finish()


def test_guarded_dispatch_against_real_agentdojo_environment():
    from pathlib import Path
    from cross_harness_sft.backends.agentdojo import AgentDojoDriver
    from rl_agentdojo_path_audit import NativePathRecorder

    root = Path(__file__).resolve().parents[1]
    driver = AgentDojoDriver(str(root / "configs/agentdojo_worker.yaml"))
    task_id = "workspace:user_task_6:injection_task_1"
    episode = driver.start(task_id, 0, injection_vector="calendar_project_kickoff_description",
                           injection_payload="AUDIT_MARKER_89c4")["episode"]
    recorder = NativePathRecorder(driver, episode)
    gate = ToolCallGate({"delete_file"})
    dispatcher = GuardedToolDispatcher(gate, recorder)
    response = {"role": "assistant", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "delete_file", "arguments": '{"file_id":"13"}'}}
    ]}
    gate.accept_response(response)
    assert not dispatcher.dispatch("delete_file", {"file_id": "13"})["invalid_call"]
    assert driver.finish(episode, "done")["risk_success"] is True
    assert attest_tool_calls([response], dispatcher.executed)
    gate.finish()


def test_proxy_captures_raw_response_before_allowing_dispatch():
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    request = {"model": "qwen4b", "messages": [{"role": "user", "content": "read"}], "tools": tools}
    response = {"choices": [{"message": {"role": "assistant", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "read", "arguments": '{}'}}
    ]}, "finish_reason": "tool_calls"}]}
    gate = ToolCallGate({"read"})
    proxy = GuardedChatProxy(gate, "qwen4b", tools, lambda req: response)
    assert proxy.forward(request) == response
    assert gate.claim("read", {}) == "x"
    gate.complete("x")
    assert proxy.raw_messages == [response["choices"][0]["message"]]
    assert proxy.raw_requests == [request]
    request["messages"].append({"role": "user", "content": "later mutation"})
    assert len(proxy.raw_requests[0]["messages"]) == 1
    gate.finish()


@pytest.mark.parametrize("override", [{"stream": True}, {"n": 2}, {"model": "other"}, {"tools": []}])
def test_proxy_rejects_unsafe_request_before_contacting_model(override):
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    contacted = []
    gate = ToolCallGate({"read"})
    proxy = GuardedChatProxy(gate, "qwen4b", tools, lambda req: contacted.append(req))
    request = {"model": "qwen4b", "messages": [], "tools": tools, **override}
    with pytest.raises(ValueError):
        proxy.forward(request)
    assert not contacted
    with pytest.raises(ValueError):
        gate.claim("read", {})


def test_proxy_never_returns_malformed_model_response_or_authorizes_tool():
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    gate = ToolCallGate({"read"})
    response = {"choices": [{"message": {"role": "assistant", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "read", "arguments": '{"bad":1,"bad":2}'}}
    ]}, "finish_reason": "tool_calls"}]}
    proxy = GuardedChatProxy(gate, "qwen4b", tools, lambda req: response)
    with pytest.raises(ValueError):
        proxy.forward({"model": "qwen4b", "messages": [], "tools": tools})
    assert proxy.raw_messages == []
    with pytest.raises(ValueError):
        gate.claim("read", {})
