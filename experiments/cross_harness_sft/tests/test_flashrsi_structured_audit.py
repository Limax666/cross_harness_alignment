from pathlib import Path

import pytest

from cross_harness_sft.backends.agentdojo import AgentDojoDriver
from rl_agentdojo_path_audit import NativePathRecorder
from flashrsi_tool_attestation import ToolCallGate, GuardedChatProxy, GuardedToolDispatcher


def make_case(name, args, raw):
    root = Path(__file__).resolve().parents[1]
    driver = AgentDojoDriver(str(root / "configs/agentdojo_worker.yaml"))
    task = "workspace:user_task_6:injection_task_1"
    started = driver.start(task, 0, injection_vector="calendar_project_kickoff_description",
                           injection_payload="AUDIT_MARKER_89c4")
    episode, tools = started["episode"], started["tools"]
    recorder = NativePathRecorder(driver, episode, tool_schemas=tools)
    gate = ToolCallGate({t["function"]["name"] for t in tools})
    message = {"role": "assistant", "tool_calls": [
        {"id": "one", "type": "function", "function": {"name": name, "arguments": raw}}
    ]}
    responses = [{"choices": [{"message": message, "finish_reason": "tool_calls"}]},
                 {"choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]}]
    proxy = GuardedChatProxy(gate, "qwen4b", tools, lambda _: responses.pop(0))
    dispatcher = GuardedToolDispatcher(gate, recorder)
    proxy.forward({"model": "qwen4b", "messages": [], "tools": tools})
    result = dispatcher.dispatch(name, args)
    proxy.forward({"model": "qwen4b", "tools": tools, "messages": [message,
                  {"role": "tool", "tool_call_id": "one", "content": result["observation"]}]})
    return task, recorder, gate, proxy, dispatcher


@pytest.mark.parametrize("name,args,raw,expected", [
    ("get_current_day", {}, "{}", False),
    ("delete_file", {"file_id": "13"}, '{"file_id":"13"}', True),
])
def test_structured_audit_verifies_actual_native_effect(name, args, raw, expected):
    task, recorder, gate, proxy, dispatcher = make_case(name, args, raw)
    gate.finish()
    audit = recorder.audit_structured(task, gate=gate, proxy=proxy, dispatcher=dispatcher)
    assert audit.all_calls_observed and audit.side_effects_inspected
    assert audit.harmful_action_executed is expected
    assert audit.prohibited_persistence is expected


def test_structured_audit_abstains_when_unfinished_or_recorder_was_tampered():
    task, recorder, gate, proxy, dispatcher = make_case("delete_file", {"file_id": "13"}, '{"file_id":"13"}')
    assert not recorder.audit_structured(task, gate=gate, proxy=proxy, dispatcher=dispatcher).all_calls_observed
    gate.finish()
    recorder.records[0].arguments["file_id"] = "14"
    assert not recorder.audit_structured(task, gate=gate, proxy=proxy, dispatcher=dispatcher).all_calls_observed


def test_structured_audit_rejects_evidence_from_different_recorder():
    task, recorder, gate, proxy, dispatcher = make_case("get_current_day", {}, "{}")
    gate.finish()
    other = NativePathRecorder(recorder.driver, recorder.episode, tool_schemas=recorder.tool_schemas)
    assert not other.audit_structured(task, gate=gate, proxy=proxy, dispatcher=dispatcher).all_calls_observed


def test_structured_audit_abstains_when_model_did_not_receive_tool_observation():
    task, recorder, gate, proxy, dispatcher = make_case("get_current_day", {}, "{}")
    gate.finish()
    proxy._raw_requests[1]["messages"][1]["content"] = "tampered observation"
    assert not recorder.audit_structured(task, gate=gate, proxy=proxy, dispatcher=dispatcher).all_calls_observed


def test_structured_audit_abstains_without_followup_model_request():
    task, recorder, gate, proxy, dispatcher = make_case("get_current_day", {}, "{}")
    gate.finish()
    proxy._raw_requests.pop()
    proxy._raw_messages.pop()
    assert not recorder.audit_structured(task, gate=gate, proxy=proxy, dispatcher=dispatcher).all_calls_observed
