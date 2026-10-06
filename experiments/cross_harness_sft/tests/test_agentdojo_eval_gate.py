"""Regression tests for the failures that invalidated the first AgentDojo run."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gateway = load("local_qwen_harness_gateway")
gate = load("check_agentdojo_eval")
sys.path.insert(0, str(SCRIPTS.parent / "src"))
from cross_harness_sft import native_harness as native


def test_codex_function_call_must_be_advertised_and_roundtrip():
    tools = [{"type":"namespace", "name":"mcp__benchmark", "tools":[
        {"type":"function", "name":"get_current_day", "parameters":{"type":"object"}}]}]
    calls = gateway.responses_function_calls([{"id":"call_1", "function":{
        "name":"get_current_day", "arguments":"{}"}}], tools)
    assert calls[0]["type"] == "function_call"
    assert calls[0]["namespace"] == "mcp__benchmark"
    assert calls[0]["name"] == "get_current_day"
    assert calls[0]["call_id"] == "call_1"
    assert "custom_tool_call" not in json.dumps(calls)
    messages, _ = gateway.responses_messages({"input":[calls[0],
        {"type":"function_call_output", "call_id":"call_1", "output":"2024-05-15"}]})
    assert messages[0]["tool_calls"][0]["id"] == "call_1"
    assert messages[1]["tool_call_id"] == "call_1"


def test_gateway_splits_flat_codex_tool_name_into_namespace_and_leaf():
    tools = [{"type":"namespace", "name":"mcp__benchmark", "tools":[
        {"type":"function", "name":"get_current_day", "parameters":{"type":"object"}}]}]
    call = gateway.responses_function_calls([{"id":"call_2", "function":{
        "name":"mcp__benchmark__get_current_day", "arguments":"{}"}}], tools)[0]
    assert call["namespace"] == "mcp__benchmark"
    assert call["name"] == "get_current_day"


def test_gateway_rejects_unadvertised_tool():
    with pytest.raises(ValueError, match="undeclared"):
        gateway.responses_function_calls([{"id":"c", "function":{"name":"exec", "arguments":"{}"}}], [])


def test_gate_rejects_missing_and_transport_errors(tmp_path):
    assert not gate.check(tmp_path, ["a:clean", "b:injection"], [0], smoke=True)
    for role in ("base", "sft"):
        for harness in ("codex_native", "claude_code"):
            group = tmp_path / role / harness
            group.mkdir(parents=True)
            rows = []
            for task, kind in (("a:clean","clean"),("b:injection","injection")):
                rows.append({"status":"completed", "native_stderr":"", "metadata":{
                    "task_id":task,"seed":0,"task_type":kind,"scoring_status":"scored",
                    "tool_history":[{"invalid_call":False,"observation":"ok"}],
                    "env_metrics":{"utility_reward":1,"safety_reward":1,"final_success":True,
                                   "risk_success":False,"parser_problem":False,"invalid_tool_call_count":0}}})
            (group/"raw_native.jsonl").write_text("\n".join(map(json.dumps, rows))+"\n")
    assert gate.check(tmp_path, ["a:clean","b:injection"], [0], smoke=True)
    broken = tmp_path/"sft"/"codex_native"/"raw_native.jsonl"
    rows = gate.records(broken)
    rows[0]["metadata"]["env_metrics"]["invalid_tool_call_count"] = 2
    broken.write_text("\n".join(map(json.dumps, rows))+"\n")
    # Model-generated unknown tools affect benchmark utility, not CLI transport.
    assert gate.check(tmp_path, ["a:clean","b:injection"], [0], smoke=True)
    rows[0]["metadata"]["env_metrics"]["parser_problem"] = True
    broken.write_text("\n".join(map(json.dumps, rows))+"\n")
    assert not gate.check(tmp_path, ["a:clean","b:injection"], [0], smoke=True)
    rows[0]["metadata"]["env_metrics"]["parser_problem"] = False
    rows[0]["native_stderr"] = "unsupported custom tool call: exec"
    broken.write_text("\n".join(map(json.dumps, rows))+"\n")
    assert not gate.check(tmp_path, ["a:clean","b:injection"], [0], smoke=True)


def test_claude_max_turns_is_a_controlled_incomplete_result(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDOJO_LOCAL_TOKEN", "smoke-token")
    harness = native.ClaudeCodeHarness({
        "id": "claude_code", "binary": "claude", "model": "qwen-base",
        "cli_model": "sonnet", "facade": {
            "base_url": "http://127.0.0.1:9444",
            "local_token_env": "AGENTDOJO_LOCAL_TOKEN",
        },
    })
    harness.executable = lambda: "claude"  # type: ignore[method-assign]
    events = [
        {"type": "assistant", "session_id": "s1", "message": {
            "role": "assistant", "content": [{"type": "tool_use", "id": "c1",
                "name": "mcp__benchmark__get_current_day", "input": {}}]}},
        {"type": "result", "subtype": "error_max_turns", "session_id": "s1"},
    ]
    completed = SimpleNamespace(
        returncode=1, stdout="\n".join(json.dumps(row) for row in events),
        stderr="", elapsed=1.0,
    )
    harness.invoke = lambda *args, **kwargs: completed  # type: ignore[method-assign]

    run = harness.run("task", tmp_path, {"command": "python", "args": [], "env": {}})
    assert run.terminal_status == "error_max_turns"
    assert run.final_answer == ""
    assert run.exit_code == 1
    assert len(run.native_events) == 2


def test_gate_allows_scored_max_turn_incomplete_but_not_unscored(tmp_path):
    for role in ("base", "sft"):
        for harness in ("codex_native", "claude_code"):
            group = tmp_path / role / harness
            group.mkdir(parents=True)
            rows = []
            for task, kind in (("a:clean", "clean"), ("b:injection", "injection")):
                rows.append({"status": "completed", "native_stderr": "", "metadata": {
                    "task_id": task, "seed": 0, "task_type": kind, "scoring_status": "scored",
                    "tool_history": [{"invalid_call": False, "observation": "ok"}],
                    "env_metrics": {"utility_reward": 0, "safety_reward": 1, "final_success": False,
                                    "risk_success": False, "parser_problem": False,
                                    "hit_max_turns": False}}})
            (group / "raw_native.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    row_path = tmp_path / "base" / "claude_code" / "raw_native.jsonl"
    rows = gate.records(row_path)
    rows[0]["status"] = "incomplete"
    rows[0]["metadata"]["env_metrics"]["hit_max_turns"] = True
    row_path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    assert gate.check(tmp_path, ["a:clean", "b:injection"], [0], smoke=True)
    rows[0]["metadata"]["scoring_status"] = "not_run"
    row_path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    assert not gate.check(tmp_path, ["a:clean", "b:injection"], [0], smoke=True)
