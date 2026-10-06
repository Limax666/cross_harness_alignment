from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rl_hermes_mcp_contract import hermes_mcp_tools
from rl_agentdojo_path_audit import NativePathRecorder


def test_hermes_wire_schema_is_prefixed_and_reversible():
    native = [{"type": "function", "function": {"name": "get_current_day", "description": "day",
               "parameters": {"type": "object", "properties": {}}}}]
    wire, names = hermes_mcp_tools(native)
    assert wire[0]["function"]["name"] == "mcp__benchmark__get_current_day"
    assert names == {"mcp__benchmark__get_current_day": "get_current_day"}
    assert native[0]["function"]["name"] == "get_current_day"


@pytest.mark.parametrize("name", ["get.day", "X" * 60])
def test_hermes_wire_schema_rejects_unverifiable_names(name):
    with pytest.raises(ValueError):
        hermes_mcp_tools([{"function": {"name": name, "parameters": {"type": "object"}}}])


def test_wire_attempt_and_native_execution_are_audited_separately():
    from types import SimpleNamespace

    class Env:
        def model_dump(self, *, mode):
            return {"calendar": {"events": {}}, "inbox": {"emails": {}, "sent": []}}

    class Driver:
        def tool(self, episode, name, arguments):
            episode.trace.append(SimpleNamespace(function=name, args=arguments))
            return {"invalid_call": False, "observation": "ok"}

    episode = SimpleNamespace(environment=Env(), trace=[])
    recorder = NativePathRecorder(Driver(), episode)
    recorder.tool("mcp__benchmark__get_current_day", {}, native_name="get_current_day")
    recorder.raw_assistant_turns = [
        "<tool_call><function=mcp__benchmark__get_current_day></function></tool_call>", "Done"
    ]
    audit = recorder.audit("workspace:user_task_5:clean")
    assert episode.trace[0].function == "get_current_day"
    assert audit.all_calls_observed is True
    assert audit.prohibited_persistence is False
