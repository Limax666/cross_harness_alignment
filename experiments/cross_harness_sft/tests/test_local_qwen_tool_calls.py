"""Regression checks for tool names and required arguments in local eval."""

import importlib.util
from pathlib import Path


SERVER = Path(__file__).resolve().parents[1] / "scripts/local_qwen_openai_server.py"
spec = importlib.util.spec_from_file_location("local_qwen_openai_server", SERVER)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)

SKILL_VIEW = [{"type": "function", "function": {
    "name": "skill_view", "parameters": {"type": "object", "properties": {
        "name": {"type": "string"}, "file_path": {"type": "string"}}, "required": ["name"]}}}]


def test_inline_signature_becomes_json_argument():
    raw = "<tool_call><function=skill_view(name='hermes-agent')>" \
          "<parameter=file_path>\"\"</parameter></function></tool_call>"
    calls, errors = server.parse_tool_calls(raw, SKILL_VIEW)
    assert errors == []
    assert calls == [("skill_view", {"name": "hermes-agent", "file_path": ""})]


def test_missing_name_is_not_dispatched():
    raw = "<tool_call><function=skill_view>" \
          "<parameter=file_path>\"\"</parameter></function></tool_call>"
    calls, errors = server.parse_tool_calls(raw, SKILL_VIEW)
    assert calls == []
    assert errors == ["skill_view: missing required argument(s): name"]


def test_unknown_tool_is_not_fuzzy_repaired():
    raw = "<tool_call><function=skill_view_xyz></function></tool_call>"
    calls, errors = server.parse_tool_calls(raw, SKILL_VIEW)
    assert calls == []
    assert errors == ["unknown tool 'skill_view_xyz'; available tools: skill_view"]


def test_hermes_bare_tool_tag_with_mismatched_closing_tag():
    raw = "<tool_call><skill_view><parameter=name>\"hermes-agent\"</parameter></function></tool_call>"
    calls, errors = server.parse_tool_calls(raw, SKILL_VIEW)
    assert errors == []
    assert calls == [("skill_view", {"name": "hermes-agent"})]


def test_hermes_tool_call_with_missing_function_equals_prefix():
    schemas = [{"type": "function", "function": {
        "name": "tool_search", "parameters": {"type": "object", "properties": {
            "queries": {"type": "array"}, "limit": {"type": "integer"}}, "required": ["queries"]}}}]
    raw = "<tool_call><tool_search><parameter=queries>[\"mockctl\"]</parameter>" \
          "<parameter=limit>10</parameter></function></tool_call>"
    calls, errors = server.parse_tool_calls(raw, schemas)
    assert errors == []
    assert calls == [("tool_search", {"queries": ["mockctl"], "limit": 10})]


def test_skill_search_alias_requires_matching_tool_schema():
    schemas = [{"type": "function", "function": {
        "name": "tool_search", "parameters": {"type": "object", "properties": {
            "queries": {"type": "array"}}, "required": ["queries"]}}}]
    raw = "<tool_call><function=skill_search><parameter=queries>[\"mockctl\"]" \
          "</parameter></function></tool_call>"
    calls, errors = server.parse_tool_calls(raw, schemas)
    assert errors == []
    assert calls == [("tool_search", {"queries": ["mockctl"]})]
def test_visible_strips_qwen_turn_end_and_serialized_following_turn():
    raw = "Task complete.<|im_end|>\\n<|im_start|>user\\nnext prompt"
    assert server.visible(raw) == "Task complete."


def test_pseudo_call_rejects_nonliteral_expression():
    raw = "<tool_call><function=skill_view(name=run())></function></tool_call>"
    calls, errors = server.parse_tool_calls(raw, SKILL_VIEW)
    assert calls == []
    assert errors[0].startswith("invalid tool-call signature:")
