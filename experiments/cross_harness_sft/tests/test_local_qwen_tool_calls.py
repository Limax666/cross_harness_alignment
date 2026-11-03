"""Regression checks for tool names and required arguments in local eval."""

import importlib.util
from pathlib import Path
import torch


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


def test_foreign_harness_tool_alias_is_rejected():
    schemas = [{"type": "function", "function": {
        "name": "tool_search", "parameters": {"type": "object", "properties": {
            "queries": {"type": "array"}}, "required": ["queries"]}}}]
    raw = "<tool_call><function=skill_search><parameter=queries>[\"mockctl\"]" \
          "</parameter></function></tool_call>"
    calls, errors = server.parse_tool_calls(raw, schemas)
    assert calls == []
    assert errors == ["unknown tool 'skill_search'; available tools: tool_search"]


def test_real_skill_search_is_not_rewritten_to_other_harness_tool():
    schemas = [{"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": {"queries": {"type": "array"}}, "required": ["queries"]}}}
        for name in ("skill_search", "tool_search")]
    raw = '<tool_call><function=skill_search><parameter=queries>["mockctl"]' \
          '</parameter></function></tool_call>'
    calls, errors = server.parse_tool_calls(raw, schemas)
    assert errors == []
    assert calls == [("skill_search", {"queries": ["mockctl"]})]


def test_namespaced_tool_suffix_is_not_automatically_repaired():
    schemas = [{"type": "function", "function": {
        "name": "mcp__benchmark__get_current_day", "parameters": {"type": "object"}}}]
    calls, errors = server.parse_tool_calls(
        "<tool_call><function=get_current_day></function></tool_call>", schemas)
    assert calls == []
    assert errors == ["unknown tool 'get_current_day'; available tools: mcp__benchmark__get_current_day"]


def test_segment_prompt_does_not_advertise_a_foreign_harness_tool():
    schemas = [{"type": "function", "function": {"name": "dir_list", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
    prompt = server.segment_prompt([{"role": "user", "content": "List the workspace."}], schemas)
    assert "skill_view" not in prompt
    assert "hermes-agent" not in prompt
    assert "dir_list" in prompt
    assert "Include every required argument" in prompt


def test_long_segment_prompt_preserves_complete_tools_and_system_context():
    class CharacterTokenizer:
        def __call__(self, value, **kwargs):
            return {"input_ids": torch.tensor([[ord(char) for char in value]])}

    schemas = [{"type": "function", "function": {"name": "dir_list", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
    messages = [
        {"role": "system", "content": "NEMOCLAW_SYSTEM_SENTINEL"},
        {"role": "user", "content": "old " * 500},
        {"role": "tool", "content": "old tool result " * 300},
        {"role": "user", "content": "NEW_USER_SENTINEL"},
    ]
    ids, trim = server.encode_segment_prompt(CharacterTokenizer(), messages, schemas, 900)
    rendered = "".join(map(chr, ids["input_ids"][0].tolist()))
    assert trim is not None and trim["dropped_messages"] >= 1
    assert len(rendered) <= 900
    assert "dir_list" in rendered and '"required": ["path"]' in rendered
    assert "NEMOCLAW_SYSTEM_SENTINEL" in rendered
    assert "NEW_USER_SENTINEL" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")


def test_visible_strips_qwen_turn_end_and_serialized_following_turn():
    raw = "Task complete.<|im_end|>\\n<|im_start|>user\\nnext prompt"
    assert server.visible(raw) == "Task complete."


def test_pseudo_call_rejects_nonliteral_expression():
    raw = "<tool_call><function=skill_view(name=run())></function></tool_call>"
    calls, errors = server.parse_tool_calls(raw, SKILL_VIEW)
    assert calls == []
    assert errors[0].startswith("invalid tool-call signature:")
