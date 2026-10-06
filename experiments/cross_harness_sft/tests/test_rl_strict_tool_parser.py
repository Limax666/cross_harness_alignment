from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rl_tool_attempt_audit import parse_strict_tool_turn

SCHEMAS = [{"type": "function", "function": {"name": "skill_view", "parameters": {
    "type": "object", "properties": {"name": {"type": "string"}, "file_path": {"type": "string"}},
    "required": ["name"], "additionalProperties": False,
}}}]


def test_json_value_syntax_matches_supervised_tool_format():
    content, calls = parse_strict_tool_turn(
        'Checking. <tool_call><function=skill_view><parameter=name>"hermes-agent"</parameter>'
        '</function></tool_call>', SCHEMAS)
    assert content == "Checking."
    assert calls == [{"name": "skill_view", "arguments": {"name": "hermes-agent"}}]


def test_exact_named_element_with_json_parameters_is_audited_alternate_envelope():
    raw = ('<tool_call><skill_view><parameter=name>"hermes-agent"</parameter>'
           '</skill_view></tool_call>')
    _, calls = parse_strict_tool_turn(raw, SCHEMAS)
    assert calls == [{"name": "skill_view", "arguments": {"name": "hermes-agent"}}]


def test_named_element_with_malformed_parameter_is_never_repaired():
    raw = '<tool_call><skill_view><parameter=name>hermes-agent</parameter></skill_view></tool_call>'
    with pytest.raises(ValueError, match="malformed or unclosed"):
        parse_strict_tool_turn(raw, SCHEMAS, allow_schema_errors=True)


@pytest.mark.parametrize("raw,reason", [
    ('<tool_call><function=skill_view(name=\'hermes-agent\')></function></tool_call>', "malformed"),
    ('<tool_call><function=skill_view><parameter=file_path>""</parameter></function></tool_call>', "missing required"),
    ('<tool_call><function=skill_view><parameter=name>hermes-agent</parameter></function></tool_call>', "JSON value"),
    ('<tool_call><function=skill_view><parameter=name>3</parameter></function></tool_call>', "wrong type"),
    ('<tool_call><function=skill_view><parameter=name>"a"</parameter><parameter=name>"b"</parameter></function></tool_call>', "duplicate"),
    ('<tool_call><function=unknown></function></tool_call>', "unknown tool"),
    ('<tool_call><function=skill_view><parameter=name>"a"</parameter>', "unclosed"),
])
def test_malformed_or_invalid_calls_fail_before_dispatch(raw, reason):
    with pytest.raises(ValueError, match=reason):
        parse_strict_tool_turn(raw, SCHEMAS)


NO_ARG_SCHEMA = [{"type": "function", "function": {"name": "get_current_day", "parameters": {
    "type": "object", "properties": {}, "additionalProperties": False,
}}}]


def test_unclosed_no_body_envelope_is_audited_alternate_envelope():
    raw = '<tool_call>\n<function=get_current_day>\n</tool_call>'
    content, calls = parse_strict_tool_turn(raw, NO_ARG_SCHEMA, allow_schema_errors=True)
    assert content == ""
    assert calls == [{"name": "get_current_day", "arguments": {}}]
    _, strict_calls = parse_strict_tool_turn(raw, NO_ARG_SCHEMA)
    assert strict_calls == calls


def test_unclosed_no_body_unknown_tool_is_blocked_not_terminal():
    raw = '<tool_call>\n<function=nonexistent_tool>\n</tool_call>'
    _, calls = parse_strict_tool_turn(raw, NO_ARG_SCHEMA, allow_schema_errors=True)
    assert calls == [{"name": "nonexistent_tool", "arguments": {},
                      "blocked_reason": "Unknown tool 'nonexistent_tool'. Use an exact available tool name."}]


def test_unclosed_no_body_required_args_stay_blocked():
    raw = '<tool_call>\n<function=skill_view>\n</tool_call>'
    _, calls = parse_strict_tool_turn(raw, SCHEMAS, allow_schema_errors=True)
    assert calls[0]["blocked_reason"].startswith("Missing required arguments")
    with pytest.raises(ValueError, match="missing required arguments"):
        parse_strict_tool_turn(raw, SCHEMAS)


def test_unclosed_envelope_with_body_is_never_repaired():
    raw = '<tool_call>\n<function=get_current_day><parameter=x>"1"</parameter>\n</tool_call>'
    with pytest.raises(ValueError, match="malformed or unclosed"):
        parse_strict_tool_turn(raw, NO_ARG_SCHEMA, allow_schema_errors=True)


def test_unclosed_no_body_envelope_attests_against_dispatched_call():
    from types import SimpleNamespace
    from rl_tool_attempt_audit import attest_native_calls
    raw = '<tool_call>\n<function=get_current_day>\n</tool_call>'
    assert attest_native_calls([raw], [SimpleNamespace(name="get_current_day", arguments={})],
                               schemas=NO_ARG_SCHEMA)
    assert not attest_native_calls([raw], [SimpleNamespace(name="get_current_day", arguments={"x": 1})],
                                   schemas=NO_ARG_SCHEMA)
