from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rl_tool_attempt_audit import attest_native_calls


def test_exact_qwen_xml_call_matches():
    raw = '<tool_call><function=get_day_calendar_events><parameter=day>"2024-05-19"</parameter></function></tool_call>'
    assert attest_native_calls([raw, "Task completed."],
                               [SimpleNamespace(name="get_day_calendar_events", arguments={"day": "2024-05-19"})])


def test_bare_text_with_no_native_tool_is_observed():
    assert attest_native_calls(["I cannot complete this."], [])


def test_missing_raw_turns_fails_closed():
    assert not attest_native_calls([], [])


def test_malformed_function_block_fails_closed():
    assert not attest_native_calls(["<tool_call><function=send_email><parameter=recipients>[oops]</parameter></function></tool_call>"], [])
    assert not attest_native_calls(["<tool_call><function=send_email"], [])


def test_parser_dropped_call_fails_closed():
    raw = '<tool_call><function=send_email><parameter=recipients>["attacker@example.com"]</parameter></function></tool_call>'
    assert not attest_native_calls([raw], [])


def test_native_call_arguments_must_match_raw():
    raw = '<tool_call><function=send_email><parameter=recipients>["attacker@example.com"]</parameter></function></tool_call>'
    call = SimpleNamespace(name="send_email", arguments={"recipients": ["sarah@example.com"]})
    assert not attest_native_calls([raw], [call])


def test_duplicate_parameter_fails_closed():
    raw = '<tool_call><function=get_current_day><parameter=day>1</parameter><parameter=day>2</parameter></function></tool_call>'
    assert not attest_native_calls([raw], [])
