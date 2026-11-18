"""Exercise the pinned native deterministic policy, without model generations."""
import copy
import re


def test_native_profile_reclaims_old_outputs_keeps_owner_and_newest_tool_round():
    import regex_rs
    if not hasattr(regex_rs, 'RegexError'):
        regex_rs.RegexError = ValueError
    if not hasattr(regex_rs, 'escape'):
        regex_rs.escape = re.escape
    from flashrsi_harnessrisk_hermes_context_profile_v13 import native_context_profile
    from flashrsi.agents.hermes.runtime.hermes_native.compression import ContextEngine
    messages = [{'role': 'system', 'content': 'Original safety policy'},
                {'role': 'user', 'content': 'Original owner task'}]
    for i in range(10):
        messages.extend([
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': str(i), 'type': 'function',
                'function': {'name': 'terminal', 'arguments': '{"command":"read"}'}}]},
            {'role': 'tool', 'tool_call_id': str(i), 'content': str(i) + 'x' * 7000},
        ])
    original = copy.deepcopy(messages)
    engine = ContextEngine({'context_length': 16384, 'compression': native_context_profile()},
        {'model_name': 'qwen35-4b-preflight', 'context_length': 16384, 'max_tokens': 4096})
    projected, count = engine.compressor.prune_tool_results_only(messages, 16000)
    assert count > 0
    assert projected[:2] == original[:2]
    assert projected[-2:] == original[-2:]
    assert len(projected) == len(original)
    assert messages == original
    assert engine.context_length == 16384
