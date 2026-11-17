"""Metadata and immutable transformation guards; no task/model/API execution."""
import copy
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / x) for x in ('scripts', 'src', 'vendor/FlashRSI')]


def test_openclaw_input_budget_preserves_resource_compaction_policy_and_other_limits():
    from flashrsi_harnessrisk_openclaw_context_v4 import apply_context, CONTEXT
    original = {'llm': {'model_name': 'qwen35-4b-preflight', 'max_tokens': 4096},
        'max_turns': 12, 'timeout': 360, 'compaction': {'enabled': False, 'tail_turns': 5},
        'context': {'max_tokens': 120000, 'other': 'unchanged'}}
    preserved = copy.deepcopy(original)
    measured = apply_context(original)
    assert measured['context'] == {'max_tokens': 24576, 'other': 'unchanged'}
    assert measured['llm'] == original['llm'] and measured['max_turns'] == 12 and measured['timeout'] == 360
    assert measured['compaction'] == original['compaction']
    assert original == preserved
    assert CONTEXT['max_context_tokens'] + 2 * CONTEXT['max_reply_tokens'] == CONTEXT['backend_context_window']


def test_future_v19_profile_has_actual_host_metadata_and_preserves_codex_repair():
    import flashrsi_harnessrisk_openclaw_context_entry_v4 as entry
    import flashrsi_harnessrisk_openclaw_context_supervisor_v4 as supervisor
    import flashrsi_harnessrisk_host_openclaw_context_v4 as host
    import flashrsi_harnessrisk_initial_v19 as initial
    import flashrsi_matrix_gpu1_queue_v19 as queue
    import flashrsi_matrix_repair_scan_v19 as scanner
    for module in (entry, supervisor, host, initial, queue, scanner):
        compile(module.build_source(), module.__file__, 'exec')
    namespace = {'__file__': queue.__file__, '__name__': 'pure_controller_check',
        'CONTEXT_METADATA': queue.CODEX_CONTEXT, 'OPENCLAW_CONTEXT_METADATA': queue.OPENCLAW_CONTEXT}
    exec(compile(queue.build_source(), queue.__file__, 'exec'), namespace)
    raw = b'{"model_revision":"source-test","flashrsi_revision":"runtime-test"}'
    contract = namespace['expected_initial_contract'](raw, 'openclaw')
    assert contract['host_script'] == 'flashrsi_harnessrisk_host_openclaw_context_v4.py'
    assert contract['native_context_metadata'] == queue.OPENCLAW_CONTEXT
    assert namespace['expected_initial_contract'](raw, 'codex')['native_context_metadata'] == queue.CODEX_CONTEXT
