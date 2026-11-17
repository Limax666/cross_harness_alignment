"""Native metadata/configuration checks, no model, API or source task execution."""
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / x) for x in ('scripts', 'src', 'vendor/FlashRSI')]


def test_actual_native_model_reserves_two_reply_windows_without_changing_snapshot(tmp_path):
    from flashrsi_harnessrisk_codex_context_v18 import CONTEXT, wrap_settings
    from flashrsi.agents.codex import _settings
    from flashrsi.agents.codex.python_core.models import Model
    (tmp_path / 'config.toml').write_text('model_context_window = 272000\n')
    old = (tmp_path / 'config.toml').read_bytes()
    observed = []
    override, _, _ = wrap_settings(_settings, observed)(
        {'llm': {'model_name': 'qwen35-4b-preflight', 'api_key': 'unit-placeholder',
                 'base_url': 'http://127.0.0.1:1/v1'}}, tmp_path, tmp_path)
    native = Model('qwen35-4b-preflight', override)
    assert native.info['context_window'] == 32768
    assert native.auto_compact_token_limit == 24576
    assert native.auto_compact_token_limit + 2 * CONTEXT['max_reply_tokens'] == 32768
    assert (tmp_path / 'config.toml').read_bytes() == old
    assert observed == [CONTEXT]
    assert 'unit-placeholder' not in str(observed)


def test_new_profile_transforms_only_future_entry_host_initial_and_queue():
    import flashrsi_harnessrisk_codex_context_entry_v18 as entry
    import flashrsi_harnessrisk_codex_context_supervisor_v18 as supervisor
    import flashrsi_harnessrisk_host_codex_context_v18 as host
    import flashrsi_harnessrisk_initial_v18 as initial
    import flashrsi_matrix_gpu1_queue_v18 as queue
    import flashrsi_matrix_repair_scan_v18 as scanner
    for module in (entry, supervisor, host, initial, queue, scanner):
        compile(module.build_source(), str(module.__file__), 'exec')
    assert 'flashrsi_harnessrisk_codex_context_entry_v18.py' in supervisor.build_source()
    assert 'flashrsi_harnessrisk_codex_context_supervisor_v18.py' in host.build_source()
    assert 'flashrsi_harnessrisk_host_codex_context_v18.py' in queue.build_source()
    assert 'flashrsi_harnessrisk_initial_v18.py' in queue.build_source()
    assert "'native_context_metadata':CONTEXT_METADATA" in initial.build_source()
    namespace = {'__file__': queue.__file__, '__name__': 'pure_controller_check',
                 'CONTEXT_METADATA': queue.CONTEXT}
    exec(compile(queue.build_source(), queue.__file__, 'exec'), namespace)
    contract = namespace['expected_initial_contract'](
        b'{"model_revision":"source-test","flashrsi_revision":"runtime-test"}', 'codex')
    assert contract['host_script'] == 'flashrsi_harnessrisk_host_codex_context_v18.py'
    assert contract['native_context_metadata'] == queue.CONTEXT
    assert contract['max_turns_per_owner_turn'] is None
    assert contract['native_timeout_seconds_per_owner_turn'] == 360
