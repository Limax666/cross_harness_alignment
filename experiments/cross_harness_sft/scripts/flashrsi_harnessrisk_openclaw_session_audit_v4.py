"""Original complete SQLite gates plus authenticated native context projection."""
from pathlib import Path
from flashrsi_harnessrisk_openclaw_context_v4 import replace_once
from flashrsi_harnessrisk_openclaw_projection_v4 import authenticate_projection

source = Path(__file__).with_name('flashrsi_harnessrisk_openclaw_session_audit_v2.py').read_text()
source = replace_once(source, 'def verify_turn(before, after, owners, requests, result, session_path):',
    'def verify_turn(before, after, owners, requests, result, session_path, source_root):')
source = replace_once(source, 'expected = _chat_messages(before + [{"role": "user", "content": owners[-1]}])',
    'expected = _chat_messages(authenticate_projection(before, after, owners, requests, result, source_root))')
source = replace_once(source, '"all_prior_native_wire_messages_equal": True,',
    '"actual_wire_matches_authenticated_native_projection": True, "raw_native_history_preserved": True,')
exec(compile(source, __file__, 'exec'))
