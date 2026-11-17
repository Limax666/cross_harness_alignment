"""Same case environment/session/permissions, record the new input metadata."""
from pathlib import Path
from flashrsi_harnessrisk_openclaw_context_v4 import CONTEXT, replace_once


def build_source():
    source = Path(__file__).with_name('flashrsi_harnessrisk_openclaw_environment_supervisor_v2.py').read_text()
    source = replace_once(source, 'from flashrsi_harnessrisk_openclaw_session_audit_v2 import read_session, verify_turn',
        'from flashrsi_harnessrisk_openclaw_session_audit_v4 import read_session, verify_turn')
    source = replace_once(source, 'proxy.requests[request_start:],result,session_path))',
                          'proxy.requests[request_start:],result,session_path,Path(exported)))')
    source = replace_once(source, 'flashrsi_harnessrisk_openclaw_environment_entry_v2.py',
                          'flashrsi_harnessrisk_openclaw_context_entry_v4.py')
    source = replace_once(source, "'adapter_protocol':result.get('adapter_protocol')",
        "'adapter_protocol':result.get('adapter_protocol'),'runtime_context_metadata':result.get('runtime_context_metadata')")
    source = replace_once(source, 'result=json.loads(child.stdout)',
        "result=json.loads(child.stdout)\n                    if result.get('runtime_context_metadata') != CONTEXT_METADATA:raise RuntimeError('actual_openclaw_context_receipt_missing')")
    return source


if __name__ == '__main__':
    CONTEXT_METADATA = CONTEXT
    exec(compile(build_source(), __file__, 'exec'))
