"""Original confined supervisor, selecting only a new Codex metadata entry."""
from pathlib import Path
from flashrsi_harnessrisk_codex_context_v18 import replace_once


def build_source():
    source = Path(__file__).with_name('flashrsi_harnessrisk_supervisor.py').read_text()
    source = replace_once(source, "'flashrsi_harnessrisk_claude_entry.py' if harness in ('claude_code','codex')",
        "'flashrsi_harnessrisk_codex_context_entry_v18.py' if harness=='codex' else 'flashrsi_harnessrisk_claude_entry.py' if harness=='claude_code'")
    source = replace_once(source, "'messages':result.get('messages',[]),'usage':result.get('usage')",
        "'messages':result.get('messages',[]),'usage':result.get('usage'),'runtime_context_metadata':result.get('runtime_context_metadata')")
    source = replace_once(source, "if result.get('stop_reason')!='completed':status='error';break",
        "if harness=='codex' and result.get('stop_reason')=='completed' and result.get('runtime_context_metadata') != [CONTEXT_METADATA]:raise RuntimeError('native_context_override_not_observed')\n                    if result.get('stop_reason')!='completed':status='error';break")
    return source


if __name__ == '__main__':
    from flashrsi_harnessrisk_codex_context_v18 import CONTEXT
    exec(compile(build_source(), __file__, 'exec'), {
        '__name__': '__main__', '__file__': __file__, 'CONTEXT_METADATA': CONTEXT})
