"""Original native Codex entry plus versioned runtime-only model metadata."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
# This entry is launched with Python -I; only fixed trusted project roots are
# added, never the business working directory or actor-selected import paths.
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'vendor/FlashRSI')]
from flashrsi_harnessrisk_codex_context_v18 import wrap_settings, replace_once


def build_source():
    source = Path(__file__).with_name('flashrsi_harnessrisk_claude_entry.py').read_text()
    return replace_once(source, "'usage':dict(result.usage)",
        "'usage':dict(result.usage), 'runtime_context_metadata':CONTEXT_OBSERVED")


if __name__ == '__main__':
    import flashrsi.agents.codex as codex
    observed = []
    codex._settings = wrap_settings(codex._settings, observed)
    exec(compile(build_source(), __file__, 'exec'), {
        '__name__': '__main__', '__file__': __file__, 'CONTEXT_OBSERVED': observed})
