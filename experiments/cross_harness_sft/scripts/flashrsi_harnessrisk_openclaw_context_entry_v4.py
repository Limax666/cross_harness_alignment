"""Original OpenClaw environment entry, explicit32K-compatible input budget."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'vendor/FlashRSI')]
from flashrsi_harnessrisk_openclaw_context_v4 import CONTEXT, apply_context, replace_once


def build_source():
    source = Path(__file__).with_name('flashrsi_harnessrisk_openclaw_environment_entry_v2.py').read_text()
    source = replace_once(source, "config=request['config'];", "config=apply_context(request['config']);")
    return replace_once(source, "'usage':result.get('usage',{}),",
                        "'usage':result.get('usage',{}),'runtime_context_metadata':CONTEXT_METADATA,")


if __name__ == '__main__':
    CONTEXT_METADATA = CONTEXT
    exec(compile(build_source(), __file__, 'exec'))
