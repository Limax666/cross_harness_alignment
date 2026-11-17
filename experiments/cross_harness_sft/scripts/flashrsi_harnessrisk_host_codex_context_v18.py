"""Same native relay and isolation, new Codex metadata supervisor only."""
from pathlib import Path
from flashrsi_harnessrisk_codex_context_v18 import replace_once


def build_source():
    source = Path(__file__).with_name('flashrsi_harnessrisk_host_native_v3.py').read_text()
    return replace_once(source, "'scripts/flashrsi_harnessrisk_supervisor.py'",
                        "'scripts/flashrsi_harnessrisk_codex_context_supervisor_v18.py'")


if __name__ == '__main__':
    exec(compile(build_source(), __file__, 'exec'))
