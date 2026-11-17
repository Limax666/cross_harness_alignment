"""Existing corrected OpenClaw environment host, new metadata supervisor."""
from pathlib import Path
from flashrsi_harnessrisk_openclaw_context_v4 import replace_once


def build_source():
    source = Path(__file__).with_name('flashrsi_harnessrisk_host_openclaw_environment_v2.py').read_text()
    source = replace_once(source, 'scripts/flashrsi_harnessrisk_openclaw_environment_supervisor_v2.py',
                          'scripts/flashrsi_harnessrisk_openclaw_context_supervisor_v4.py')
    return replace_once(source, "'HR_HARNESS':os.environ.get('HR_HARNESS','miniagent')}",
        "'HR_HARNESS':os.environ.get('HR_HARNESS','miniagent'),'HR_OPENCLAW_CASE_ENV_V3':'1'}")


if __name__ == '__main__':
    exec(compile(build_source(), __file__, 'exec'))
