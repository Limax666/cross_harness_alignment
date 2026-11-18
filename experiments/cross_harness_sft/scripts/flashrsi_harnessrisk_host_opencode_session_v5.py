"""v5 host: v4 isolation and privacy gates, with v5 audited supervisor."""
from pathlib import Path

source = (Path(__file__).with_name("flashrsi_harnessrisk_host_opencode_session_v4.py").read_text())
source = source.replace("flashrsi_harnessrisk_opencode_session_supervisor_v4.py", "flashrsi_harnessrisk_opencode_session_supervisor_v5.py")
source = source.replace("private-v1-opencode-native-session-v4-mime", "private-v1-opencode-native-session-v5-compaction")
if source == (Path(__file__).with_name("flashrsi_harnessrisk_host_opencode_session_v4.py").read_text()):
    raise RuntimeError("v4 host shape changed; refusing unreviewed transform")
exec(compile(source, "flashrsi_harnessrisk_host_opencode_session_v4.py", "exec"))
