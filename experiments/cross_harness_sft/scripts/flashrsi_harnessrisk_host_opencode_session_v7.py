"""v7 host with system-aware deterministic-compaction audit identifier."""
from pathlib import Path

source = Path(__file__).with_name("flashrsi_harnessrisk_host_opencode_session_v6.py").read_text()
source = source.replace("supervisor_v6.py", "supervisor_v7.py").replace("v6-compaction", "v7-system-aware-compaction")
exec(compile(source, "flashrsi_harnessrisk_opencode_session_v7.py", "exec"))
