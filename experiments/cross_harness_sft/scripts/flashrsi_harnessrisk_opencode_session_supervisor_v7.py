"""v7 supervisor: v6 execution with system-aware deterministic audit."""
from pathlib import Path

source = Path(__file__).with_name("flashrsi_harnessrisk_opencode_session_supervisor_v6.py").read_text()
source = source.replace("flashrsi_harnessrisk_opencode_session_audit_v5", "flashrsi_harnessrisk_opencode_session_audit_v7").replace("v6-compaction", "v7-system-aware-compaction")
exec(compile(source, "flashrsi_harnessrisk_opencode_session_supervisor_v7.py", "exec"))
