"""Frozen v7 OpenCode launcher with system-aware compaction audit."""
from pathlib import Path

source = Path(__file__).with_name("flashrsi_matrix_opencode_session_v6_20261115.py").read_text()
source = source.replace("host_opencode_session_v6.py", "host_opencode_session_v7.py").replace("v6 deterministic compaction", "v7 system-aware deterministic compaction").replace("v6-compaction", "v7-system-aware-compaction")
if "host_opencode_session_v6.py" in source:
    raise RuntimeError("v6 launcher host remained after v7 transform")
exec(compile(source, "flashrsi_matrix_opencode_session_v7_20261115.py", "exec"))
