"""Frozen v6 OpenCode launcher with one consistent compaction protocol label."""
from pathlib import Path

source = Path(__file__).with_name("flashrsi_matrix_opencode_session_v4_20261114.py").read_text()
source = source.replace(
    "flashrsi_harnessrisk_host_opencode_session_v4.py",
    "flashrsi_harnessrisk_host_opencode_session_v6.py",
).replace(
    "native session v4 with builtin MIME",
    "native session v6 deterministic compaction",
).replace(
    "private-v1-opencode-native-session-v4-mime",
    "private-v1-opencode-native-session-v6-compaction",
)
if "flashrsi_harnessrisk_host_opencode_session_v4.py" in source:
    raise RuntimeError("v4 launcher host remained after v6 transform")
exec(compile(source, "flashrsi_matrix_opencode_session_v4_20261114.py", "exec"))
