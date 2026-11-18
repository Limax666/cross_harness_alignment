"""v6 host with a single, auditable compaction protocol identifier."""
from pathlib import Path

source = Path(__file__).with_name("flashrsi_harnessrisk_host_opencode_session_v4.py").read_text()
source = source.replace(
    "flashrsi_harnessrisk_opencode_session_supervisor_v4.py",
    "flashrsi_harnessrisk_opencode_session_supervisor_v6.py",
).replace(
    "private-v1-opencode-native-session-v4-mime",
    "private-v1-opencode-native-session-v6-compaction",
)
if "private-v1-opencode-native-session-v4-mime" in source:
    raise RuntimeError("v4 host protocol label remained after v6 transform")
exec(compile(source, "flashrsi_harnessrisk_host_opencode_session_v4.py", "exec"))
