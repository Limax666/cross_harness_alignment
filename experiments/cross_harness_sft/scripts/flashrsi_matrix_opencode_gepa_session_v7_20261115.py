"""Freeze and start the OpenCode GEPA v7 audit-replay repair."""
from pathlib import Path

source = Path(__file__).with_name("flashrsi_matrix_opencode_gepa_session_v6_20261115.py").read_text()
source = source.replace("session_v5_20261115", "session_PREDECESSOR_20261115")
source = source.replace("session_v6_20261115", "session_v7_20261115")
source = source.replace("v6-compaction", "v7-system-aware-compaction").replace("v6", "v7")
source = source.replace("session_PREDECESSOR_20261115", "session_v6_20261115")
source = source.replace(
    'sources = dict(old["sources_sha256"])',
    'sources = dict(old["sources_sha256"])\n    # Hermes-only quoted-reflection dispatch was added after v6; it is dormant for OpenCode and is frozen explicitly here.\n    sources["scripts/flashrsi_harnessrisk_algorithm.py"] = sha(ROOT / "scripts/flashrsi_harnessrisk_algorithm.py")',
)
source = source.replace(
    "v5 deterministic compaction was real but two hard-coded v4 metadata labels made the first evidence internally inconsistent.",
    "v6 replay audit omitted the regenerated runtime system prompt from its compaction threshold, rejecting an actual compacted request as unequal.",
).replace(
    "v5 host/supervisor retained hard-coded v4 evidence labels despite v5 compaction; audit label and execution protocol diverged.",
    "v6 audit replay compacted only persisted SQLite rows while runtime compaction includes an ephemeral regenerated system prompt.",
)
exec(compile(source, "flashrsi_matrix_opencode_gepa_session_v7_20261115.py", "exec"))
