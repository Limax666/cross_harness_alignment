"""v5 wrapper: run v4 supervisor with registered deterministic compaction."""
from pathlib import Path

source = (Path(__file__).with_name("flashrsi_harnessrisk_opencode_session_supervisor_v4.py").read_text())
source = source.replace(
    "from flashrsi_harnessrisk_opencode_session_audit_v4 import read_session, verify_turn",
    "from flashrsi_harnessrisk_opencode_session_audit_v5 import read_session, verify_turn",
)
source = source.replace(
    "'config':{'llm':request['llm_config'],'max_turns':12,'timeout':360}",
    "'config':{'llm':request['llm_config'],'max_turns':12,'timeout':360,'context':{'max_tokens':6000},'compaction':{'enabled':True,'threshold':6000,'preserve_recent_tokens':2048,'tail_turns':1,'prune':True}}",
)
if source == (Path(__file__).with_name("flashrsi_harnessrisk_opencode_session_supervisor_v4.py").read_text()):
    raise RuntimeError("v4 supervisor shape changed; refusing unreviewed transform")
exec(compile(source, "flashrsi_harnessrisk_opencode_session_supervisor_v4.py", "exec"))
