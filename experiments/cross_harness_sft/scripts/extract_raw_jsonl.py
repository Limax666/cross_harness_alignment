"""Extract raw Claude Code session JSONLs from an isolation directory.

Usage: python extract_raw_jsonl.py <proj_dir> <dest_dir> <task_id>
"""
import re
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
task_id = sys.argv[3]

dst.mkdir(parents=True, exist_ok=True)
count = 0
for jf in src.rglob("*.jsonl"):
    agent_dir = jf.parent.name
    # Extract role from encoded path: everything after the 8-char hex run-id
    m = re.search(r"-[0-9a-f]{8}-(.+)$", agent_dir)
    role = m.group(1) if m else agent_dir
    dest = dst / f"{task_id}_{role}.jsonl"
    shutil.copy2(str(jf), str(dest))
    count += 1
print(count)
