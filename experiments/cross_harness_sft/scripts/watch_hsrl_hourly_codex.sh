#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/home/liumingxiao/cross_harness_alignment
MON="$ROOT/experiments/cross_harness_sft/outputs/rl/hsrl_hourly_monitor_20261117"
THREAD=01a1aa6a-0266-7d60-9abf-997271f68415
export XDG_RUNTIME_DIR=/run/user/1026
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
mkdir -p "$MON/requests"
exec 9>"$MON/wake.lock"
if ! /usr/bin/flock -n 9; then printf '[%s] duplicate wake transport skipped\n' "$(date -Is)"; exit 0; fi
if [[ -f "$MON/disabled" ]]; then printf '[%s] RL hourly loop disabled\n' "$(date -Is)"; exit 0; fi
REQUEST_ID="$(/usr/bin/python3 -c 'import uuid; print(uuid.uuid4())')"
MESSAGE="$(< "$MON/wake_prompt.txt")"
MESSAGE="$MESSAGE 请求ID：$REQUEST_ID"
RECEIPT="$MON/requests/$REQUEST_ID.log"
printf '[%s] queueing RL audit request=%s thread=%s\n' "$(date -Is)" "$REQUEST_ID" "$THREAD"
set +e
/usr/bin/timeout --signal=TERM 60 /data/home/liumingxiao/.local/bin/codex queue -C "$ROOT" --thread "$THREAD" --message "$MESSAGE" >"$RECEIPT" 2>&1
CODE=$?
set -e
/usr/bin/python3 - "$MON" "$REQUEST_ID" "$CODE" "$RECEIPT" <<'STATE'
import datetime,json,sys
from pathlib import Path
mon,request,code,receipt=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3]),sys.argv[4]
row={'at':datetime.datetime.now().astimezone().isoformat(),'request_id':request,'returncode':code,'receipt_path':receipt,'status':'queued_not_yet_audited' if code==0 else 'queue_failed','thread_id':'01a1aa6a-0266-7d60-9abf-997271f68415'}
with (mon/'queue_events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
if code==0:
 p=mon/'last_queued.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(row,indent=2)+'\n');t.replace(p)
STATE
printf '[%s] RL audit request=%s queue_returncode=%s receipt=%s\n' "$(date -Is)" "$REQUEST_ID" "$CODE" "$RECEIPT"
exit "$CODE"
