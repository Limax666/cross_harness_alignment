#!/usr/bin/env bash
# Passive tmux handoff: wait for the current driver, then resume only missing batches.
set -euo pipefail

ROOT=/data/home/liumingxiao/cross_harness_alignment
EXP="$ROOT/experiments/cross_harness_sft"
OUT="$EXP/outputs/harnessrisk_qwen35_2b_full_v7"
: "${ROLE:?Set base or sft}"
: "${HARNESS:?Set hermes or openclaw}"
: "${GPU:?Set GPU 1 for base or 2 for sft}"
: "${PORT:?Set the existing role/harness server port}"
: "${RUN_TAG:?Set v7p5 for base or v7p6 for sft}"
: "${ACTIVE_PID:?Set the currently running matrix driver PID}"

[[ "$HARNESS" == hermes || "$HARNESS" == openclaw ]] || exit 2
if [[ "$ROLE" == base ]]; then
  [[ "$GPU" == 1 && "$RUN_TAG" == v7p5 ]] || exit 2
elif [[ "$ROLE" == sft ]]; then
  [[ "$GPU" == 2 && "$RUN_TAG" == v7p6 ]] || exit 2
else
  exit 2
fi
[[ "$PORT" =~ ^[0-9]+$ && "$ACTIVE_PID" =~ ^[0-9]+$ ]] || exit 2

mkdir -p "$OUT/locks"
exec 9>"$OUT/locks/${ROLE}-${HARNESS}.lock"
flock -n 9 || { echo "Another watcher owns ${ROLE}/${HARNESS}" >&2; exit 3; }

echo "[$(date -u +%FT%TZ)] waiting for existing ${ROLE}/${HARNESS} driver PID ${ACTIVE_PID}"
while true; do
  state="$(ps -o stat= -p "$ACTIVE_PID" 2>/dev/null || true)"
  [[ -n "$state" && "$state" != Z* ]] || break
  sleep 30
done

# A completed batch may contain scoreable failures, but a missing trajectory
# cannot be judged. Retry only those cases, preserving the old attempt.
if python3 - "$OUT" "$HARNESS" "$ROLE" "$RUN_TAG" <<'PY'
import json, sys
from pathlib import Path
root, harness, role, tag = Path(sys.argv[1]), *sys.argv[2:]
for repetition in (1, 2, 3):
    name = f"{tag}-{role}-{harness}-r{repetition}"
    batch = root / "runs" / f"{harness}-{role}" / name
    manifest = batch / "batch_manifest.jsonl"
    judge = batch / "llm_judge_multi_harness" / "manifest.json"
    if not manifest.is_file() or not judge.is_file():
        raise SystemExit(1)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(rows) != 128 or len({row.get("case_id") for row in rows}) != 128:
        raise SystemExit(1)
    judge_rows = json.loads(judge.read_text())
    if len(judge_rows) != 128 or {row.get("case_id") for row in judge_rows} != {row.get("case_id") for row in rows}:
        raise SystemExit(1)
PY
then
  echo "[$(date -u +%FT%TZ)] all three ${ROLE}/${HARNESS} batches recorded and judged; no restart"
  exit 0
fi

echo "[$(date -u +%FT%TZ)] resuming only unrecorded ${ROLE}/${HARNESS} cases on GPU ${GPU}"
export ROLE GPU PORT RUN_TAG
export REPETITIONS=3 CASE_TIMEOUT_SECONDS=300 MAX_NEW_TOKENS=1024
export JUDGE_MODEL=openai/gpt-5.4-nano
export RETRY_MISSING_TRAJECTORY=1
if command -v ionice >/dev/null 2>&1; then
  exec ionice -c 3 bash "$EXP/scripts/resume_harnessrisk_${HARNESS}_qwen35_2b_v7.sh"
fi
exec bash "$EXP/scripts/resume_harnessrisk_${HARNESS}_qwen35_2b_v7.sh"
