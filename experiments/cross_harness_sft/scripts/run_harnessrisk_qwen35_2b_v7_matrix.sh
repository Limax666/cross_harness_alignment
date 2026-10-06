#!/usr/bin/env bash
# Evaluate one Qwen3.5-2B v7 checkpoint through the official HarnessRisk adapters.
set -euo pipefail

ROOT=/data/home/liumingxiao/cross_harness_alignment
EXP="$ROOT/experiments/cross_harness_sft"
OUT="$EXP/outputs/harnessrisk_qwen35_2b_full_v7"
RUN_TAG="${RUN_TAG:-v7p1}"
NODE24="$EXP/runtime/node24/node_modules/node/bin"
if [[ -x "$NODE24/node" ]]; then
  export PATH="$NODE24:$PATH"
fi
ROLE="${ROLE:?Set ROLE=base or ROLE=sft}"
GPU="${GPU:?Set GPU to an idle card}"
PORT="${PORT:?Set a unique local port}"
HARNESS_LIST="${HARNESS_LIST:-hermes nanobot openclaw}"
REPETITIONS="${REPETITIONS:-3}"
CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-300}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5.4-nano}"
SFT_MODEL="$EXP/checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267"
BASE_MODEL=Qwen/Qwen3.5-2B-Base

[[ "$ROLE" == base || "$ROLE" == sft ]] || { echo "ROLE must be base or sft" >&2; exit 2; }
[[ "$REPETITIONS" =~ ^[1-9][0-9]*$ ]] || { echo "REPETITIONS must be positive" >&2; exit 2; }
[[ -f "$SFT_MODEL/model.safetensors" ]] || { echo "Merged full-SFT checkpoint missing" >&2; exit 2; }
[[ "$(find "$ROOT/HarnessRisk/data/HarnessRisk" -maxdepth 1 -name '*.json' | wc -l)" -eq 128 ]] || {
  echo "Expected exactly 128 HarnessRisk case files" >&2; exit 2;
}

model="$BASE_MODEL"
[[ "$ROLE" == sft ]] && model="$SFT_MODEL"
mkdir -p "$OUT"
protocol="$OUT/${RUN_TAG}-${ROLE}-${HARNESS_LIST// /-}-protocol.txt"
printf 'role=%s\nmodel=%s\ngpu=%s\nport=%s\nrepetitions=%s\nharnesses=%s\ntimeout_seconds=%s\nprompt_format=segment\ntemperature=0.3\nmax_input_tokens=16384\nmax_new_tokens=%s\nmax_requests_per_episode=24\nscorer=HarnessRisk canonical LLM judge\njudge_model=%s\njudge_endpoint=https://router.shengsuanyun.com/api/v1\nstarted_utc=%s\n' \
  "$ROLE" "$model" "$GPU" "$PORT" "$REPETITIONS" "$HARNESS_LIST" "$CASE_TIMEOUT_SECONDS" "$MAX_NEW_TOKENS" "$JUDGE_MODEL" "$(date -u +%FT%TZ)" > "$protocol"

for repetition in $(seq 1 "$REPETITIONS"); do
  for harness in $HARNESS_LIST; do
    echo "[$(date -u +%FT%TZ)] $ROLE $harness repetition $repetition/$REPETITIONS"
    ROLE="$ROLE" HARNESS="$harness" GPU="$GPU" PORT="$PORT" \
      BASE_MODEL="$model" MODEL_ID="qwen35-2b-${ROLE}-hr-v7" \
      FULL_MODEL=1 PROMPT_FORMAT=segment TEMPERATURE=0.3 \
      MAX_NEW_TOKENS="$MAX_NEW_TOKENS" MAX_INPUT_TOKENS=16384 MAX_REQUESTS_PER_EPISODE=24 \
      CASE_TIMEOUT_SECONDS="$CASE_TIMEOUT_SECONDS" \
      OUT="$OUT" BATCH_ID="${RUN_TAG}-${ROLE}-${harness}-r${repetition}" \
      SERVER_SESSION="hr2b-${ROLE}-${harness}-gpu${GPU}" \
      LOCAL_QWEN_SEED_URL="http://127.0.0.1:${PORT}" \
      BENCH_REPETITION="$repetition" RESUME_BATCH=1 \
      bash "$EXP/scripts/run_harnessrisk_local_qwen.sh" \
      >> "$OUT/${RUN_TAG}-${ROLE}-${harness}-r${repetition}.log" 2>&1
    batch="$OUT/runs/${harness}-${ROLE}/${RUN_TAG}-${ROLE}-${harness}-r${repetition}"
    /data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/python \
      "$EXP/scripts/judge_harnessrisk_shengsuanyun.py" "$batch" --deployment "$JUDGE_MODEL" \
      >> "$OUT/${RUN_TAG}-${ROLE}-${harness}-r${repetition}-judge.log" 2>&1
    python3 - "$batch" "$JUDGE_MODEL" <<'PY'
import json
import sys
from pathlib import Path

batch, model = Path(sys.argv[1]), sys.argv[2]
manifest = [json.loads(line) for line in (batch / "batch_manifest.jsonl").read_text().splitlines() if line.strip()]
judged = json.loads((batch / "llm_judge_multi_harness" / "manifest.json").read_text())
cases = {row.get("case_id") for row in manifest}
judge_cases = {row.get("case_id") for row in judged}
if len(manifest) != 128 or len(cases) != 128 or len(judged) != 128 or judge_cases != cases:
    raise SystemExit(f"Incomplete HarnessRisk batch {batch.name}: {len(manifest)}/128 trajectories, {len(judged)}/128 judge rows")
if any(row.get("status") not in ("judged", "existing") or
       (row.get("judge") or {}).get("_llm_metadata", {}).get("deployment") != model for row in judged):
    raise SystemExit(f"Incomplete HarnessRisk judge {batch.name}: status/model mismatch")
PY
    echo "[$(date -u +%FT%TZ)] completed $ROLE $harness repetition $repetition"
  done
done
printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)" >> "$protocol"
