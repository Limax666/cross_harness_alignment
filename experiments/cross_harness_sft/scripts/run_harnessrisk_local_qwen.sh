#!/usr/bin/env bash
# Run HarnessRisk official adapters against the SFT Qwen model.
set -euo pipefail

ROOT="/data/home/liumingxiao/cross_harness_alignment"
BENCH="$ROOT/HarnessRisk"
EXP="$ROOT/experiments/cross_harness_sft"
OUT="${OUT:-$EXP/outputs/harnessrisk_local_qwen}"
MODEL_PY="/data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/python"
HERMES_CMD="${HERMES_CMD:-/data/home/liumingxiao/.hermes/hermes-agent/venv/bin/hermes}"
NANOBOT_CMD="${NANOBOT_CMD:-$EXP/.venv-nanobot/bin/nanobot}"
OPENCLAW_ROOT="${OPENCLAW_ROOT:-$ROOT/openclaw-source}"
ROLE="${ROLE:-sft}"
HARNESS="${HARNESS:-hermes}"
if [[ "$HARNESS" == openclaw && -x "$EXP/runtime/node24/node_modules/node/bin/node" ]]; then
  export PATH="$EXP/runtime/node24/node_modules/node/bin:$PATH"
fi
GPU="${GPU:-0}"
CASES="${CASES:-}"
PORT="${PORT:-9000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-8192}"
HERMES_MAX_TURNS="${HERMES_MAX_TURNS:-20}"
CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-480}"
NANOBOT_MAX_TOOL_ITERATIONS="${NANOBOT_MAX_TOOL_ITERATIONS:-16}"
MODEL_ID="${MODEL_ID:-qwen35-${ROLE}-local}"
CHECKPOINT="${CHECKPOINT:-$EXP/checkpoints/qwen35-9b-base-multiharness-verl-sft-v2/hf_merged/lora_adapter}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B-Base}"
USE_ADAPTER="${USE_ADAPTER:-1}"
STATE="$OUT/state/${HARNESS}-${ROLE}"
RUNS="$OUT/runs/${HARNESS}-${ROLE}"
FULL_MODEL="${FULL_MODEL:-0}"
PROMPT_FORMAT="${PROMPT_FORMAT:-native}"
TEMPERATURE="${TEMPERATURE:-0}"
EVAL_SEED="${EVAL_SEED:-42}"
MAX_REQUESTS_PER_EPISODE="${MAX_REQUESTS_PER_EPISODE:-0}"
BATCH_ID="${BATCH_ID:-local-${ROLE}-$(date -u +%Y%m%dT%H%M%SZ)}"
STATE="$OUT/state/${HARNESS}-${ROLE}-${BATCH_ID}"

[[ "$HARNESS" == hermes || "$HARNESS" == nanobot || "$HARNESS" == openclaw ]] || { echo 'HARNESS must be hermes, nanobot, or openclaw' >&2; exit 2; }
[[ "$ROLE" == base || "$ROLE" == sft ]] || { echo 'ROLE must be base or sft' >&2; exit 2; }
[[ "$CASE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo 'CASE_TIMEOUT_SECONDS must be a positive integer' >&2; exit 2; }
[[ "$NANOBOT_MAX_TOOL_ITERATIONS" =~ ^[1-9][0-9]*$ ]] || { echo 'NANOBOT_MAX_TOOL_ITERATIONS must be a positive integer' >&2; exit 2; }
[[ -x "$MODEL_PY" && -d "$BENCH/data/HarnessRisk" ]] || { echo 'model environment or HarnessRisk data missing' >&2; exit 2; }
if [[ "$ROLE" == sft && "$FULL_MODEL" != 1 ]]; then
  [[ -f "$CHECKPOINT/adapter_model.safetensors" ]] || { echo "SFT adapter missing: $CHECKPOINT" >&2; exit 2; }
fi

mkdir -p "$OUT/logs" "$STATE" "$RUNS"
export LOCAL_QWEN_KEY=local-harnessrisk-key
server_session="${SERVER_SESSION:-harnessrisk-qwen-${ROLE}-${GPU}}"
if ! tmux has-session -t "$server_session" 2>/dev/null; then
  adapter_arg=""
  if [[ "$ROLE" == sft && "$USE_ADAPTER" == 1 && "$FULL_MODEL" != 1 ]]; then
    adapter_arg="--adapter '$CHECKPOINT'"
  fi
  full_arg=""
  [[ "$FULL_MODEL" == 1 ]] && full_arg="--full-precision"
  tmux new-session -d -s "$server_session" \
    "cd '$ROOT' && env CUDA_VISIBLE_DEVICES='$GPU' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '$MODEL_PY' '$EXP/scripts/local_qwen_openai_server.py' --role '$ROLE' --base-model '$BASE_MODEL' --model-id '$MODEL_ID' --port '$PORT' --max-new-tokens '$MAX_NEW_TOKENS' --max-input-tokens '$MAX_INPUT_TOKENS' --prompt-format '$PROMPT_FORMAT' --temperature '$TEMPERATURE' --seed '$EVAL_SEED' --max-requests-per-episode '$MAX_REQUESTS_PER_EPISODE' $full_arg $adapter_arg > '$OUT/logs/${HARNESS}-${ROLE}-server.log' 2>&1"
fi
for _ in {1..180}; do curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break; sleep 1; done
curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health" >/dev/null || { echo 'local Qwen server did not start' >&2; exit 1; }
health_json="$(curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health")"
python3 - "$health_json" "$MODEL_ID" "$MAX_NEW_TOKENS" "$MAX_INPUT_TOKENS" "$PROMPT_FORMAT" "$TEMPERATURE" <<'PY'
import json, sys
health = json.loads(sys.argv[1])
expected = {"model": sys.argv[2], "max_new_tokens": int(sys.argv[3]),
            "max_input_tokens": int(sys.argv[4]), "prompt_format": sys.argv[5],
            "temperature": float(sys.argv[6])}
if any(health.get(key) != value for key, value in expected.items()):
    raise SystemExit(f"Local model server configuration mismatch: expected {expected}, got {health}")
PY

if [[ "$HARNESS" == hermes || "$HARNESS" == nanobot ]]; then
  CMD="$HERMES_CMD"
  [[ "$HARNESS" == nanobot ]] && CMD="$NANOBOT_CMD"
  [[ -x "$CMD" ]] || { echo "$HARNESS missing: $CMD" >&2; exit 2; }
  NANOBOT_MAX_TOOL_ITERATIONS="$NANOBOT_MAX_TOOL_ITERATIONS" "$BENCH/harness_adapter/scripts/setup_scripts/setup_agent.sh" --harness "$HARNESS" --provider local-qwen --model "$MODEL_ID" \
    --base-url "http://127.0.0.1:${PORT}/v1" --api-key-env LOCAL_QWEN_KEY --cmd "$CMD" --state-dir "$STATE" --run-root "$RUNS" --strict
  # Hermes stores its turn limit in home/config.yaml. Nanobot has a
  # different state layout, so attempting this write for Nanobot makes the
  # evaluation wrapper exit before run_batch.sh is launched.
  if [[ "$HARNESS" == hermes ]]; then
    [[ "$HERMES_MAX_TURNS" =~ ^[1-9][0-9]*$ ]] || { echo 'HERMES_MAX_TURNS must be a positive integer' >&2; exit 2; }
    if [[ "${RESUME_BATCH:-0}" != 1 ]] || ! grep -qE '^[[:space:]]*max_turns:' "$STATE/home/config.yaml"; then
      printf '  max_turns: %s\n' "$HERMES_MAX_TURNS" >> "$STATE/home/config.yaml"
    fi
  fi
else
  [[ -f "$OPENCLAW_ROOT/scripts/run-node.mjs" ]] || { echo "OpenClaw missing: $OPENCLAW_ROOT" >&2; exit 2; }
  export OPENCLAW_REPO_ROOT="$OPENCLAW_ROOT"
  if [[ "${RESUME_BATCH:-0}" != 1 || ! -f "$STATE/openclaw.json" ]]; then
    "$BENCH/harness_adapter/scripts/setup_scripts/setup_agent.sh" --harness openclaw --provider local-qwen --model "$MODEL_ID" \
      --base-url "http://127.0.0.1:${PORT}/v1" --api-key-env LOCAL_QWEN_KEY --compatibility openai --state-dir "$STATE" --non-interactive --strict
  else
    python3 - "$STATE/openclaw.json" "$MODEL_ID" "$PORT" <<'PY'
import json, sys
config_path = sys.argv[1]
config = json.load(open(config_path, encoding="utf-8"))
provider = config["models"]["providers"]["local-qwen"]
assert any(model["id"] == sys.argv[2] for model in provider["models"])
expected = f"http://127.0.0.1:{sys.argv[3]}/v1"
if provider["baseUrl"] != expected:
    provider["baseUrl"] = expected
    with open(config_path, "w", encoding="utf-8") as out:
        json.dump(config, out, ensure_ascii=False, indent=2)
        out.write("\n")
print("OpenClaw model URL:", provider["baseUrl"])
PY
    echo "Reusing validated OpenClaw state: $STATE"
  fi
fi
extra=()
[[ -n "$CASES" ]] && extra+=(--cases "$CASES")
if [[ "$HARNESS" == hermes || "$HARNESS" == nanobot ]]; then
  "$BENCH/harness_adapter/scripts/run_case_scripts/run_batch.sh" --harness "$HARNESS" --multiturn --data-dir "$BENCH/data/HarnessRisk" --run-root "$RUNS" --batch-id "$BATCH_ID" "${extra[@]}" -- \
    --state-dir "$STATE" --cmd "$CMD" --model "$MODEL_ID" --provider local-qwen --api-key-env LOCAL_QWEN_KEY --timeout-seconds "$CASE_TIMEOUT_SECONDS"
else
  "$BENCH/harness_adapter/scripts/run_case_scripts/run_batch.sh" --harness openclaw --data-dir "$BENCH/data/HarnessRisk" --run-root "$RUNS" --batch-id "$BATCH_ID" "${extra[@]}" -- \
    --state-dir "$STATE" --model "local-qwen/$MODEL_ID" --api-key-env LOCAL_QWEN_KEY --timeout-seconds "$CASE_TIMEOUT_SECONDS"
fi
