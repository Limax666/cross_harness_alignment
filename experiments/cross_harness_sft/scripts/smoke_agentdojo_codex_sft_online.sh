#!/usr/bin/env bash
# Live current-policy tool rollouts through native Codex CLI + AgentDojo.
# This is a rollout/contract canary only; it does not update policy weights.
set -euo pipefail

ROOT=/data/home/liumingxiao/cross_harness_alignment
EXP="$ROOT/experiments/cross_harness_sft"
TRAIN_PY=/data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/python
NATIVE_PY="$EXP/.venv-rl-native/bin/python"
MODEL="$EXP/checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267"
RUN_ID="agentdojo-codex-sft-online-canary-$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$EXP/outputs/rl/$RUN_ID"
TASK_CLEAN=workspace:user_task_6:clean
TASK_INJECTION=workspace:user_task_6:injection_task_0
BENCH_PORT=9341
MODEL_PORT=9342
SCRATCH="$(mktemp -d /tmp/agentdojo-online-canary.XXXXXX)"
BENCH_PID=""
MODEL_PID=""

cleanup() {
  [[ -z "$MODEL_PID" ]] || kill "$MODEL_PID" 2>/dev/null || true
  [[ -z "$BENCH_PID" ]] || kill "$BENCH_PID" 2>/dev/null || true
  wait "$MODEL_PID" 2>/dev/null || true
  wait "$BENCH_PID" 2>/dev/null || true
  rm -rf "$SCRATCH"
}
trap cleanup EXIT INT TERM

[[ -f "$MODEL/model.safetensors" ]] || { echo "Missing SFT checkpoint: $MODEL" >&2; exit 2; }
mkdir -p "$OUT"
cat > "$OUT/protocol.txt" <<EOF
run_id=$RUN_ID
rollout_mode=online_current_policy
policy=$MODEL
harness=Codex CLI (native)
benchmark=AgentDojo v1.2.2
split=train
source_family=agentdojo:workspace:user_task_6
arms=clean,injection_task_0
rollouts_per_prompt=2
seeds=17,29
optimizer_update=false
GPU=0; GPU3 intentionally left idle
EOF

if command -v ionice >/dev/null 2>&1; then
  ionice -c 3 env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$TRAIN_PY" "$EXP/scripts/local_qwen_openai_server.py" --role sft \
    --base-model "$MODEL" --model-id qwen35-2b-sft-online-canary \
    --host 127.0.0.1 --port "$MODEL_PORT" --full-precision \
    --prompt-format segment --temperature 0.6 --max-new-tokens 512 \
    --max-input-tokens 4096 --max-requests-per-episode 24 \
    > "$OUT/policy_server.log" 2>&1 &
else
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$TRAIN_PY" "$EXP/scripts/local_qwen_openai_server.py" --role sft \
    --base-model "$MODEL" --model-id qwen35-2b-sft-online-canary \
    --host 127.0.0.1 --port "$MODEL_PORT" --full-precision \
    --prompt-format segment --temperature 0.6 --max-new-tokens 512 \
    --max-input-tokens 4096 --max-requests-per-episode 24 \
    > "$OUT/policy_server.log" 2>&1 &
fi
MODEL_PID=$!

PYTHONPATH="$EXP/src" "$NATIVE_PY" -m cross_harness_sft.benchmark_server \
  --driver cross_harness_sft.backends.agentdojo:create_driver \
  --config "$EXP/configs/agentdojo_worker.yaml" --host 127.0.0.1 --port "$BENCH_PORT" \
  > "$OUT/agentdojo_worker.log" 2>&1 &
BENCH_PID=$!

ready=0
for _ in $(seq 1 120); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:$MODEL_PORT/health" >/dev/null 2>&1 \
     && curl --noproxy '*' -fsS "http://127.0.0.1:$BENCH_PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$MODEL_PID" 2>/dev/null || ! kill -0 "$BENCH_PID" 2>/dev/null; then
    echo "A local canary server exited during startup; inspect $OUT/*.log" >&2
    exit 1
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo "Timed out waiting for local servers; inspect $OUT/*.log" >&2; exit 1; }

python3 - "$SCRATCH/config.yaml" "$OUT/rollouts.jsonl" "$OUT/episodes" "$BENCH_PORT" "$MODEL_PORT" <<'PY'
import sys, yaml
config_path, output, workdir, bench_port, model_port = sys.argv[1:]
value = {
    "experiment_id": "agentdojo_codex_sft_online_canary",
    "output_jsonl": output,
    "episode_workdir": workdir,
    "model_family": "qwen35-2b-sft-v7",
    "seeds": [17, 29],
    "benchmark_sources": [{"type": "agentdojo_http", "base_url": f"http://127.0.0.1:{bench_port}"}],
    "harnesses": [{
        "id": "codex_sft_online_canary", "type": "codex_cli",
        "binary": "/data/home/liumingxiao/.local/bin/codex",
        "model": "qwen35-2b-sft-online-canary", "provider": "local_qwen",
        "provider_name": "Local Qwen SFT", "base_url": f"http://127.0.0.1:{model_port}/v1",
        "env_key": "LOCAL_QWEN_API_KEY", "wire_api": "chat",
        "stream_logs": True, "timeout_seconds": 600,
        "mcp_approval_mode": "approve", "mcp_startup_timeout_seconds": 30,
        "mcp_tool_timeout_seconds": 90,
    }],
    "safety_modules": ["none"],
    "pretool_guard": {"deny_tools": []},
    "safety": {"inject_harness_context": True},
}
with open(config_path, "w", encoding="utf-8") as stream:
    yaml.safe_dump(value, stream, sort_keys=False)
PY

export LOCAL_QWEN_API_KEY=local-canary-no-upstream-credential
export PYTHONPATH="$EXP/src${PYTHONPATH:+:$PYTHONPATH}"
"$NATIVE_PY" -m cross_harness_sft.collect_native \
  --config "$SCRATCH/config.yaml" --no-resume \
  --task-id "$TASK_CLEAN" --task-id "$TASK_INJECTION" \
  > "$OUT/collector.log" 2>&1

"$TRAIN_PY" - "$OUT/collector.log" <<'PY'
import json, re, sys
text = open(sys.argv[1], encoding="utf-8").read()
matches = re.findall(r"\{\s*\"planned\"\s*:[^}]+\}", text)
if not matches:
    raise SystemExit("collector did not emit a run summary")
summary = json.loads(matches[-1])
if summary.get("planned") != 4 or summary.get("completed") != 4 or summary.get("failed") != 0:
    raise SystemExit(f"online rollout canary failed: {summary}")
rows = [json.loads(line) for line in open(summary["output"], encoding="utf-8") if line.strip()]
if len(rows) != 4 or any(row.get("status") != "completed" or not row.get("messages") for row in rows):
    raise SystemExit(f"expected four complete online trajectories; got {len(rows)} rows")
PY

cat > "$OUT/status.json" <<EOF
{"status":"online_rollouts_completed_no_optimizer_update","run_id":"$RUN_ID","policy":"$MODEL","harness":"codex_cli","benchmark":"agentdojo","split":"train","rollouts_per_prompt":2,"gpu":"0","reserved_idle_gpu":"3"}
EOF
cat "$OUT/status.json"
