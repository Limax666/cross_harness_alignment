#!/usr/bin/env bash
# Resume the real paired AgentDojo evaluation on at most two GPUs.
set -euo pipefail

ROOT="/data/home/liumingxiao/cross_harness_alignment"
EXP="$ROOT/experiments/cross_harness_sft"
OUT="$EXP/outputs/eval_agentdojo_9b"
MODEL_PY="/data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/python"
BENCH_PY="/data/home/liumingxiao/miniforge3/envs/harnessaudit/bin/python"
CHECKPOINT="$EXP/checkpoints/qwen35-9b-base-agentdojo-codex-claude-sft"
IFS=, read -r GPU_BASE GPU_SFT GPU_EXTRA <<<"${GPUS:-0,1}"
[[ -n "$GPU_BASE" && -n "$GPU_SFT" && -z "${GPU_EXTRA:-}" && "$GPU_BASE" != "$GPU_SFT" ]] || {
  echo "GPUS must contain exactly two distinct IDs, e.g. GPUS=0,1" >&2
  exit 2
}
command -v tmux >/dev/null 2>&1 || { echo "tmux is required" >&2; exit 2; }
[[ -x "$MODEL_PY" && -x "$BENCH_PY" ]] || { echo "evaluation environments are missing" >&2; exit 2; }
[[ -f "$CHECKPOINT/adapter_model.safetensors" ]] || { echo "completed SFT adapter is missing" >&2; exit 2; }
mkdir -p "$OUT"

python3 "$EXP/scripts/prepare_agentdojo_9b_eval.py" \
  --train "$EXP/data/sft/agentdojo_codex_claude_gpt56_sol_full/train.jsonl" \
  --validation "$EXP/data/sft/agentdojo_codex_claude_gpt56_sol_full/validation.jsonl" \
  --teacher-raw "$EXP/outputs/codex_agentdojo_native_gpt56_sol_full/raw_native.jsonl" \
  --output-dir "$OUT"

if ! curl --noproxy '*' --max-time 3 -fsS http://127.0.0.1:8111/health >/dev/null; then
  tmux new-session -d -s agentdojo-eval-worker \
    "cd '$ROOT' && PYTHONPATH='$EXP/src' '$BENCH_PY' -m cross_harness_sft.benchmark_server --driver cross_harness_sft.backends.agentdojo:create_driver --config '$EXP/configs/agentdojo_worker.yaml' --port 8111 > '$OUT/worker.log' 2>&1"
  for _ in {1..30}; do
    curl --noproxy '*' --max-time 3 -fsS http://127.0.0.1:8111/health >/dev/null 2>&1 && break
    sleep 1
  done
fi
curl --noproxy '*' --max-time 3 -fsS http://127.0.0.1:8111/health

for role in base sft; do
  session="agentdojo-eval-$role"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "$session is already running"
    continue
  fi
  gpu="$GPU_BASE"
  [[ "$role" == sft ]] && gpu="$GPU_SFT"
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && env CUDA_VISIBLE_DEVICES='$gpu' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '$MODEL_PY' '$EXP/scripts/evaluate_agentdojo_9b.py' --role '$role' --plan '$OUT/evaluation_plan.jsonl' --output-dir '$OUT' --checkpoint '$CHECKPOINT' >> '$OUT/$role.log' 2>&1"
  echo "$session started on GPU $gpu"
done

if ! tmux has-session -t agentdojo-eval-visualizer 2>/dev/null; then
  tmux new-session -d -s agentdojo-eval-visualizer \
    "bash '$EXP/scripts/watch_agentdojo_9b_comparison.sh' >> '$OUT/visualization.log' 2>&1"
fi

echo "Per-task results and logs: $OUT"
echo "Refresh charts: $MODEL_PY $EXP/scripts/plot_agentdojo_9b_comparison.py --output-dir $OUT"
