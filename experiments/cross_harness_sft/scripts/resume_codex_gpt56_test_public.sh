#!/bin/bash
# Resume Codex + GPT-5.6-sol AgentHarm test_public collection
LOG="/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/agentharm_codex_gpt56_sol_native/collect-test-public-v1.log"

echo "[$(date -Iseconds)] Resuming Codex + GPT-5.6-sol test_public collection..." >> "$LOG"

cd /data/home/liumingxiao/cross_harness_alignment

# Check if already running
if pgrep -f "collect_native.*agentharm_codex_gpt56_sol_test_public" >/dev/null 2>&1; then
    echo "[$(date -Iseconds)] Already running, skipping." >> "$LOG"
    exit 0
fi

AGENTHARM_CONFIG=experiments/cross_harness_sft/configs/agentharm_codex_gpt56_sol_test_public.yaml \
AGENTHARM_WORKER_CONFIG=experiments/cross_harness_sft/configs/agentharm_worker_test_public.yaml \
nohup bash experiments/cross_harness_sft/scripts/run_agentharm_codex_native.sh --resume \
    >> "$LOG" 2>&1 &

PID=$!
echo "$PID" > experiments/cross_harness_sft/outputs/agentharm_codex_gpt56_sol_native/collect-test-public.pid
echo "[$(date -Iseconds)] Started with PID $PID" >> "$LOG"
