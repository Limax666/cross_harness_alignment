#!/bin/bash
# Auto-resume AgentHarm collection (called by cron at 23:30)
LOG="/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/agentharm_codex_gpt56_sol_native/collect-val-v2-cron.log"

echo "[$(date -Iseconds)] Starting AgentHarm collection resume..." >> "$LOG"

cd /data/home/liumingxiao/cross_harness_alignment

# Check if already running
if pgrep -f "collect_native.*agentharm" >/dev/null 2>&1; then
    echo "[$(date -Iseconds)] Collection already running, skipping." >> "$LOG"
    exit 0
fi

nohup bash experiments/cross_harness_sft/scripts/run_agentharm_codex_native.sh \
    >> "$LOG" 2>&1 &

PID=$!
echo "$PID" > experiments/cross_harness_sft/outputs/agentharm_codex_gpt56_sol_native/collect.pid
echo "[$(date -Iseconds)] Collection started with PID $PID" >> "$LOG"
