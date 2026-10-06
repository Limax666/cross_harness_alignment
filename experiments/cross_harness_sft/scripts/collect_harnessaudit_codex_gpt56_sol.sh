#!/usr/bin/env bash
# Collect the complete HarnessAudit multi-agent suite with its official Codex runner.
# Re-running this script is safe: completed tasks are skipped by their result artifacts.
set -euo pipefail

REPO_ROOT="/data/home/liumingxiao/cross_harness_alignment/HarnessAudit"
PYTHON_BIN="/data/home/liumingxiao/miniforge3/envs/harnessaudit/bin/python"
OUT_ROOT="/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/harnessaudit_codex_gpt56_sol_full"

export HARNESS="codex"
export FRAMEWORK="clawteam"
export MODEL="gpt-5.6-sol"
export CODEX_AUTH_MODE="cli_login"
export TASK_WORKERS="${TASK_WORKERS:-1}"
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-900}"
export MAX_TURNS="${MAX_TURNS:-30}"
export SKIP_JUDGE="1"
export SKIP_EXISTING="1"
export PYTHON_CMD="$PYTHON_BIN"
export OUTPUT_DIR="$OUT_ROOT/results"
export TRACE_DIR="$OUT_ROOT/traces"
export LOG_DIR="$OUT_ROOT/run_logs"
export MASP_WORKSPACES_ROOT="$OUT_ROOT/workspaces"
export MASP_KEEP_CODEX_ISOLATION="1"
export MASP_CODEX_ISOLATION_ROOT="$OUT_ROOT/codex_isolation"

mkdir -p "$OUTPUT_DIR" "$TRACE_DIR" "$LOG_DIR" "$MASP_WORKSPACES_ROOT" "$MASP_CODEX_ISOLATION_ROOT"
cd "$REPO_ROOT"
exec bash multi_agent/run_ma.sh
