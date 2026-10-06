#!/usr/bin/env bash
# Collect the complete HarnessAudit suite with its official Claude Code runner
# routed to OpenRouter's OpenAI gpt-5.6-sol model.
set -euo pipefail

REPO_ROOT="/data/home/liumingxiao/cross_harness_alignment/HarnessAudit"
PYTHON_BIN="/data/home/liumingxiao/miniforge3/envs/harnessaudit/bin/python"
KEY_ENV="/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.env"
OUT_ROOT="/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/harnessaudit_claude_openrouter_gpt56_sol_full"
CLAUDE_BIN_DIR="/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/runtime/claude-code/node_modules/.bin"

if [[ ! -f "$KEY_ENV" ]]; then
  echo "Missing OpenRouter environment file: $KEY_ENV" >&2
  exit 1
fi

set -a
source "$KEY_ENV"
set +a
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is absent from $KEY_ENV" >&2
  exit 1
fi

export PATH="$CLAUDE_BIN_DIR:$PATH"
export HARNESS="claude"
export FRAMEWORK="clawteam"
export MODEL="openai/gpt-5.6-sol"
export CLAUDE_CODE_AUTH_MODE="api_key"
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
export ANTHROPIC_API_KEY=""
export HTTP_PROXY="http://ustc-course.com:48527"
export HTTPS_PROXY="http://ustc-course.com:48527"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
export CLAUDE_CODE_SUBAGENT_MODEL="$MODEL"
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
export MASP_KEEP_CLAUDE_ISOLATION="1"
export MASP_CLAUDE_ISOLATION_ROOT="$OUT_ROOT/claude_isolation"

mkdir -p "$OUTPUT_DIR" "$TRACE_DIR" "$LOG_DIR" "$MASP_WORKSPACES_ROOT" "$MASP_CLAUDE_ISOLATION_ROOT"
cd "$REPO_ROOT"
exec bash multi_agent/run_ma.sh
