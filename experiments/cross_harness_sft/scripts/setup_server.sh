#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/cross_harness_sft"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m venv "$EXP/.venv"
source "$EXP/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "$EXP"
echo "Core runtime installed. No mock pipeline or tests were executed."
echo "Next: setup_benchmarks.sh, install_harnesses.sh, start_benchmark_workers.sh."
