#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../" && pwd)"
EXP="$ROOT/experiments/cross_harness_sft"
python -m cross_harness_sft.preflight --config "$EXP/configs/formal_native.yaml" --mode collect
python -m cross_harness_sft.collect_native --config "$EXP/configs/formal_native.yaml"
python -m cross_harness_sft.verify --input "$EXP/outputs/formal/raw_native.jsonl" \
  --accepted "$EXP/outputs/formal/accepted.jsonl" --rejected "$EXP/outputs/formal/rejected.jsonl" \
  --report "$EXP/outputs/formal/filter_report.json" --max-turns 20 --max-response-length 16384
python -m cross_harness_sft.build_rft --input "$EXP/outputs/formal/accepted.jsonl" \
  --output-dir "$EXP/data/sft/formal" --validation-fraction 0.1 --max-turns 20 --max-response-length 16384
