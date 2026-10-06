#!/usr/bin/env bash
set -euo pipefail

cd /data/home/liumingxiao/cross_harness_alignment
experiment=experiments/cross_harness_sft
release="$experiment/data/sft/teacher_release_v5"
python_bin=/data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/python

for attempt in 1 2 3; do
  echo "Judge pass $attempt started at $(date -Is)"
  "$python_bin" "$experiment/scripts/build_teacher_sft_release.py" judge \
    --output-dir "$release" --model ali/qwen3.8-flash --workers 24 --timeout 120
  if "$python_bin" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "judged_complete")' \
    "$release/preprocessing_manifest.json"; then
    "$python_bin" "$experiment/scripts/verify_teacher_sft_release.py" "$release" --require-complete
    echo "Judge release complete at $(date -Is)"
    exit 0
  fi
done
echo "Judge release is incomplete after three passes; inspect preprocessing_manifest.json" >&2
exit 1
