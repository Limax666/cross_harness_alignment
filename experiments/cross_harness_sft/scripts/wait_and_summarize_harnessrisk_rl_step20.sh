#!/usr/bin/env bash
# Produce the SFT-vs-RL report only after all three independent jobs finish.
set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$EXP/outputs/harnessrisk_qwen35_2b_rl_step20"
SESSIONS=(hr-rl-hermes-gpu1 hr-rl-nanobot-gpu0 hr-rl-openclaw-gpu0)

while true; do
  active=0
  for session in "${SESSIONS[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then active=1; fi
  done
  [[ "$active" -eq 0 ]] && break
  sleep 60
done

for harness in hermes nanobot openclaw; do
  protocol="$OUT/rl20p1-rl-${harness}-protocol.txt"
  if [[ ! -f "$protocol" ]] || ! grep -q '^finished_utc=' "$protocol"; then
    printf 'Incomplete %s evaluation; inspect %s\n' "$harness" "$OUT/rl20p1-${harness}-coordinator.log" >&2
    exit 1
  fi
done
python3 "$EXP/scripts/summarize_harnessrisk_sft_rl_step20.py"
