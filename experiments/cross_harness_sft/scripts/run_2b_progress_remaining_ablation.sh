#!/usr/bin/env bash
# Sequential, bounded online engineering ablation; fresh optimizer per arm.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/../.."
EXP="$PWD/experiments/cross_harness_sft"
BASE="$EXP/outputs/rl/2b_verified_progress_ablation_remaining_20261030"
DIAG="$EXP/outputs/rl/diagnose_step150_backends_20261030"
VALIDATION="$EXP/outputs/rl/2b_whitebox_validation_20261030_v1/comparison.json"
[[ ! -e "$BASE/queue.started" ]] || { echo 'ablation queue already exists';exit 2; }
mkdir -p "$BASE"
python3 "$EXP/scripts/rl_progress_ablation_preflight.py" --validation "$VALIDATION" --deadline "$DIAG/ray_vllm_deadline.json" --replay "$DIAG/progress_reward_replayed_96_v3.json" --output "$BASE/preflight.json"
date -Is > "$BASE/queue.started"
export CUDA_VISIBLE_DEVICES=0,3 CUDA_HOME=/usr/local/cuda-12.6 MAX_JOBS=2 OMP_NUM_THREADS=2
export PATH="$CUDA_HOME/bin:$EXP/.venv-rl-pilot/bin:$PATH"
export RL_2B_INITIAL=chspo150 RL_RESPONSE_LENGTH=2048 RL_MAX_MODEL_LEN=8192 RL_SEPARATE_REFERENCE_GPU=false
export RL_STEPS=2 RL_SAVE_FREQ=2 VERL_GENERATION_TIMEOUT_SECONDS=480
for entry in 'GRPO-progress-v1:progress_protocol' 'CHS-PO-progress-v1:progress_protocol'; do
 algorithm="${entry%%:*}";variant="${entry#*:}"; name="${algorithm}_${variant}";run="$BASE/$name"
 [[ ! -e "$run" ]] || { echo "run exists: $run";exit 2; }
 mkdir -p "$run"
 export RL_ALGORITHM="$algorithm" RL_LOOP_CONFIG="$EXP/configs/rl_2b_whitebox_verified_${variant}.yaml" RL_RUN_DIR="$run"
 python3 - "$run" "$algorithm" "$variant" <<'PY'
import json,sys,datetime,pathlib
run=pathlib.Path(sys.argv[1]);data={'algorithm':sys.argv[2],'reward_variant':sys.argv[3],'oracle':'agentdojo-workspace-progress-v3','initial_actor_and_reference':'CHS-PO step150','optimizer':'fresh full parameter','steps':2,'checkpoint_every':2,'task_groups_per_step':24,'group_size':4,'assignment':'fixed K=2 whitebox ledger, consecutive configurations','sampling':'independent generated seeds; same cells but not paired policy RNG','gpus':[0,3],'response_length':2048,'request_timeout_seconds':480,'workflow':'soft','entropy_stop_above':3.0,'reference_kl_stop_above':0.1,'variable_group_fraction_stop_at_or_below':0.05,'warning':'engineering pilot; not formal RL result or heldout safety claim','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};(run/'run_manifest.json').write_text(json.dumps(data,indent=2)+'\n')
PY
 echo "$(date -Is) starting $entry" | tee -a "$BASE/queue.log"
 /data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/python "$EXP/scripts/watch_rl_9b_whitebox.py" "$run" > "$run/watch.log" 2>&1 & watcher=$!
 set +e
 bash "$EXP/scripts/train_rl_2b_whitebox.sh" +trainer.whitebox_pilot_health_guard=true > "$run/launch.log" 2>&1
 code=$?
 set -e
 printf '%s\n' "$code" > "$run/exit_code"
 wait "$watcher" || true
 if [[ "$code" != 0 ]]; then echo "$(date -Is) stopped $entry exit=$code" | tee -a "$BASE/queue.log";exit "$code"; fi
 [[ -s "$run/checkpoints/latest_checkpointed_iteration.txt" && "$(cat "$run/checkpoints/latest_checkpointed_iteration.txt")" == 2 ]] || { echo "missing step-2 checkpoint: $entry";exit 2; }
 echo "$(date -Is) finished $entry" | tee -a "$BASE/queue.log"
done
date -Is > "$BASE/queue.finished"
