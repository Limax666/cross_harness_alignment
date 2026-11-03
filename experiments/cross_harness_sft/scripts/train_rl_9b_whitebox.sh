#!/usr/bin/env bash
# 9B whitebox online acceptance run; true CLI products are held out.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-rl-pilot"
RL_MODEL_SIZE="${RL_MODEL_SIZE:-9b}"
if [[ "$RL_MODEL_SIZE" == 2b ]]; then
  SFT="$ROOT/checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267"
  if [[ "${RL_2B_INITIAL:-sft267}" == chspo150 ]]; then
    SFT="$ROOT/outputs/rl/multiharness_chspo_r20g_v4_from130_20261012/hf_merged_step_150"
  elif [[ "${RL_2B_INITIAL:-sft267}" != sft267 ]]; then
    echo 'RL_2B_INITIAL must be sft267 or chspo150' >&2; exit 2
  fi
else
  [[ "$RL_MODEL_SIZE" == 9b ]] || { echo 'RL_MODEL_SIZE must be 2b or 9b'; exit 2; }
  SFT="$ROOT/checkpoints/qwen35-9b-base-v7-a6000_20261018_r7/hf_merged_step_291"
fi
POOL="${RL_WHITEBOX_POOL:-$ROOT/outputs/rl/9b_whitebox_pool_v4}"
DATA="$POOL/train.parquet"
LEDGER="$POOL/assignment_ledger.json"
LOOP="${RL_LOOP_CONFIG:-$ROOT/configs/rl_9b_whitebox_loop.yaml}"
RUN="${RL_RUN_DIR:?explicit fresh RL_RUN_DIR required}"
RL_ALGORITHM="${RL_ALGORITHM:-CHS-PO-whitebox-v1}"
RL_STEPS="${RL_STEPS:-20}"
RL_ATTEMPT_EPOCHS="$RL_STEPS"
RL_SAVE_FREQ="${RL_SAVE_FREQ:-10}"
RL_PROMPT_LENGTH=4096
RL_RESPONSE_LENGTH="${RL_RESPONSE_LENGTH:-8192}"
RL_MAX_MODEL_LEN="${RL_MAX_MODEL_LEN:-12288}"
RL_AGENT_NUM_WORKERS=2
RL_REWARD_NUM_WORKERS=1
RL_ACTOR_SP_SIZE=1
RL_ACTOR_PARAM_OFFLOAD="${RL_ACTOR_PARAM_OFFLOAD:-true}"
RL_FUSED_KERNELS=false
RL_GPU_UTIL=0.43
RL_MAX_NUM_SEQS=1
RL_N_GPUS="${RL_N_GPUS:-2}"
RL_SEPARATE_REFERENCE_GPU="${RL_SEPARATE_REFERENCE_GPU:-true}"
RL_TRAIN_BATCH_SIZE=24
if [[ "$RL_SEPARATE_REFERENCE_GPU" == true ]]; then default_devices="0,2,3"; else default_devices="0,3"; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$default_devices}"
[[ "$RL_N_GPUS" == 2 ]] || { echo 'This layout requires two actor/rollout GPUs'; exit 2; }
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
if [[ "$RL_SEPARATE_REFERENCE_GPU" == true ]]; then
  [[ ${#devices[@]} == 3 ]] || { echo 'Separate reference requires three visible GPUs'; exit 2; }
  [[ "${devices[0]}" != "${devices[1]}" && "${devices[0]}" != "${devices[2]}" && "${devices[1]}" != "${devices[2]}" ]] || { echo 'GPU IDs must be distinct'; exit 2; }
elif [[ "$RL_SEPARATE_REFERENCE_GPU" == false ]]; then
  [[ ${#devices[@]} == 2 && "${devices[0]}" != "${devices[1]}" ]] || { echo 'Colocated reference requires two distinct GPUs'; exit 2; }
else
  echo 'RL_SEPARATE_REFERENCE_GPU must be true or false'; exit 2
fi
if [[ "${RL_CONFIG_ONLY:-0}" != 1 ]]; then
for device in "${devices[@]}"; do
 used=$(nvidia-smi -i "$device" --query-gpu=memory.used --format=csv,noheader,nounits)
 (( used < 256 )) || { echo "GPU $device busy ($used MiB), refusing to launch"; exit 2; }
done
available_kib=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
required_ram_gib=153
if [[ "$RL_MODEL_SIZE" == 2b ]]; then required_ram_gib=80; fi
(( available_kib > required_ram_gib * 1024 * 1024 )) || { echo "RAM guard: need >${required_ram_gib} GiB available; currently $((available_kib/1024/1024)) GiB"; exit 2; }
[[ ! -e "$RUN/launch.started" ]] || { echo 'Run directory already launched'; exit 2; }
[[ -f "$DATA" && -f "$LEDGER" && -f "$SFT/model.safetensors" ]] || exit 2
mkdir -p "$RUN"
date -Is > "$RUN/launch.started"
fi
export PYTHONPATH="$ROOT/vendor/verl:$ROOT/scripts:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# r11 OOM (2026-10-28): Ray killed update_actor ranks at 96.1% node RAM; enable
# the validated Project-local optimizer park so post-update weight sync does not
# overlap CPU-Adam anonymous pages with vLLM wake + FSDP state-dict collection.
if [[ "$RL_MODEL_SIZE" == 2b ]]; then
  # The 2B FSDP2 optimizer has CPU DTensor states; the file-backed 9B
  # parking helper accepts plain Tensor only. 2B has ample measured RAM headroom.
  export VERL_PARK_OPTIMIZER_STATES_FOR_SYNC=0
else
  export VERL_PARK_OPTIMIZER_STATES_FOR_SYNC=1
fi
export VERL_REFERENCE_KEEP_ON_GPU=1
export VERL_SYNC_WEIGHT_DTYPE=bf16
if [[ "$RL_SEPARATE_REFERENCE_GPU" == true ]]; then export VERL_REF_FUSED_TORCH=1; else export VERL_REF_FUSED_TORCH=0; fi
export VERL_REF_FUSED_CHUNK_SIZE=128
export VERL_CHUNKED_SCALED_LOGPROBS=1
export VERL_QWEN_SCHEMA_STRING_PARAMS=1
export VERL_GENERATION_TIMEOUT_SECONDS="${VERL_GENERATION_TIMEOUT_SECONDS:-0}"
export NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_P2P_DISABLE=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 RAY_DISABLE_DOCKER_CPU_WARNING=1
export WANDB_MODE=offline
unset CHSPO_REWARD_VERSION CHSPO_OBJECTIVE_VERSION
exec "$VENV/bin/python" "$ROOT/scripts/train_rl_9b_whitebox.py" \
  "actor_rollout_ref.model.path=$SFT" \
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
  "actor_rollout_ref.model.lora_rank=0" \
  "actor_rollout_ref.model.enable_activation_offload=true" \
  "actor_rollout_ref.model.use_fused_kernels=$RL_FUSED_KERNELS" \
  "actor_rollout_ref.actor.strategy=fsdp2" \
  "actor_rollout_ref.actor.fsdp_config.offload_policy=true" \
  "actor_rollout_ref.actor.fsdp_config.model_dtype=fp32" \
  "actor_rollout_ref.ref.fsdp_config.model_dtype=bf16" \
  "actor_rollout_ref.actor.use_kl_loss=true" \
  "actor_rollout_ref.actor.kl_loss_coef=0.001" \
  "actor_rollout_ref.actor.ppo_mini_batch_size=4" \
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=$RL_ACTOR_SP_SIZE" \
  "actor_rollout_ref.actor.entropy_from_logits_with_chunking=true" \
  "actor_rollout_ref.actor.entropy_from_logits_chunk_size=128" \
  "actor_rollout_ref.actor.optim.override_optimizer_config={foreach:false}" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true" \
  "actor_rollout_ref.actor.fsdp_config.param_offload=$RL_ACTOR_PARAM_OFFLOAD" \
  "actor_rollout_ref.actor.fsdp_config.use_torch_compile=false" \
  "actor_rollout_ref.ref.fsdp_config.offload_policy=false" \
  "actor_rollout_ref.ref.fsdp_config.param_offload=false" \
  "actor_rollout_ref.ref.fsdp_config.optimizer_offload=false" \
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1" \
  "actor_rollout_ref.rollout.name=vllm" \
  "actor_rollout_ref.rollout.n=4" \
  "actor_rollout_ref.rollout.temperature=0.8" \
  "actor_rollout_ref.rollout.top_p=0.95" \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=2" \
  "actor_rollout_ref.rollout.max_model_len=$RL_MAX_MODEL_LEN" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.skip_mm_profiling=true" \
  "actor_rollout_ref.rollout.max_num_seqs=$RL_MAX_NUM_SEQS" \
  "actor_rollout_ref.rollout.gpu_memory_utilization=$RL_GPU_UTIL" \
  "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256" \
  "actor_rollout_ref.rollout.enforce_eager=true" \
  "actor_rollout_ref.rollout.calculate_log_probs=true" \
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" \
  "actor_rollout_ref.rollout.agent.default_agent_loop=whitebox_agentdojo_v1" \
  "actor_rollout_ref.rollout.agent.num_workers=$RL_AGENT_NUM_WORKERS" \
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=$LOOP" \
  "data.train_files=$DATA" \
  "data.val_files=$DATA" \
  "data.train_batch_size=$RL_TRAIN_BATCH_SIZE" \
  "data.max_prompt_length=$RL_PROMPT_LENGTH" \
  "data.max_response_length=$RL_RESPONSE_LENGTH" \
  "data.filter_overlong_prompts=false" \
  "data.shuffle=false" \
  "data.dataloader_num_workers=0" \
  "algorithm.adv_estimator=grpo" \
  "trainer.n_gpus_per_node=$RL_N_GPUS" \
  "trainer.nnodes=1" \
  "trainer.use_v1=false" \
  "trainer.total_epochs=$RL_ATTEMPT_EPOCHS" \
  "trainer.total_training_steps=$RL_STEPS" \
  "trainer.resume_mode=disable" \
  "trainer.val_before_train=false" \
  "trainer.test_freq=-1" \
  "trainer.critic_warmup=0" \
  "trainer.balance_batch=false" \
  "trainer.save_freq=$RL_SAVE_FREQ" \
  "trainer.max_actor_ckpt_to_keep=2" \
  "trainer.logger=[console]" \
  "trainer.project_name=cross_harness_rl" \
  "trainer.experiment_name=$(basename "$RUN")" \
  "trainer.default_local_dir=$RUN/checkpoints" \
  "+trainer.rl_algorithm=$RL_ALGORITHM" \
  "+trainer.separate_reference_gpu=$RL_SEPARATE_REFERENCE_GPU" \
  "+trainer.assignment_ledger=$LEDGER" \
  "+trainer.pilot_policy_snapshot=step:0" \
  "+trainer.pilot_metrics_path=$RUN/metrics.jsonl" \
  "reward.reward_model.enable=false" \
  "reward.num_workers=$RL_REWARD_NUM_WORKERS" \
  "actor_rollout_ref.actor.optim.lr=5e-7" \
  "actor_rollout_ref.actor.calculate_entropy=true" \
  "algorithm.use_kl_in_reward=false" \
  '+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="1"' \
  "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NET=Socket" \
  '+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE="1"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1"' \
  "+ray_kwargs.ray_init.object_store_memory=1073741824" \
  '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_REFERENCE_KEEP_ON_GPU="1"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_SYNC_WEIGHT_DTYPE="bf16"' \
  "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_REF_FUSED_TORCH=\"$VERL_REF_FUSED_TORCH\"" \
  '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_REF_FUSED_CHUNK_SIZE="128"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_CHUNKED_SCALED_LOGPROBS="1"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_QWEN_SCHEMA_STRING_PARAMS="1"' \
  "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_GENERATION_TIMEOUT_SECONDS=\"$VERL_GENERATION_TIMEOUT_SECONDS\"" \
  "ray_kwargs.ray_init.num_cpus=16" \
  "$@"
