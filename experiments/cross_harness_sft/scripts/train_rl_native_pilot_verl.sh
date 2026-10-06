#!/usr/bin/env bash
# Actual online one-step native VeRL GRPO (not a smoke test or teacher replay).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$(cd "$ROOT/../.." && pwd)"
VENV="$ROOT/.venv-rl-pilot"
RUN="${RL_RUN_DIR:-$ROOT/outputs/rl/native_pilot_online_20260930}"
RL_STEPS="${RL_STEPS:-1}"
RL_SAVE_FREQ="${RL_SAVE_FREQ:-1}"
RL_RESPONSE_LENGTH="${RL_RESPONSE_LENGTH:-2048}"
RL_ACTOR_SP_SIZE="${RL_ACTOR_SP_SIZE:-1}"
RL_ACTOR_PARAM_OFFLOAD="${RL_ACTOR_PARAM_OFFLOAD:-true}"
RL_FUSED_KERNELS="${RL_FUSED_KERNELS:-false}"
RL_GPU_UTIL="${RL_GPU_UTIL:-0.90}"
RL_MAX_NUM_SEQS="${RL_MAX_NUM_SEQS:-8}"
RL_MAX_MODEL_LEN="${RL_MAX_MODEL_LEN:-6144}"
RL_N_GPUS="${RL_N_GPUS:-2}"
RL_GPU_MAX_PREEXISTING_MIB="${RL_GPU_MAX_PREEXISTING_MIB:-256}"
SFT="$ROOT/checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267"
DATA="${RL_DATA:-$ROOT/outputs/rl/native_pilot_train_20260930_hermes_mcp.parquet}"
RL_TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE:-4}"
LOOP="$ROOT/configs/rl_native_pilot_agent_loop.yaml"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# The allocator's reserved blocks fragment across optimizer offload/onload
# cycles; the second update's backward OOMs on these 24 GB cards without
# expandable segments.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# This host's default NCCL network path segfaults even for a one-rank
# broadcast; the socket path passes the same minimal collective.
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
# GPU0/1 on this host do not support direct peer access; the two-rank
# collective passed with the shared-memory/socket fallback.
export NCCL_P2P_DISABLE=1
# Ray's one-GPU visibility masks trigger a peer-access failure for this host's
# two-rank NCCL broadcast. Keep both devices visible and let VeRL select rank.
IFS=',' read -r -a gpu_list <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#gpu_list[@]} != RL_N_GPUS )); then
  echo "RL_N_GPUS=$RL_N_GPUS does not match CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
for gpu in "${gpu_list[@]}"; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if (( used > RL_GPU_MAX_PREEXISTING_MIB )); then
    echo "GPU $gpu already holds ${used}MiB; refusing to preempt another job." >&2
    exit 2
  fi
done
ray_visible_device_overrides=()
if (( RL_N_GPUS > 1 )); then
  # This host's two-GPU NCCL setup requires Ray to leave the shared device
  # visibility mask intact. With one visible GPU, let Ray remap its physical
  # id (for example GPU 1) to local CUDA ordinal 0 in each worker.
  export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
  ray_visible_device_overrides+=( '+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1"' )
else
  unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
fi
[[ -e "$RUN/metrics.jsonl" ]] && { echo "Refusing to overwrite $RUN/metrics.jsonl" >&2; exit 2; }
export PYTHONPATH="$ROOT/vendor/verl:$ROOT/scripts:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export RAY_DISABLE_DOCKER_CPU_WARNING=1
exec "$VENV/bin/python" "$ROOT/scripts/train_rl_native_pilot_verl.py" \
  "actor_rollout_ref.model.path=$SFT" \
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
  "actor_rollout_ref.model.lora_rank=0" \
  "actor_rollout_ref.model.enable_activation_offload=true" \
  "actor_rollout_ref.model.use_fused_kernels=$RL_FUSED_KERNELS" \
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
  "actor_rollout_ref.ref.fsdp_config.param_offload=true" \
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1" \
  "actor_rollout_ref.rollout.name=vllm" \
  "actor_rollout_ref.rollout.n=4" \
  "actor_rollout_ref.rollout.temperature=0.8" \
  "actor_rollout_ref.rollout.top_p=0.95" \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
  "actor_rollout_ref.rollout.max_model_len=$RL_MAX_MODEL_LEN" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.skip_mm_profiling=true" \
  "actor_rollout_ref.rollout.max_num_seqs=$RL_MAX_NUM_SEQS" \
  "actor_rollout_ref.rollout.gpu_memory_utilization=$RL_GPU_UTIL" \
  "actor_rollout_ref.rollout.enforce_eager=true" \
  "actor_rollout_ref.rollout.calculate_log_probs=true" \
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" \
  "actor_rollout_ref.rollout.agent.default_agent_loop=native_agentdojo_pilot" \
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=$LOOP" \
  "data.train_files=$DATA" \
  "data.val_files=$DATA" \
  "data.train_batch_size=$RL_TRAIN_BATCH_SIZE" \
  "data.max_prompt_length=4096" \
  "data.max_response_length=$RL_RESPONSE_LENGTH" \
  "data.filter_overlong_prompts=false" \
  "data.shuffle=false" \
  "data.dataloader_num_workers=0" \
  "algorithm.adv_estimator=grpo" \
  "trainer.n_gpus_per_node=$RL_N_GPUS" \
  "trainer.nnodes=1" \
  "trainer.use_v1=false" \
  "trainer.total_epochs=$RL_STEPS" \
  "trainer.total_training_steps=$RL_STEPS" \
  "trainer.resume_mode=disable" \
  "trainer.val_before_train=false" \
  "trainer.test_freq=-1" \
  "trainer.critic_warmup=0" \
  "trainer.balance_batch=false" \
  "trainer.save_freq=$RL_SAVE_FREQ" \
  "trainer.logger=[console]" \
  "trainer.project_name=cross_harness_rl" \
  "trainer.experiment_name=$(basename "$RUN")" \
  "trainer.default_local_dir=$RUN/checkpoints" \
  "+trainer.pilot_policy_snapshot=step:0" \
  "+trainer.pilot_metrics_path=$RUN/metrics.jsonl" \
  "reward.reward_model.enable=false" \
  "algorithm.use_kl_in_reward=false" \
  '+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="1"' \
  "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NET=Socket" \
  '+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE="1"' \
  "${ray_visible_device_overrides[@]}" \
  "ray_kwargs.ray_init.num_cpus=16"
