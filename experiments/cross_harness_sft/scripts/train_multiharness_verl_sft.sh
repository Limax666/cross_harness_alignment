#!/usr/bin/env bash
# VeRL-native multi-turn, harness-balanced SFT. The caller selects the model
# and whether this is LoRA (LORA_RANK>0) or full-parameter SFT (LORA_RANK=0).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/cross_harness_sft"
SFT_PYTHON="${SFT_PYTHON:-/data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/python}"
SFT_TORCHRUN="${SFT_TORCHRUN:-/data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/torchrun}"
[[ -x "$SFT_PYTHON" && -x "$SFT_TORCHRUN" ]] || {
  echo "Missing VeRL training environment: $SFT_PYTHON / $SFT_TORCHRUN" >&2; exit 2;
}
GPUS="${GPUS:-0,1}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
WORLD_SIZE="${#GPU_LIST[@]}"
if [[ "$WORLD_SIZE" != "2" ]]; then
  echo "This resource profile requires exactly two GPUs; got GPUS=$GPUS." >&2
  exit 2
fi

# Install VeRL in the active environment, or point VERL_HOME to an official source checkout.
VERL_HOME="${VERL_HOME:-$EXP/vendor/verl}"
if [[ -f "$VERL_HOME/verl/trainer/sft_trainer.py" ]]; then
  export PYTHONPATH="$VERL_HOME${PYTHONPATH:+:$PYTHONPATH}"
fi
if ! "$SFT_PYTHON" -c 'import verl' >/dev/null 2>&1; then
  if [[ ! -f "$VERL_HOME/verl/trainer/sft_trainer.py" ]]; then
    cat >&2 <<EOF
VeRL is unavailable. Install the official repository in the active environment first:
  git clone https://github.com/verl-project/verl.git "$VERL_HOME"
  cd "$VERL_HOME" && pip install -e .
Or export VERL_HOME to an existing VeRL checkout.
EOF
    exit 2
  fi
  export PYTHONPATH="$VERL_HOME${PYTHONPATH:+:$PYTHONPATH}"
fi

# VeRL main permits Transformers <5.11, but Transformers 5.10.x imports the
# UE8M0 FP8 dtype at module import time. Torch 2.6 (the CUDA-12.6 stack on this
# server) does not provide that dtype. VeRL's own tested pin is 5.9.0.
"$SFT_PYTHON" - <<'PY'
import importlib.metadata as metadata
import sys
import torch
from packaging.version import Version

tf = Version(metadata.version("transformers"))
if not (Version("5.5.3") <= tf < Version("5.10.0")):
    sys.exit(
        f"Unsupported Transformers {tf} for this Torch {torch.__version__} VeRL run.\n"
        "Run: pip install --no-deps --force-reinstall 'transformers==5.9.0'"
    )
PY

MODEL="${MODEL:-Qwen/Qwen3.5-9B-Base}"
DATA_DIR="${DATA_DIR:-$EXP/data/sft/multi_harness_verl_v2}"
OUT_DIR="${OUT_DIR:-$EXP/checkpoints/qwen35-9b-base-multiharness-verl-sft-v2}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35-9b-multiharness-verl-sft}"
# 6144 tokens exhausted 24 GiB during the unfused full-vocabulary log-softmax.
# Use a smaller context budget; complete over-length trajectories are excluded.
MAX_LENGTH="${MAX_LENGTH:-4096}"
TRAIN_TRAJECTORIES_PER_HARNESS="${TRAIN_TRAJECTORIES_PER_HARNESS:-80}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
DATA_HARNESSES="${DATA_HARNESSES:-codex,claude_code,hermes,nanobot,qoder,claudecode,openagent,opencode,qwenpaw}"
MIN_QUALIFIED_PER_HARNESS="${MIN_QUALIFIED_PER_HARNESS:-1}"
MIN_QUALIFIED_VALIDATION="${MIN_QUALIFIED_VALIDATION:-1}"
TEST_FREQ="${TEST_FREQ:-32}"
# The previous run (2 epochs, 1e-4, rank 32) drove train loss near zero while
# validation loss stopped improving. Keep one pass over the expanded corpus
# and use a smaller adapter/update size as the safer default. Every setting is
# still overridable from the shell for controlled ablations.
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
LR="${LR:-5e-5}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WARMUP_RATIO="${WARMUP_RATIO:-0.10}"
SEED="${SEED:-42}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
# VeRL tokenizes individual turns then concatenates them. This role-segment
# template is paired with the dataset builder, which serializes tool calls into
# assistant content, and is required because Qwen3.5's native template only
# accepts a complete conversation containing a user turn.
VERL_SEGMENT_CHAT_TEMPLATE="$(PYTHONPATH="$EXP/scripts${PYTHONPATH:+:$PYTHONPATH}" "$SFT_PYTHON" -c 'from build_multiharness_verl_sft import VERL_SEGMENT_CHAT_TEMPLATE; print(VERL_SEGMENT_CHAT_TEMPLATE)')"
export VERL_SEGMENT_CHAT_TEMPLATE

if [[ -n "${TRAIN_JSONL:-}" && -n "${VAL_JSONL:-}" ]]; then
  [[ -s "$TRAIN_JSONL" && -s "$VAL_JSONL" ]] || {
    echo "Curated train/validation JSONL must both be nonempty." >&2; exit 2;
  }
else
  # Historical 9B LoRA baseline only. New full-SFT runs supply curated input.
  "$SFT_PYTHON" "$EXP/scripts/merge_multiharness_actbench_sft.py"
  TRAIN_JSONL="$EXP/data/sft/multi_harness_combined_v1/train.jsonl"
  VAL_JSONL="$EXP/data/sft/multi_harness_combined_v1/validation.jsonl"
fi

# Build a static full training view. This executes the exact tokenizer template
# preflight, rejects over-length full trajectories, and writes immutable audits.
"$SFT_PYTHON" "$EXP/scripts/build_multiharness_verl_sft.py" \
  --train-jsonl "$TRAIN_JSONL" \
  --validation-jsonl "$VAL_JSONL" \
  --output-dir "$DATA_DIR" --model "$MODEL" --max-length "$MAX_LENGTH" \
  --train-trajectories-per-harness "$TRAIN_TRAJECTORIES_PER_HARNESS" \
  --harnesses "$DATA_HARNESSES" --min-qualified-per-harness "$MIN_QUALIFIED_PER_HARNESS" \
  --min-qualified-validation "$MIN_QUALIFIED_VALIDATION" --seed "$SEED"

mkdir -p "$OUT_DIR"
# Qwen3.5 processor has no enable_thinking kwarg. Visible <think> text remains
# in messages; this disables only the unsupported template option.
"$SFT_TORCHRUN" --standalone --nnodes=1 --nproc_per_node="$WORLD_SIZE" \
  -m verl.trainer.sft_trainer \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/validation.parquet" \
  data.messages_key=messages data.tools_key=tools \
  data.enable_thinking_key=__unused_enable_thinking__ data.enable_thinking_default=null \
  data.pad_mode=no_padding data.truncation=error \
  data.max_length="$MAX_LENGTH" data.max_token_len_per_gpu="$MAX_LENGTH" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" data.micro_batch_size_per_gpu=1 data.use_dynamic_bsz=False \
  data.num_workers=0 data.ignore_input_ids_mismatch=False \
  engine=fsdp engine.strategy=fsdp engine.fsdp_size=-1 \
  engine.ulysses_sequence_parallel_size=1 engine.dtype=bfloat16 \
  engine.model_dtype=bfloat16 engine.reshard_after_forward=True \
  model.path="$MODEL" model.custom_chat_template='${oc.env:VERL_SEGMENT_CHAT_TEMPLATE}' model.use_remove_padding=True \
  +model.override_config.attn_implementation=sdpa \
  model.enable_gradient_checkpointing=True model.enable_activation_offload=False \
  model.lora_rank="$LORA_RANK" model.lora_alpha="$LORA_ALPHA" model.target_modules=all-linear \
  optim.optimizer=AdamW optim.optimizer_impl=torch.optim optim.lr="$LR" \
  optim.lr_scheduler_type=cosine optim.lr_warmup_steps_ratio="$WARMUP_RATIO" \
  optim.weight_decay="$WEIGHT_DECAY" optim.clip_grad=1.0 \
  trainer.default_local_dir="$OUT_DIR" trainer.project_name=cross-harness-sft \
  trainer.experiment_name="$EXPERIMENT_NAME" trainer.logger='["console","wandb"]' \
  trainer.n_gpus_per_node="$WORLD_SIZE" trainer.total_epochs="$TOTAL_EPOCHS" trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.test_freq="$TEST_FREQ" trainer.save_freq=after_each_epoch trainer.max_ckpt_to_keep=2 \
  trainer.resume_mode=disable trainer.seed="$SEED" \
  "$@"
