#!/usr/bin/env bash
# Train the immutable full v7 Qwen3.5-2B qualified set after runtime preflight.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/cross_harness_sft"
PYTHON="${SFT_PYTHON:-/data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/python}"
TORCHRUN="${SFT_TORCHRUN:-/data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/torchrun}"
DATA="$EXP/data/sft/teacher_release_v7/verl_qwen35_2b_base_bs16_full_v2"
VIEW="$EXP/data/sft/teacher_release_v7/full_view_qwen35_2b_base_bs16_v2"
PREFLIGHT="$EXP/data/sft/teacher_release_v7/preflight_qwen35_2b_base_4096_v1"
GPUS="${GPUS:-1,2}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
[[ "${#GPU_LIST[@]}" == 2 ]] || { echo "Exactly two GPUs required" >&2; exit 2; }
[[ -x "$PYTHON" && -x "$TORCHRUN" ]] || { echo "Missing SFT runtime" >&2; exit 2; }
VERL_SAMPLER_PATCH="$EXP/patches/verl_sft_sampler_shuffle.patch"
if ! git -C "$ROOT" apply --reverse --check "$VERL_SAMPLER_PATCH" >/dev/null 2>&1; then
  git -C "$ROOT" apply --check "$VERL_SAMPLER_PATCH"
  git -C "$ROOT" apply "$VERL_SAMPLER_PATCH"
fi
export PYTHONPATH="$EXP/vendor/verl${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" - "$DATA" "$VIEW" "$PREFLIGHT" <<'PY'
import collections,hashlib,json,sys
from pathlib import Path
data,view,preflight=map(Path,sys.argv[1:])
m=json.loads((data/'materialization_manifest.json').read_text())
v=json.loads((view/'sampling_manifest.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert m['sampling_manifest_sha256']==sha(view/'sampling_manifest.json')
assert m['model']=='Qwen/Qwen3.5-2B-Base' and m['max_length']==4096
assert m['expected_global_batch']==16 and m['sampler_shuffle_required'] is False
assert v['format']=='complete_teacher_full_preflight_view_v1'
assert v['source_preflight_manifest_sha256']==sha(preflight/'preflight_manifest.json')
assert v['qualified_train_sha256']==sha(preflight/'qualified_train.jsonl')
assert v['qualified_validation_sha256']==sha(preflight/'qualified_validation.jsonl')
assert v['unique_train_rows']==1421 and v['train_rows']==1424
assert v['padding_duplicates']==3 and len(v['padding_record_ids'])==3
assert v['unique_validation_rows']==232 and v['validation_rows']==240
assert v['validation_padding_duplicates']==8 and len(v['validation_padding_record_ids'])==8
assert v['optimizer_batches_per_epoch']==89
assert v['no_replacement_except_explicit_padding']
assert m['outputs']['train']['rows']==1424 and m['outputs']['validation']['rows']==240
source_ids={json.loads(line)['record_id'] for line in (preflight/'qualified_train.jsonl').open()}
view_ids=collections.Counter(json.loads(line)['record_id'] for line in (view/'train_balanced.jsonl').open())
assert len(source_ids)==1421 and set(view_ids)==source_ids
assert sorted(record_id for record_id,count in view_ids.items() if count==2)==sorted(v['padding_record_ids'])
assert all(count in (1,2) for count in view_ids.values())
val_source_ids={json.loads(line)['record_id'] for line in (preflight/'qualified_validation.jsonl').open()}
val_view_ids=collections.Counter(json.loads(line)['record_id'] for line in (view/'validation_natural.jsonl').open())
assert len(val_source_ids)==232 and set(val_view_ids)==val_source_ids
assert sorted(record_id for record_id,count in val_view_ids.items() if count==2)==sorted(v['validation_padding_record_ids'])
assert all(count in (1,2) for count in val_view_ids.values())
for line in (view/'batch_audit.jsonl').open():
    batch=json.loads(line)
    assert batch['rows']==16 and batch['benign']>0 and batch['risk']>0
for split in ('train','validation'):
    assert m['outputs'][split]['sha256']==sha(data/f'{split}.parquet')
    assert m['outputs'][split]['runtime_preflight']=='passed'
print('Full 1,421-unique-row v7 SFT data gate passed')
PY
if [[ "${CHSFT_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export CHSFT_REQUIRE_FULL_TRAINABLE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VERL_SEGMENT_CHAT_TEMPLATE="$(PYTHONPATH="$EXP/scripts${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c 'from build_multiharness_verl_sft import VERL_SEGMENT_CHAT_TEMPLATE; print(VERL_SEGMENT_CHAT_TEMPLATE)')"
OUT_DIR="${OUT_DIR:-$EXP/checkpoints/qwen35-2b-base-teacher-v7-all1421-full-sft}"
mkdir -p "$OUT_DIR"
exec "$TORCHRUN" --standalone --nnodes=1 --nproc_per_node=2 \
  -m verl.trainer.sft_trainer \
  data.train_files="$DATA/train.parquet" data.val_files="$DATA/validation.parquet" \
  data.messages_key=messages data.tools_key=tools \
  data.enable_thinking_key=__unused_enable_thinking__ data.enable_thinking_default=null \
  data.pad_mode=no_padding data.truncation=error data.max_length=4096 \
  data.max_token_len_per_gpu=4096 data.train_batch_size=16 data.micro_batch_size_per_gpu=1 \
  data.use_dynamic_bsz=False data.num_workers=0 data.ignore_input_ids_mismatch=False \
  +data.sampler_shuffle=false \
  engine=fsdp engine.strategy=fsdp engine.fsdp_size=-1 \
  engine.ulysses_sequence_parallel_size=1 engine.dtype=bfloat16 engine.model_dtype=bfloat16 \
  engine.reshard_after_forward=True \
  model.path=Qwen/Qwen3.5-2B-Base model.custom_chat_template='${oc.env:VERL_SEGMENT_CHAT_TEMPLATE}' \
  model.use_remove_padding=True +model.override_config.attn_implementation=sdpa \
  model.enable_gradient_checkpointing=True model.enable_activation_offload=False \
  model.lora_rank=0 model.lora_alpha=0 \
  optim.optimizer=AdamW optim.optimizer_impl=torch.optim optim.lr="${LR:-1e-5}" \
  optim.lr_scheduler_type=cosine optim.lr_warmup_steps_ratio=0.10 \
  optim.weight_decay=0.05 optim.clip_grad=1.0 \
  trainer.default_local_dir="$OUT_DIR" trainer.project_name=cross-harness-sft \
  trainer.experiment_name=qwen35-2b-base-teacher-v7-all1421-full-sft \
  trainer.logger='["console","wandb"]' trainer.n_gpus_per_node=2 \
  trainer.total_epochs="${TOTAL_EPOCHS:-3}" trainer.total_training_steps=null trainer.test_freq=8 \
  trainer.save_freq=after_each_epoch trainer.max_ckpt_to_keep=2 \
  trainer.resume_mode=disable trainer.seed=42 "$@"
