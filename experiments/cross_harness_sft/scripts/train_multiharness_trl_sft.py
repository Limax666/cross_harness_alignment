#!/usr/bin/env python3
"""Harness-balanced, harness-conditioned QLoRA SFT using TRL's SFTTrainer.

This is intentionally a thin orchestration layer around TRL.  It does not implement a
custom loss or training loop.  It makes the experimental contracts executable:

* every prompt starts with the model-visible harness_context produced by the filter;
* every retained row fits the selected max_length after the *actual* Qwen chat template
  and tool schemas are rendered;
* samples are drawn with equal expected mass for each harness, not proportional to the
  number of collected trajectories; and
* the run writes data, length, balance, package, and metric artifacts next to the adapter.

Examples are never silently truncated.  Rows longer than max_length are excluded from
this run and recorded in `data_length_audit.jsonl`; segment them in the dataset builder
when their final safety decision cannot otherwise be retained.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from torch.utils.data import WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, set_seed

try:
    from trl import SFTConfig, SFTTrainer
except ImportError as exc:  # give an actionable error before a costly model download
    raise SystemExit(
        "TRL is required. Activate cross-harness-sft and install the pinned training "
        "stack, e.g. pip install 'trl[peft]' transformers datasets accelerate bitsandbytes."
    ) from exc


PROJECT = Path(__file__).resolve().parents[3]
HARNESS_ORDER = ("codex", "claude_code", "hermes", "nanobot", "qoder")


def parse_args() -> argparse.Namespace:
    exp = PROJECT / "experiments/cross_harness_sft"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-data", type=Path, default=exp / "data/sft/multi_harness_trl_v1/train.jsonl")
    parser.add_argument("--validation-data", type=Path, default=exp / "data/sft/multi_harness_trl_v1/validation.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--output-dir", type=Path, default=exp / "checkpoints/qwen35-9b-base-multiharness-trl-sft")
    parser.add_argument("--max-length", type=int, default=8192)
    # Prompt-completion conversion yields ~3k assistant-action examples. One balanced
    # pass is about 380 optimizer updates on two GPUs with grad_accum=4.
    parser.add_argument("--epochs", type=float, default=1.0)
    # TRL recommends a roughly 10x larger LR for LoRA than full SFT (2e-4).
    # Use the conservative midpoint for this small, safety-sensitive 9B corpus.
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=400,
                        help="Fixed optimizer-update budget; overrides epoch-derived duration when positive.")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def render_token_ids(
    tokenizer: Any, row: dict[str, Any], conversation: list[dict[str, Any]], add_generation_prompt: bool = False
) -> list[int]:
    kwargs: dict[str, Any] = {
        "conversation": conversation,
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        # Do not inject an unobserved Qwen thinking prefix between prompt and target.
        # Recorded visible <think> content, when present, remains in the completion.
        "enable_thinking": False,
    }
    # AgentDojo rows have callable tool schemas. HarnessRisk rows deliberately have an
    # empty list; omit it to retain Qwen's ordinary conversational rendering.
    if row.get("tools"):
        kwargs["tools"] = row["tools"]
    ids = tokenizer.apply_chat_template(**kwargs)
    # Qwen3.5 with the installed Transformers version returns a BatchEncoding even
    # when `tokenize=True` and no tensor return type was requested.
    if hasattr(ids, "get") and ids.get("input_ids") is not None:
        ids = ids["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list) or not ids or not all(isinstance(token, int) for token in ids):
        raise ValueError(f"chat template did not return a token-id list for {row_id(row)}")
    return ids


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("metadata", {}).get("record_id", "<unknown-record>"))


def valid_harness(row: dict[str, Any]) -> str:
    name = row.get("metadata", {}).get("harness_name")
    if name not in HARNESS_ORDER:
        raise ValueError(f"{row_id(row)} has invalid harness_name: {name!r}")
    context = row.get("metadata", {}).get("harness_context", "")
    first = (row.get("messages") or [{}])[0]
    if first.get("role") != "system" or context not in str(first.get("content", "")):
        raise ValueError(f"{row_id(row)} does not expose its harness_context as the first system message")
    return name


def prepare_split(
    rows: list[dict[str, Any]], tokenizer: Any, max_length: int, split: str
) -> tuple[Dataset, list[dict[str, Any]]]:
    retained: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for row in rows:
        harness = valid_harness(row)
        messages = row["messages"]
        # Qwen3.5's current template lacks `{% generation %}`, so TRL cannot derive
        # reliable assistant masks from a full conversation. Use TRL's documented
        # prompt-completion route instead: each recorded assistant action is the only
        # completion target, while all preceding system/user/tool turns are context.
        for turn_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            prompt, completion = messages[:turn_index], [message]
            event = {
                "record_id": row_id(row), "split": split, "harness_name": harness,
                "assistant_turn_index": turn_index, "max_length": max_length,
            }
            if not prompt:
                event.update({"accepted_for_run": False, "reason": "assistant_without_context"})
                audit.append(event)
                continue
            full_ids = render_token_ids(tokenizer, row, prompt + completion)
            prompt_ids = render_token_ids(tokenizer, row, prompt, add_generation_prompt=True)
            event["rendered_tokens"] = len(full_ids)
            event["completion_tokens_estimate"] = max(0, len(full_ids) - len(prompt_ids))
            if len(full_ids) > max_length:
                event.update({"accepted_for_run": False, "reason": "over_length_unsegmentable"})
                audit.append(event)
                continue
            if full_ids[:len(prompt_ids)] != prompt_ids:
                event.update({"accepted_for_run": False, "reason": "template_boundary_mismatch"})
                audit.append(event)
                continue
            if event["completion_tokens_estimate"] == 0:
                event.update({"accepted_for_run": False, "reason": "empty_completion_after_template"})
                audit.append(event)
                continue
            event.update({"accepted_for_run": True})
            audit.append(event)
            # `prompt` / `completion` is TRL's conversational prompt-completion schema.
            # SFTTrainer builds the completion-only labels; this script does not.
            retained.append({
                "prompt": prompt, "completion": completion, "tools": row.get("tools") or [],
                "harness_name": harness, "record_id": row_id(row), "assistant_turn_index": turn_index,
                "chat_template_kwargs": {"enable_thinking": False},
            })
    if not retained:
        raise ValueError(f"All {split} rows exceed max_length={max_length}")
    # A multi-turn trajectory becomes several next-action supervision examples. Give
    # every source trajectory equal total mass within its harness, then give harnesses
    # equal total mass. This retains every turn without letting long episodes dominate.
    turns_per_trajectory = Counter((item["harness_name"], item["record_id"]) for item in retained)
    trajectories_per_harness: dict[str, set[str]] = {}
    for harness, record_id in turns_per_trajectory:
        trajectories_per_harness.setdefault(harness, set()).add(record_id)
    for item in retained:
        key = (item["harness_name"], item["record_id"])
        item["sampling_weight"] = 1.0 / (
            len(trajectories_per_harness[item["harness_name"]]) * turns_per_trajectory[key]
        )
    return Dataset.from_list(retained), audit


class HarnessBalancedSFTTrainer(SFTTrainer):
    """TRL trainer with an equal-harness expected-mass sampler.

    The underlying SFT loss, tokenization, distributed training and optimizer remain
    TRL/Transformers implementations. Replacement sampling is intentional: it prevents
    Codex/Claude collection volume from dominating smaller harnesses in an epoch.
    """

    def __init__(self, *args: Any, sampling_weights: list[float], **kwargs: Any) -> None:
        # SFTTrainer removes auxiliary columns before it invokes the sampler. Keep a
        # row-aligned sidecar from the already audited prompt-completion dataset.
        self._sampling_weights = list(sampling_weights)
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset: Dataset | None = None):  # type: ignore[override]
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None
        if len(self._sampling_weights) != len(dataset):
            raise RuntimeError(
                "TRL changed the number of rows after prompt-completion preprocessing "
                f"({len(self._sampling_weights)} -> {len(dataset)}); refusing misaligned balanced sampling"
            )
        weights = torch.as_tensor(self._sampling_weights, dtype=torch.double)
        # The Accelerate dataloader shard distributes batches over ranks. A shared seeded
        # generator makes this expected mixture reproducible for the complete job.
        generator = torch.Generator()
        generator.manual_seed(int(self.args.seed))
        return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=generator)


class JsonHistoryCallback(TrainerCallback):
    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.history: list[dict[str, Any]] = []

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any):
        if state.is_world_process_zero and logs:
            self.history.append({"step": state.global_step, "epoch": state.epoch, **logs})
            self.destination.write_text(json.dumps(self.history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return control


def save_curve(history_path: Path, output_path: Path) -> None:
    """Draw train and validation loss with separate axes so low validation NLL is visible."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    train = [(item["step"], item["loss"]) for item in history if "loss" in item]
    valid = [(item["step"], item["eval_loss"]) for item in history if "eval_loss" in item]
    if not train and not valid:
        return
    fig, ax_train = plt.subplots(figsize=(10, 6))
    if train:
        ax_train.plot(*zip(*train), color="#377eb8", label="train loss")
        ax_train.set_ylabel("train cross-entropy", color="#377eb8")
    ax_train.set_xlabel("global step")
    if valid:
        ax_valid = ax_train.twinx()
        ax_valid.plot(*zip(*valid), "o-", color="#4daf4a", label="validation loss")
        ax_valid.set_ylabel("validation cross-entropy", color="#4daf4a")
    fig.suptitle("Harness-balanced TRL SFT")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def package_versions() -> dict[str, str]:
    packages = ("trl", "transformers", "datasets", "accelerate", "peft", "bitsandbytes", "torch")
    return {name: importlib.metadata.version(name) for name in packages}


def main() -> None:
    args = parse_args()
    if args.max_length < 1024:
        raise ValueError("max_length below 1024 is prohibited for agent trajectories")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QLoRA run")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 2:
        raise ValueError(f"At most two GPUs are permitted; got WORLD_SIZE={world_size}")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError("CUDA_VISIBLE_DEVICES exposes fewer devices than torchrun ranks")
    torch.cuda.set_device(local_rank)
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_set, train_audit = prepare_split(read_jsonl(args.train_data), tokenizer, args.max_length, "train")
    validation_set, validation_audit = prepare_split(read_jsonl(args.validation_data), tokenizer, args.max_length, "validation")

    # All ranks write the same deterministic audit; rank zero owns durable artifacts.
    if local_rank == 0:
        (args.output_dir / "data_length_audit.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train_audit + validation_audit),
            encoding="utf-8",
        )
        balance = {name: int(Counter(train_set["harness_name"])[name]) for name in HARNESS_ORDER}
        print(json.dumps({
            "supervision": "TRL conversational prompt-completion / completion_only_loss",
            "train_assistant_targets_retained": len(train_set),
            "validation_assistant_targets_retained": len(validation_set),
            "train_targets_by_harness_before_sampling": balance,
        }, ensure_ascii=False))
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(), "base_model": args.base_model,
            "max_length": args.max_length, "world_size": world_size, "train_rows_retained": len(train_set),
            "validation_rows_retained": len(validation_set), "train_rows_by_harness_before_sampling": balance,
            "sampling_policy": "WeightedRandomSampler: equal expected mass per harness and per source trajectory within harness, replacement=True",
            "supervision": "TRL conversational prompt-completion, completion_only_loss=True",
            "packing": False, "eval_strategy": "steps",
            "eval_steps": args.eval_steps, "max_steps": args.max_steps, "data": {
                "train": str(args.train_data), "validation": str(args.validation_data),
            }, "package_versions": package_versions(),
        }
        (args.output_dir / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=quantization, torch_dtype=torch.bfloat16,
        device_map={"": local_rank}, trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
    )
    training_args = SFTConfig(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate, per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True, tf32=True, max_length=args.max_length, packing=False,
        # This is assistant-only supervision by construction. Do not set
        # assistant_only_loss=True: the Qwen3.5 template lacks generation masks.
        assistant_only_loss=False, completion_only_loss=True, shuffle_dataset=False,
        dataset_num_proc=1, dataset_kwargs={"skip_prepare_dataset": False},
        logging_strategy="steps", logging_steps=args.logging_steps,
        # About 12 validation points for the default ~380-update job, rather than two.
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="epoch", save_total_limit=args.save_total_limit,
        report_to="none", remove_unused_columns=True, seed=args.seed,
    )
    history_callback = JsonHistoryCallback(args.output_dir / "training_history.json")
    trainer = HarnessBalancedSFTTrainer(
        model=model, args=training_args, train_dataset=train_set, eval_dataset=validation_set,
        processing_class=tokenizer, peft_config=peft_config, callbacks=[history_callback],
        sampling_weights=list(train_set["sampling_weight"]),
    )
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    # Evaluation is collective under DDP: every rank participates, then rank zero writes.
    metrics = trainer.evaluate(metric_key_prefix="final_validation")
    if trainer.is_world_process_zero():
        (args.output_dir / "final_validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        save_curve(args.output_dir / "training_history.json", args.output_dir / "training_curves.png")


if __name__ == "__main__":
    main()
