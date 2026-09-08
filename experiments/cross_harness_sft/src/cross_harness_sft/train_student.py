"""Real QLoRA SFT for verified multi-turn agent trajectories.

Adapted from chapter8/cot-distillation: CUDA is mandatory, there is no mock
success path, and the manifest binds the checkpoint to the exact dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def chat_ids(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]], generation: bool = False) -> list[int]:
    value = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=generation)
    if isinstance(value, dict) or hasattr(value, "keys"): value = value["input_ids"]
    if hasattr(value, "tolist"): value = value.tolist()
    if value and isinstance(value[0], list): value = value[0]
    if not isinstance(value, list): raise TypeError("chat template did not produce input_ids")
    return value


@dataclass
class Encoded:
    input_ids: list[int]
    labels: list[int]


def encode_assistant_only(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]], max_length: int) -> Encoded:
    """Incrementally render the native template and label assistant deltas only."""
    ids: list[int] = []
    labels: list[int] = []
    prefix: list[dict[str, Any]] = []
    for message in messages:
        before = chat_ids(tokenizer, prefix, tools) if prefix else []
        prefix.append(message)
        after = chat_ids(tokenizer, prefix, tools)
        common = 0
        for left, right in zip(before, after):
            if left != right: break
            common += 1
        # Some templates rewrite an end marker when another message is appended.
        if common < len(ids):
            ids = after[:common]
            labels = labels[:common]
        delta = after[common:]
        ids.extend(delta)
        supervise = message.get("role") == "assistant"
        labels.extend(delta if supervise else [-100] * len(delta))
    ids, labels = ids[:max_length], labels[:max_length]
    if not ids or not any(value != -100 for value in labels):
        raise ValueError("empty example or max_length removed every assistant token")
    return Encoded(ids, labels)


def load(path: Path) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    result: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip(): continue
            row = json.loads(line); messages = row.get("messages")
            if not isinstance(messages, list) or not messages: raise ValueError(f"{path}:{number}: messages missing")
            roles = [str(x.get("role")) for x in messages]
            if roles[0] != "system" or "user" not in roles or roles[-1] != "assistant":
                raise ValueError(f"{path}:{number}: invalid multi-turn role structure")
            final = str(messages[-1].get("content") or "")
            if "<think>" not in final or "</think>" not in final: raise ValueError(f"{path}:{number}: final target lacks think tags")
            tools = row.get("tools")
            if not isinstance(tools, list) or not tools: raise ValueError(f"{path}:{number}: tool registry missing")
            result.append((messages, tools))
    if not result: raise ValueError("no verified training samples")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", required=True, type=Path)
    p.add_argument("--validation-data", type=Path)
    p.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=16)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args(); train_messages = load(args.train_data)
    validation_messages = load(args.validation_data) if args.validation_data else []
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except (ImportError, RuntimeError) as exc:
        raise SystemExit(f"training stack unavailable: {type(exc).__name__}: {exc}") from exc
    if not torch.cuda.is_available(): raise SystemExit("real SFT requires CUDA; no CPU/mock fallback")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    train_rows = [encode_assistant_only(tokenizer, messages, tools, args.max_length) for messages, tools in train_messages]
    validation_rows = [encode_assistant_only(tokenizer, messages, tools, args.max_length) for messages, tools in validation_messages]

    class Rows(Dataset):
        def __init__(self, rows: list[Encoded]): self.rows = rows
        def __len__(self) -> int: return len(self.rows)
        def __getitem__(self, index: int) -> dict[str, list[int]]:
            row = self.rows[index]; return {"input_ids": row.input_ids, "labels": row.labels}

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        width = max(len(x["input_ids"]) for x in batch); ids, masks, labels = [], [], []
        for row in batch:
            pad = width - len(row["input_ids"]); ids.append(row["input_ids"] + [tokenizer.pad_token_id] * pad)
            masks.append([1] * len(row["input_ids"]) + [0] * pad); labels.append(row["labels"] + [-100] * pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks), "labels": torch.tensor(labels)}

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": args.trust_remote_code}
    if args.load_in_4bit:
        kwargs.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs); model.config.use_cache = False
    if args.load_in_4bit: model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    elif args.gradient_checkpointing: model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                                             bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = TrainingArguments(output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation, logging_steps=1,
        save_strategy="epoch", eval_strategy="epoch" if validation_rows else "no", seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(), report_to="none",
        remove_unused_columns=False)
    trainer = Trainer(model=model, args=training, train_dataset=Rows(train_rows),
                      eval_dataset=Rows(validation_rows) if validation_rows else None, data_collator=collate)
    outcome = trainer.train(); trainer.save_model(str(args.output_dir)); tokenizer.save_pretrained(str(args.output_dir))
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate", "peft", "bitsandbytes")}
    manifest = {"schema": "cross_harness.sft_manifest.v2", "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(), "base_model": args.base_model,
        "dataset": {"train": str(args.train_data.resolve()), "train_sha256": sha256(args.train_data),
                    "train_samples": len(train_rows), "validation": str(args.validation_data.resolve()) if args.validation_data else None,
                    "validation_sha256": sha256(args.validation_data) if args.validation_data else None,
                    "validation_samples": len(validation_rows)}, "host": {"platform": platform.platform(),
                    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
        "dependencies": versions, "training": vars(args) | {"metrics": outcome.metrics}}
    (args.output_dir / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
