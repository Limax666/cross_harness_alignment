"""Real QLoRA SFT for verified multi-turn agent trajectories.

Adapted from chapter8/cot-distillation: CUDA is mandatory, there is no mock
success path, and the manifest binds the checkpoint to the exact dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
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
        # Qwen3.5 rejects a system-only conversation.  It is still a valid
        # prefix in our trajectory, so defer rendering until its user turn.
        before = chat_ids(tokenizer, prefix, tools) if any(x.get("role") == "user" for x in prefix) else []
        prefix.append(message)
        after = chat_ids(tokenizer, prefix, tools) if any(x.get("role") == "user" for x in prefix) else []
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


def compact_tool_schema(value: Any) -> Any:
    """Deterministically reduce every candidate tool, without looking at targets."""
    if isinstance(value, dict):
        return {key: compact_tool_schema(item) for key, item in value.items()
                if key not in {"description", "title", "examples", "default"}}
    if isinstance(value, list):
        return [compact_tool_schema(item) for item in value]
    return value


def save_training_curves(log_history: list[dict[str, Any]], output_dir: Path) -> None:
    """Persist raw Trainer metrics and render loss curves for experiment review."""
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(log_history, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("install matplotlib to write training_curves.png") from exc

    train = [(float(item["step"]), float(item["loss"])) for item in log_history
             if isinstance(item.get("step"), (int, float)) and isinstance(item.get("loss"), (int, float))]
    evaluation = [(float(item["step"]), float(item["eval_loss"])) for item in log_history
                  if isinstance(item.get("step"), (int, float)) and isinstance(item.get("eval_loss"), (int, float))]
    figure, axis = plt.subplots(figsize=(10, 5.5), layout="constrained")
    if train:
        steps, losses = zip(*train)
        axis.plot(steps, losses, alpha=.35, linewidth=1, label="train loss (step)")
        window = min(10, len(losses))
        smoothed = [sum(losses[max(0, index - window + 1):index + 1]) / min(window, index + 1)
                    for index in range(len(losses))]
        axis.plot(steps, smoothed, linewidth=2, label=f"train loss (moving mean, {window} steps)")
    if evaluation:
        steps, losses = zip(*evaluation)
        axis.plot(steps, losses, "o-", linewidth=2, markersize=6, label="validation loss")
    axis.set(title="Qwen3.5-9B Cross-Harness SFT", xlabel="global step", ylabel="cross-entropy loss")
    axis.grid(alpha=.25)
    if train or evaluation:
        axis.legend()
    figure.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(figure)


def encode_fitting_example(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]], max_length: int) -> Encoded | None:
    """Keep the newest trajectory context while guaranteeing assistant supervision."""
    for candidate_tools in (tools, compact_tool_schema(tools)):
        candidate_messages = list(messages)
        while True:
            try:
                return encode_assistant_only(tokenizer, candidate_messages, candidate_tools, max_length)
            except ValueError:
                # Keep system, user, and the final answer; discard the oldest
                # complete assistant/tool exchange when it is too long.
                if len(candidate_messages) <= 3:
                    break
                del candidate_messages[2]
                while len(candidate_messages) > 2 and candidate_messages[2].get("role") == "tool":
                    del candidate_messages[2]
    return None


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
    p.add_argument("--base-model", default="Qwen/Qwen3.5-9B-Base")
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
    # Qwen can use Hub kernels when installed, but they are optional and have
    # a tightly pinned `kernels` package dependency.  Keep real SFT portable.
    p.add_argument("--use-kernels", action=argparse.BooleanOptionalAction, default=False)
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
    def encode_split(items: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]], name: str) -> list[Encoded]:
        result: list[Encoded] = []
        dropped = 0
        for messages, tools in items:
            row = encode_fitting_example(tokenizer, messages, tools, args.max_length)
            if row is None:
                dropped += 1
            else:
                result.append(row)
        if not result:
            raise ValueError(f"{name}: no examples retain assistant supervision at max_length={args.max_length}")
        print(json.dumps({"split": name, "input": len(items), "encoded": len(result), "dropped_overlong": dropped}))
        return result

    train_rows = encode_split(train_messages, "train")
    validation_rows = encode_split(validation_messages, "validation") if validation_messages else []

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

    class AssistantOnlyTrainer(Trainer):
        """Avoid materialising [sequence_length, vocabulary] logits for masked tokens."""
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False,
                         num_items_in_batch: Any = None) -> Any:
            labels = inputs.pop("labels")
            if labels.shape[0] != 1:
                raise ValueError("sparse-logit trainer requires per-device batch-size 1")
            # A causal-LM logit at position i predicts the label at i + 1.
            positions = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).squeeze(-1)
            if positions.numel() == 0:
                raise ValueError("batch has no supervised next-token labels")
            outputs = model(**inputs, logits_to_keep=positions)
            targets = labels[0, 1:].index_select(0, positions).to(outputs.logits.device)
            loss = torch.nn.functional.cross_entropy(outputs.logits.float().view(-1, outputs.logits.shape[-1]), targets)
            return (loss, outputs) if return_outputs else loss

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs: dict[str, Any] = {"dtype": dtype, "trust_remote_code": args.trust_remote_code,
                              "use_kernels": args.use_kernels}
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.load_in_4bit:
        # Each torchrun process owns exactly one quantized model replica.
        # Mapping every rank to GPU 0 silently corrupts multi-GPU QLoRA runs.
        kwargs.update(device_map={"": local_rank}, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs); model.config.use_cache = False
    if args.load_in_4bit: model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    elif args.gradient_checkpointing: model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                                             bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = TrainingArguments(output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation, logging_steps=1,
        save_strategy="epoch", eval_strategy="epoch" if validation_rows else "no", seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(), report_to="none",
        remove_unused_columns=False, ddp_find_unused_parameters=False)
    trainer = AssistantOnlyTrainer(model=model, args=training, train_dataset=Rows(train_rows),
                                   eval_dataset=Rows(validation_rows) if validation_rows else None, data_collator=collate)
    outcome = trainer.train(); trainer.save_model(str(args.output_dir))
    if int(os.environ.get("RANK", "0")) != 0:
        return
    tokenizer.save_pretrained(str(args.output_dir))
    save_training_curves(trainer.state.log_history, args.output_dir)
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
