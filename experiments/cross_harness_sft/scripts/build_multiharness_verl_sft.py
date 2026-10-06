#!/usr/bin/env python3
"""Build the audited, harness-balanced VeRL multi-turn SFT Parquet dataset.

This is a *data-view builder*, not a trainer.  It preserves each accepted source
trajectory as one complete multi-turn conversation.  VeRL's
``MultiTurnSFTDataset`` then masks every assistant turn automatically.  The only
resampling is at trajectory level: every harness has equal probability mass, while
real and accepted synthetic trajectories are sampled identically.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[3]
EXP = PROJECT / "experiments/cross_harness_sft"
HARNESS_ORDER = (
    "codex", "claude_code", "hermes", "nanobot", "qoder",
    "claudecode", "openagent", "opencode", "qwenpaw",
)
# VeRL's MultiTurnSFTDataset renders and concatenates each message separately.
# Qwen3.5's shipped template intentionally rejects a standalone system/tool/
# assistant message, so use this role-preserving segment template for both the
# data-length audit and VeRL. Tool calls are serialized into assistant content
# below, making every supervised action explicit in the trajectory text.
VERL_SEGMENT_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message['content'] is string %}{{ message['content'] }}"
    "{% else %}{% for block in message['content'] %}"
    "{% if block['type'] == 'text' %}{{ block['text'] }}{% endif %}"
    "{% endfor %}{% endif %}{{ '<|im_end|>\n' }}{% endfor %}"
)



def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train-jsonl", type=Path, default=EXP / "data/sft/multi_harness_trl_v1/train.jsonl")
    p.add_argument("--validation-jsonl", type=Path, default=EXP / "data/sft/multi_harness_trl_v1/validation.jsonl")
    p.add_argument("--output-dir", type=Path, default=EXP / "data/sft/multi_harness_verl_v1")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--train-trajectories-per-harness", type=int, default=80,
                   help="Uniform trajectory draws per harness. 80 x 5 gives 400 one-sequence FSDP updates.")
    p.add_argument("--harnesses", default=",".join(HARNESS_ORDER),
                   help="Comma-separated harnesses required in the balanced training view")
    p.add_argument("--min-qualified-per-harness", type=int, default=1,
                   help="Fail if fewer complete max-length-qualified rows exist for a required harness")
    p.add_argument("--min-qualified-validation", type=int, default=1,
                   help="Fail if fewer complete max-length-qualified validation rows remain")
    p.add_argument("--use-all-train", action="store_true",
                   help="Use every qualified training trajectory instead of fixed per-harness sampling.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def require_packages() -> tuple[Any, Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Dataset building requires pyarrow and transformers in the VeRL environment. "
            "Install VeRL first, then run this builder again."
        ) from exc
    return pa, pq, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}") from exc
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    if not isinstance(value, dict):
        raise ValueError("SFT row lacks metadata")
    return value


def row_id(row: dict[str, Any]) -> str:
    return str(metadata(row).get("record_id", "<unknown>"))


def assert_harness_contract(row: dict[str, Any]) -> str:
    meta = metadata(row)
    harness = meta.get("harness_name")
    if harness not in HARNESS_ORDER:
        raise ValueError(f"{row_id(row)} has invalid harness_name {harness!r}")
    messages = row.get("messages")
    context = meta.get("harness_context")
    if not isinstance(messages, list) or not messages or messages[0].get("role") != "system":
        raise ValueError(f"{row_id(row)} lacks an initial system harness context")
    if not isinstance(context, str) or context not in str(messages[0].get("content", "")):
        raise ValueError(f"{row_id(row)} does not expose its model-visible harness context")
    if not any(m.get("role") == "assistant" for m in messages if isinstance(m, dict)):
        raise ValueError(f"{row_id(row)} has no assistant supervision")
    return harness


def token_length(tokenizer: Any, row: dict[str, Any]) -> int:
    """Use the same complete-chat rendering that VeRL validates for multi-turn data."""
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
        "return_tensors": "pt",
        # Never synthesize an unobserved thinking prefix. Existing visible reasoning is data.
        "enable_thinking": False,
    }
    # Audit exactly the representation stored in Parquet: tool calls have
    # already become explicit assistant text and no separate tool schema is
    # injected by the VeRL segment template.
    encoded = tokenizer.apply_chat_template(verl_messages(row["messages"]), **kwargs)
    return int(encoded["input_ids"].shape[-1])


def json_safe(value: Any) -> Any:
    """Convert numpy/Arrow nested values into stable, serializable tool text."""
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def verl_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Render Qwen3.5's actual tool-call syntax, as used by the eval parser."""
    rendered: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = message.get("content")
        content = "" if content is None else str(content)
        calls = message.get("tool_calls")
        if role == "assistant" and calls:
            blocks = []
            for call in calls:
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                if not name or any(ch in name for ch in "<>\n\r"):
                    raise ValueError(f"invalid tool name {name!r}")
                arguments = function.get("arguments") or {}
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a mapping")
                params = []
                for key, value in arguments.items():
                    if any(ch in str(key) for ch in "<>\n\r"):
                        raise ValueError(f"invalid tool parameter name {key!r}")
                    params.append(f"<parameter={key}>\n{json.dumps(json_safe(value), ensure_ascii=False)}\n</parameter>")
                blocks.append(f"<tool_call>\n<function={name}>\n" + "\n".join(params) + "\n</function>\n</tool_call>")
            content = content.rstrip() + ("\n\n" if content.strip() else "") + "\n".join(blocks)
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role {role!r}")
        rendered.append({"role": role, "content": content})
    return rendered


def canonical_row(row: dict[str, Any], token_count: int, sample_instance: int | None = None) -> dict[str, Any]:
    """Limit Parquet columns to VeRL's documented fields plus simple audit metadata."""
    meta = metadata(row)
    item: dict[str, Any] = {
        "messages": verl_messages(row["messages"]),
        # Calls are textual assistant targets above; the runtime supplies the
        # actual harness tool contract, so VeRL need not inject a second schema.
        "tools": [],
        "enable_thinking": False,
        "record_id": str(meta.get("record_id")),
        "harness_name": str(meta.get("harness_name")),
        "benchmark": str(meta.get("benchmark", "")),
        "family": str(meta.get("family", "")),
        "source_kind": str(meta.get("source_kind", "")),
        "category": str(meta.get("category", "uncategorized")),
        "synthetic": bool(meta.get("synthetic", False)),
        "rendered_tokens": token_count,
    }
    if sample_instance is not None:
        item["sample_instance"] = sample_instance
    return item


def audit_and_qualify(rows: list[dict[str, Any]], tokenizer: Any, max_length: int, split: str) -> tuple[list[tuple[dict[str, Any], int]], list[dict[str, Any]]]:
    accepted: list[tuple[dict[str, Any], int]] = []
    audit: list[dict[str, Any]] = []
    for row in rows:
        event: dict[str, Any] = {"split": split, "record_id": row_id(row), "max_length": max_length}
        try:
            event["harness_name"] = assert_harness_contract(row)
            length = token_length(tokenizer, row)
            event["rendered_tokens"] = length
            event["effective_max_length"] = max_length
            if length > max_length:
                event.update({"accepted": False, "reason": "over_length_unsegmented"})
            else:
                event.update({"accepted": True, "reason": "accepted"})
                accepted.append((row, length))
        except Exception as exc:  # audit faulty input rather than silently changing it
            event.update({"accepted": False, "reason": "render_or_contract_error", "detail": str(exc)})
        audit.append(event)
    return accepted, audit


def balanced_train_view(eligible: list[tuple[dict[str, Any], int]], per_harness: int, seed: int,
                        harness_order: tuple[str, ...] = HARNESS_ORDER) -> list[dict[str, Any]]:
    pools: dict[str, list[tuple[dict[str, Any], int]]] = defaultdict(list)
    for item in eligible:
        pools[assert_harness_contract(item[0])].append(item)
    missing = [name for name in harness_order if not pools[name]]
    if missing:
        raise ValueError(f"No max-length-qualified training trajectories for harness(es): {missing}")
    rng = random.Random(seed)
    result: list[dict[str, Any]] = []
    # Equal harness mass, then round-robin over benchmark/category strata within
    # each harness. This preserves rare safety behaviors without replacement.
    draws = min(per_harness, *(len(pools[name]) for name in harness_order))
    for harness in harness_order:
        strata: dict[tuple[str, str], list[tuple[dict[str, Any], int]]] = defaultdict(list)
        for item in pools[harness]:
            row = item[0]
            meta = metadata(row)
            strata[(str(meta.get("benchmark", "unknown")), str(meta.get("category", "uncategorized")))].append(item)
        for values in strata.values():
            rng.shuffle(values)
        chosen: list[tuple[dict[str, Any], int]] = []
        keys = sorted(strata)
        while len(chosen) < draws:
            progressed = False
            for key in keys:
                if strata[key] and len(chosen) < draws:
                    chosen.append(strata[key].pop())
                    progressed = True
            if not progressed:
                break
        result.extend(canonical_row(row, length) for row, length in chosen)
    rng.shuffle(result)
    return result


def all_train_view(eligible: list[tuple[dict[str, Any], int]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    result = [canonical_row(row, length) for row, length in eligible]
    rng.shuffle(result)
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = args_parser()
    harness_order = tuple(x.strip() for x in args.harnesses.split(",") if x.strip())
    if not harness_order or len(set(harness_order)) != len(harness_order) or any(x not in HARNESS_ORDER for x in harness_order):
        raise ValueError("--harnesses must be nonempty, unique known harness names")
    if args.max_length <= 0 or args.train_trajectories_per_harness <= 0:
        raise ValueError("max length and trajectories per harness must be positive")
    pa, pq, AutoTokenizer = require_packages()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    tokenizer.chat_template = VERL_SEGMENT_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_ok, train_audit = audit_and_qualify(read_jsonl(args.train_jsonl), tokenizer, args.max_length, "train")
    val_ok, val_audit = audit_and_qualify(read_jsonl(args.validation_jsonl), tokenizer, args.max_length, "validation")
    qualified_counts = Counter(assert_harness_contract(row) for row, _ in train_ok)
    scarce = {h: qualified_counts[h] for h in harness_order if qualified_counts[h] < args.min_qualified_per_harness}
    if scarce:
        raise ValueError(f"Not enough complete tokenizer-qualified trajectories per harness: {scarce}")
    if len(val_ok) < args.min_qualified_validation:
        raise ValueError(f"Only {len(val_ok)} complete tokenizer-qualified validation rows; need {args.min_qualified_validation}")
    train_rows = all_train_view(train_ok, args.seed) if args.use_all_train else balanced_train_view(train_ok, args.train_trajectories_per_harness, args.seed, harness_order)
    # Validation remains an un-repeated held-out set. Per-harness/macro evaluation is
    # calculated later from its record-level audit, not by duplicating small harnesses.
    validation_rows = [canonical_row(row, length) for row, length in val_ok]
    if not validation_rows:
        raise ValueError("No max-length-qualified validation trajectories")
    train_path, val_path = args.output_dir / "train.parquet", args.output_dir / "validation.parquet"
    try:
        pq.write_table(pa.Table.from_pylist(train_rows), train_path, compression="zstd")
        pq.write_table(pa.Table.from_pylist(validation_rows), val_path, compression="zstd")
    except Exception as exc:
        raise RuntimeError("Failed to serialize nested messages/tools to Parquet for VeRL") from exc
    from omegaconf import OmegaConf
    from verl.utils import hf_processor
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    processor = hf_processor(args.model, trust_remote_code=False)
    if processor is not None:
        processor.chat_template = VERL_SEGMENT_CHAT_TEMPLATE
        processor.tokenizer.chat_template = VERL_SEGMENT_CHAT_TEMPLATE
    config = OmegaConf.create(dict(
        max_length=args.max_length, pad_mode="no_padding", truncation="error",
        messages_key="messages", tools_key="tools",
        enable_thinking_key="__unused_enable_thinking__", enable_thinking_default=None,
        ignore_input_ids_mismatch=False,
    ))
    for path, rows in ((train_path, train_rows), (val_path, validation_rows)):
        dataset = MultiTurnSFTDataset(str(path), tokenizer, config, processor=processor)
        for index, row in enumerate(rows):
            sample = dataset[index]
            actual = sample["input_ids"].numel()
            if actual != row["rendered_tokens"]:
                raise ValueError(f"{path.name} row {index}: runtime {actual} != audit {row['rendered_tokens']}")
            if not sample["loss_mask"].any():
                raise ValueError(f"{path.name} row {index}: no assistant supervision")
        print(f"Runtime preflight passed: {path.name}, {len(rows)} complete trajectories", flush=True)
    audit_path = args.output_dir / "length_audit.jsonl"
    audit_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in train_audit + val_audit), encoding="utf-8")
    manifest = {
        "format": "verl.MultiTurnSFTDataset parquet v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_tokenizer": args.model,
        "max_length": args.max_length,
        "enable_thinking": False,
        "supervision": "VeRL assistant-turn loss masks over complete multi-turn trajectories; tool calls are explicit assistant text",
        "chat_template": "verl_qwen35_role_segment_v2",
        "harness_order": list(harness_order),
        "equal_real_synthetic_admission_and_sampling": True,
        "train_trajectories_per_harness": None if args.use_all_train else args.train_trajectories_per_harness,
        "actual_balanced_trajectories_per_harness": None if args.use_all_train else len(train_rows) // len(harness_order),
        "use_all_train": args.use_all_train,
        "train_rows": len(train_rows), "validation_rows": len(validation_rows),
        "train_source_eligible_by_harness": dict(sorted(Counter(assert_harness_contract(row) for row, _ in train_ok).items())),
        "train_view_by_harness": dict(sorted(Counter(row["harness_name"] for row in train_rows).items())),
        "train_view_by_benchmark_category": dict(sorted(Counter(
            f"{row['benchmark']}/{row['category']}" for row in train_rows
        ).items())),
        "validation_by_harness": dict(sorted(Counter(row["harness_name"] for row in validation_rows).items())),
        "train_real_synthetic": dict(sorted(Counter("synthetic" if row["synthetic"] else "real" for row in train_rows).items())),
        "validation_real_synthetic": dict(sorted(Counter("synthetic" if row["synthetic"] else "real" for row in validation_rows).items())),
        "rejected_by_reason": dict(sorted(Counter(x["reason"] for x in train_audit + val_audit if not x["accepted"]).items())),
        "input_sha256": {"train": sha256(args.train_jsonl), "validation": sha256(args.validation_jsonl)},
        "output_sha256": {"train": sha256(train_path), "validation": sha256(val_path)},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
