#!/usr/bin/env python3
"""Serialize the frozen teacher view to VeRL Parquet and check every loss mask."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("view", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--max-length", type=int, default=4096)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Output exists; materialize a new immutable version")
    from transformers import AutoTokenizer
    import pyarrow as pa
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    script = Path(__file__).with_name("build_multiharness_verl_sft.py")
    spec = importlib.util.spec_from_file_location("verl_renderer", script)
    renderer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(renderer)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    tokenizer.chat_template = renderer.VERL_SEGMENT_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    view_manifest = json.loads((args.view / "sampling_manifest.json").read_text())
    if view_manifest["format"] not in {"complete_teacher_cell_stratified_view_v1", "complete_teacher_full_preflight_view_v1"}:
        ap.error("Unsupported sampling view")
    valid_no_replacement = view_manifest.get("no_replacement_within_epoch") or view_manifest.get("no_replacement_except_explicit_padding")
    if view_manifest["global_batch"] <= 0 or not valid_no_replacement:
        ap.error("Sampling view does not meet the batch/no-replacement contract")
    args.output.mkdir(parents=True)
    outputs = {}
    for split, source in (("train", "train_balanced.jsonl"), ("validation", "validation_natural.jsonl")):
        records = load(args.view / source)
        items = []
        for row in records:
            preflight = row["token_preflight"]
            if preflight["model"] != args.model or preflight["max_length"] != args.max_length:
                raise ValueError("Tokenizer preflight configuration mismatch")
            if preflight["template_sha256"] != hashlib.sha256(tokenizer.chat_template.encode()).hexdigest():
                raise ValueError("Chat template changed since tokenizer preflight")
            items.append({"messages": renderer.verl_messages(row["messages"]), "tools": [],
                          "enable_thinking": False, "record_id": row["record_id"],
                          "harness_name": row["harness_name"], "benchmark": row["benchmark"],
                          "source_family_id": row["source_family_id"],
                          "primary_behavior_type": row["primary_behavior_type"],
                          "quality_category": row["quality_category"],
                          "rendered_tokens": preflight["rendered_tokens"],
                          "assistant_tokens": preflight["assistant_tokens"]})
        path = args.output / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(items), path, compression="zstd")
        outputs[split] = {"rows": len(items), "sha256": digest(path)}
    # Use the same runtime dataset class and loss mask as the actual trainer.
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
    config = OmegaConf.create({"max_length": args.max_length, "pad_mode": "no_padding",
                               "truncation": "error", "messages_key": "messages", "tools_key": "tools",
                               "enable_thinking_key": "__unused_enable_thinking__",
                               "enable_thinking_default": None, "ignore_input_ids_mismatch": False})
    for split, source in (("train", "train_balanced.jsonl"), ("validation", "validation_natural.jsonl")):
        dataset = MultiTurnSFTDataset(str(args.output / f"{split}.parquet"), tokenizer, config, processor=None)
        records = load(args.view / source)
        for index, row in enumerate(records):
            sample = dataset[index]
            actual = int(sample["input_ids"].numel())
            supervised = int(sample["loss_mask"].sum().item())
            if actual != row["token_preflight"]["rendered_tokens"] or supervised != row["token_preflight"]["assistant_tokens"]:
                raise ValueError(f"{split} row {index}: runtime length/loss mask differs from frozen preflight")
            if actual > args.max_length or supervised == 0:
                raise ValueError(f"{split} row {index}: invalid complete trajectory")
        outputs[split]["runtime_preflight"] = "passed"
    manifest = {"format": "teacher_sft_verl_materialization_v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "sampling_manifest_sha256": digest(args.view / "sampling_manifest.json"),
                "model": args.model, "max_length": args.max_length,
                "sampler_shuffle_required": False, "expected_global_batch": view_manifest["global_batch"],
                "outputs": outputs}
    (args.output / "materialization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
