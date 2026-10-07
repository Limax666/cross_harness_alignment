#!/usr/bin/env python3
"""Freeze an executable, family-gated AgentDojo/AgentHarm RL task pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_harness_sft.backends.agentdojo import AgentDojoDriver
from cross_harness_sft.backends.agentharm import AgentHarmDriver
from rl_hermes_mcp_contract import hermes_mcp_tools
from rl_mcp_benchmark_contract import mcp_benchmark_tools
from rl_multiharness_chat_template import BENCHMARK_MCP_CHAT_TEMPLATE
from rl_native_agentharm_episode import READ_ONLY_TOOLS
from rl_agentdojo_path_audit import _ground_truth_tools, GENERALIZED_FAMILIES, READ_TOOLS


STRATA = (("agentdojo", "codex"), ("agentdojo", "claude_code"),
          ("agentharm", "codex"), ("agentharm", "hermes"))
MIN_PER_ARM = 4
MAX_PER_ARM = 24


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-limit", type=int, default=6144)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError(args.output)
    import pandas as pd
    from transformers import AutoTokenizer

    catalog_manifest = json.loads((args.catalog.parent / "catalog_manifest.json").read_text())
    if catalog_manifest.get("online_rollout_ready") is not False:
        raise ValueError("source catalog status changed unexpectedly")
    catalog = {}
    for line in args.catalog.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["benchmark"], row["harness_name"], row["task_id"])
        if key in catalog or row.get("split") != "train":
            raise ValueError(f"duplicate or non-train catalog task: {key}")
        catalog[key] = row
    dojo = AgentDojoDriver(str(ROOT / "configs/agentdojo_worker.yaml"))
    harm = AgentHarmDriver(str(ROOT / "configs/agentharm_rl_train_worker.yaml"))
    native = {
        "agentdojo": {row["task_id"]: row for row in dojo.cases()},
        "agentharm": {row["task_id"]: row for row in harm.cases()},
    }
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267",
        local_files_only=True,
    )
    tokenizer.chat_template = BENCHMARK_MCP_CHAT_TEMPLATE
    rows = []
    for benchmark, harness in STRATA:
        selected = Counter()
        pools = {}
        audited_families = {}
        for key, item in catalog.items():
            if key[:2] != (benchmark, harness):
                continue
            if benchmark == "agentdojo" and not (key[2].endswith(":clean") or
                                                 key[2].rpartition(":")[2].startswith("injection_task_")):
                continue
            arm = "benign" if key[2].endswith(":clean") or key[2].startswith("benign:") else "risk"
            pools.setdefault((arm, item["source_family_id"]), []).append(key[2])
        candidates = []
        for arm in ("benign", "risk"):
            families = sorted(family for kind, family in pools if kind == arm)
            for offset in range(max(len(pools[(arm, family)]) for family in families)):
                for family in families:
                    ids = sorted(pools[(arm, family)])
                    if offset < len(ids):
                        candidates.append((arm, ids[offset]))
        for arm, task_id in candidates:
            if selected[arm] >= MAX_PER_ARM:
                continue
            item = catalog[(benchmark, harness, task_id)]
            case = native[benchmark].get(task_id)
            if case is None:
                continue
            if benchmark == "agentdojo":
                family = task_id.rpartition(":")[0]
                if family not in audited_families:
                    try:
                        plan = _ground_truth_tools(dojo.start(task_id, 0)["episode"])
                        audited_families[family] = plan is not None and (family in GENERALIZED_FAMILIES or
                                                                          plan <= READ_TOOLS)
                    except Exception:
                        audited_families[family] = False
                if not audited_families[family]:
                    continue
            if item["source_objective"].strip() != case["user_query"].strip():
                raise ValueError(f"catalog prompt differs from native task: {benchmark}/{harness}/{task_id}")
            native_tools = case["tools"]
            if benchmark == "agentharm" and not {t["function"]["name"] for t in native_tools} <= READ_ONLY_TOOLS:
                continue
            tools, mapping = (hermes_mcp_tools(native_tools) if harness == "hermes"
                              else mcp_benchmark_tools(native_tools))
            if len(mapping) != len(native_tools):
                raise ValueError("MCP conversion dropped a native tool")
            messages = [{"role": "system", "content": item["harness_context"]},
                        {"role": "user", "content": case["user_query"]}]
            rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True)
            token_ids = rendered["input_ids"] if isinstance(rendered, Mapping) else rendered
            if hasattr(token_ids, "shape"):
                length = int(token_ids.shape[-1])
            elif token_ids and isinstance(token_ids[0], list):
                length = len(token_ids[0])
            else:
                length = len(token_ids)
            if length > args.prompt_limit:
                continue
            rows.append({
                "data_source": f"{benchmark}_{harness}_native_train",
                "benchmark": benchmark, "harness_name": harness,
                "task_id": task_id, "cell_id": f"{benchmark}/{harness}/{task_id}",
                "source_family_id": item["source_family_id"], "arm": arm,
                "split": "train", "policy_snapshot": "step:0",
                "agent_name": "native_multiharness_v1", "prompt": messages,
                "prompt_tokens": length, "tool_contract_sha256": digest(tools),
            })
            selected[arm] += 1
        if min(selected["benign"], selected["risk"]) < MIN_PER_ARM:
            raise ValueError(f"too few qualified train cells in {benchmark}/{harness}: {selected}")
    by_harness = Counter(row["harness_name"] for row in rows)
    by_stratum = Counter((row["harness_name"], row["arm"]) for row in rows)
    by_source_arm = Counter((row["benchmark"], row["harness_name"], row["arm"]) for row in rows)
    if len(by_source_arm) != 8 or min(by_source_arm.values()) < MIN_PER_ARM:
        raise ValueError("expected at least four cells in each source/harness/arm stratum")
    if len({row["cell_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate multi-harness training cell")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    manifest = {
        "status": "native_train_tasks_selected_pending_full_online_gate",
        "model_initialization": "qwen35-2b-v7-sft-step-267",
        "algorithm_family": ["grpo", "chs_po", "dapo"],
        "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "family_manifest_sha256": catalog_manifest["family_manifest_sha256"],
        "cells": [{key: row[key] for key in ("cell_id", "source_family_id", "arm", "prompt_tokens", "tool_contract_sha256")}
                  for row in rows],
        "harness_groups": dict(by_harness),
        "harness_arm_groups": {f"{harness}/{arm}": count for (harness, arm), count in by_stratum.items()},
        "source_harness_arm_groups": {"/".join(key): count for key, count in by_source_arm.items()},
        "max_prompt_tokens": max(row["prompt_tokens"] for row in rows),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "groups": len(rows),
                      "max_prompt_tokens": manifest["max_prompt_tokens"],
                      "harness_arm_groups": manifest["harness_arm_groups"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
