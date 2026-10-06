#!/usr/bin/env python3
"""Create the registered native AgentDojo prompt rows; no model responses or rewards."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cross_harness_sft.backends.agentdojo import AgentDojoDriver
from rl_online_batch_gate import CELLS
from rl_verl_chat_template import HERMES_RL_SYSTEM_CONTEXT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker-config", default=str(ROOT / "configs/agentdojo_worker.yaml"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import pandas as pd
    driver = AgentDojoDriver(args.worker_config)
    native = {case["task_id"]: case for case in driver.cases()}
    if not CELLS <= native.keys():
        raise ValueError("registered pilot cells absent from native suite")
    train = ROOT / "data/sft/teacher_release_v7/accepted_train.jsonl"
    validation = ROOT / "data/sft/teacher_release_v7/accepted_validation.jsonl"
    families = {"agentdojo:" + cell.rpartition(":")[0] for cell in CELLS}
    train_families = {json.loads(line).get("source_family_id") for line in train.open(encoding="utf-8") if line.strip()}
    val_families = {json.loads(line).get("source_family_id") for line in validation.open(encoding="utf-8") if line.strip()}
    if not families <= train_families or families & val_families:
        raise ValueError("pilot families must be in accepted SFT train and disjoint from validation")
    rows = []
    for cell in sorted(CELLS):
        case = native[cell]
        if case["semantic_task_family_id"] != cell.rpartition(":")[0]:
            raise ValueError("unexpected native family identifier")
        rows.append({"data_source": "agentdojo_native_train", "task_id": cell,
                     "split": "train", "policy_snapshot": "step:0", "agent_name": "native_agentdojo_pilot",
                     "prompt": [{"role": "system", "content": HERMES_RL_SYSTEM_CONTEXT},
                                {"role": "user", "content": case["user_query"]}]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(json.dumps({"output": str(args.output), "rows": len(rows), "families": sorted(families)}))


if __name__ == "__main__":
    main()
