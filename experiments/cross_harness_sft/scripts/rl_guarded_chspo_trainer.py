"""CHS-PO ablation: bounded, train-only worst-stratum advantage weighting.

The optimizer and rollouts are VeRL GRPO. Only the source×harness surrogate
weight differs from the matched GRPO control; no validation/test signal enters.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

from rl_guarded_multiharness_trainer import GuardedMultiharnessTrainer
from rl_multiharness_update_guard import require_verl_update_batch


class GuardedCHSPOTrainer(GuardedMultiharnessTrainer):
    def _update_actor(self, batch):
        snapshot = f"step:{self.global_steps - 1}"
        import pandas as pd
        cells = {row["cell_id"]: row for row in pd.read_parquet(self.config.data.train_files).to_dict("records")}
        require_verl_update_batch(batch, policy_snapshot=snapshot, cells=cells)
        evidence = batch.non_tensor_batch["multiharness_evidence"]
        by_stratum = defaultdict(list)
        for item in evidence:
            by_stratum[f"{item['benchmark']}/{item['harness']}"].append(float(item["reward"]))
        if len(by_stratum) != 4:
            raise ValueError("CHS-PO requires four registered source×harness strata")
        # Freeze the weight for this update from the current train batch. Higher
        # weight goes to lower-reward strata; clip before normalization.
        raw = {name: max(0.5, min(2.0, 1.0 - sum(values) / len(values)))
               for name, values in by_stratum.items()}
        normalizer = sum(raw[f"{item['benchmark']}/{item['harness']}"] for item in evidence) / len(evidence)
        weights = [raw[f"{item['benchmark']}/{item['harness']}"] / normalizer for item in evidence]
        advantages = batch.batch.get("advantages")
        if advantages is None or advantages.shape[0] != len(weights):
            raise ValueError("CHS-PO advantage tensor does not match admitted rollouts")
        weighted = advantages * torch.tensor(weights, dtype=advantages.dtype, device=advantages.device).unsqueeze(1)
        if not torch.isfinite(weighted).all():
            raise ValueError("CHS-PO produced nonfinite weighted advantages")
        batch.batch["advantages"] = weighted
        result = super()._update_actor(batch)
        path = Path(self.config.trainer.pilot_metrics_path).with_name("stratum_weights.jsonl")
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"step": int(self.global_steps), "train_reward_means":
                                     {name: sum(values) / len(values) for name, values in by_stratum.items()},
                                     "raw_weights": raw, "normalizer": normalizer}, sort_keys=True) + "\n")
        return result
