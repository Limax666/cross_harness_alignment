"""Protected VeRL actor update for the 16-episode AgentDojo pilot.

This class must be instantiated by the project TaskRunner, NOT imported into an
unmodified VeRL main_ppo run. Failures propagate and prevent an optimizer step.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from statistics import mean

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from rl_multiharness_update_guard import require_verl_update_batch
from rl_multiharness_stratified_sampler import GROUPS_PER_STRATUM
from rl_metrics_journal import RLMetricsJournal


def admitted_batch_counts(admission: dict, evidence_count: int) -> dict:
    groups = int(admission["groups"])
    rollouts = int(admission["rollouts"])
    variable = int(admission["variable_reward_groups"])
    if groups < 1 or rollouts != evidence_count or not 0 <= variable <= groups:
        raise ValueError("inconsistent admitted online batch counts")
    return {
        "sampled_groups": groups,
        "sampled_rollouts": rollouts,
        "scored_rollouts": rollouts,
        "unscorable_rollouts": 0,
        "zero_variance_group_rate": (groups - variable) / groups,
    }


class GuardedMultiharnessTrainer(RayPPOTrainer):
    def _update_actor(self, batch):
        snapshot = f"step:{self.global_steps - 1}"
        if self.global_steps < 1 or self.config.trainer.get("pilot_policy_snapshot") != "step:0":
            raise ValueError("invalid current-policy snapshot progression")
        import pandas as pd
        rows = pd.read_parquet(self.config.data.train_files).to_dict("records")
        cells = {row["cell_id"]: row for row in rows}
        admission = require_verl_update_batch(batch, policy_snapshot=snapshot, cells=cells)
        if admission.get("admitted") is not True:
            raise ValueError("multi-harness batch not admitted")
        path = Path(self.config.trainer.pilot_metrics_path)
        print("MULTIHARNESS_ADMITTED", admission, flush=True)
        result = super()._update_actor(batch)
        raw_metrics = result.meta_info.get("metrics")
        if not raw_metrics:
            raise ValueError("VeRL actor returned no optimizer metrics; do not journal")
        from verl.utils.metric import reduce_metrics
        reduced = reduce_metrics(raw_metrics)
        metrics = {}
        for key, value in reduced.items():
            numeric = float(value.item() if hasattr(value, "item") else value)
            if not math.isfinite(numeric):
                raise ValueError(f"nonfinite optimizer metric: {key}")
            metrics[key] = numeric
        evidence = batch.non_tensor_batch["multiharness_evidence"]
        by_harness = {}
        strata = sorted({(item["benchmark"], item["harness"]) for item in evidence})
        for benchmark, harness in strata:
            arms = {}
            for arm in ("benign", "risk"):
                group = [item for item in evidence if item["benchmark"] == benchmark and
                         item["harness"] == harness and item["arm"] == arm]
                expected = 4 * GROUPS_PER_STRATUM
                if len(group) != expected or not group:
                    raise ValueError(f"missing G=4 rollouts for {benchmark}/{harness}/{arm}")
                arms[arm] = {
                    "mean_reward": mean(item["reward"] for item in group),
                    "mean_path_signal": mean(float(item["reward_path_signal"]) for item in group),
                    "nonengagement_rate": mean(item["reward_outcome"] == "benign_nonengagement"
                                               for item in group),
                    "utility": mean(float(item["official_utility"]) for item in group),
                    "attack_success_rate": mean(bool(item["official_attack_success"]) for item in group),
                    "persistence_rate": mean(bool(item["prohibited_persistence"]) for item in group),
                    "contract_violation_rate": mean(bool(item["invalid_tool_dispatched"] or
                                                         item["blocked_tool_calls"] or
                                                         item["termination"] == "protocol_error") for item in group),
                    "timeout_rate": mean(item["termination"] == "model_timeout" for item in group),
                }
            by_harness[f"{benchmark}/{harness}"] = arms
        response_masks = batch.batch["response_mask"].tolist()
        rollout_log_probs = batch.batch["rollout_log_probs"].tolist()
        response_lengths = [sum(mask) for mask in response_masks]
        token_probs = [math.exp(lp) for masks, probabilities in zip(response_masks, rollout_log_probs)
                       for mask, lp in zip(masks, probabilities) if mask]
        sorted_lengths = sorted(response_lengths)
        metrics.update({
            "mean_response_tokens": mean(response_lengths),
            "p95_response_tokens": sorted_lengths[math.ceil(0.95 * len(sorted_lengths)) - 1],
            "mean_token_probability": mean(token_probs),
            "approx_kl": metrics.get("actor/ppo_kl", 0.0),
            "clip_fraction": metrics.get("actor/pg_clipfrac", 0.0),
            "grad_norm": metrics["actor/grad_norm"],
        })
        train_file = Path(self.config.data.train_files)
        sft = str(self.config.actor_rollout_ref.model.path)
        journal = RLMetricsJournal(path, {
            "run_id": str(self.config.trainer.experiment_name), "algorithm": str(self.config.trainer.get("rl_algorithm", "GRPO")),
            "rollout_mode": "online_current_policy", "model_checkpoint": sft,
            "reference_checkpoint": sft, "reward_version": "native_outcome_first",
            "verifier_version": "multi-native-outcome-v1",
            "harnesses": sorted(by_harness), "train_split_hash": hashlib.sha256(train_file.read_bytes()).hexdigest(),
        })
        journal.log({
            "step": int(self.global_steps), "phase": "train",
            **admitted_batch_counts(admission, len(evidence)),
            "rewards_by_cell": admission["rewards_by_cell"],
            "optimizer": metrics,
            "harness_metrics": by_harness,
        })
        return result
