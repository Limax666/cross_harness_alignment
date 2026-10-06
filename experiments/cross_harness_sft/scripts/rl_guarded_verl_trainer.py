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
from rl_verl_update_guard import require_verl_update_batch
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


class GuardedNativePilotTrainer(RayPPOTrainer):
    def _update_actor(self, batch):
        snapshot = f"step:{self.global_steps - 1}"
        if self.global_steps < 1 or self.config.trainer.get("pilot_policy_snapshot") != "step:0":
            raise ValueError("invalid current-policy snapshot progression")
        admission = require_verl_update_batch(batch, policy_snapshot=snapshot)
        if admission.get("admitted") is not True:
            raise ValueError("native pilot batch not admitted")
        path = Path(self.config.trainer.pilot_metrics_path)
        print("NATIVE_PILOT_ADMITTED", admission, flush=True)
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
        evidence = batch.non_tensor_batch["pilot_evidence"]
        arms = {}
        for arm, label in (("clean", "benign"), ("injection_task_1", "risk")):
            rows = [item for item in evidence if item["task_id"].endswith(":" + arm)]
            arms[label] = {
                "mean_reward": sum(item["reward"] for item in rows) / len(rows),
                "utility": sum(float(item["official_utility"]) for item in rows) / len(rows),
                "attack_success_rate": sum(bool(item["official_attack_success"]) for item in rows) / len(rows),
                "persistence_rate": mean(bool(item["prohibited_persistence"]) for item in rows),
                "contract_violation_rate": mean(bool(item["invalid_tool_dispatched"] or
                                                     item["blocked_tool_calls"] or
                                                     item["termination"] == "protocol_error") for item in rows),
                "timeout_rate": mean(item["termination"] == "model_timeout" for item in rows),
            }
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
            "run_id": str(self.config.trainer.experiment_name), "algorithm": "GRPO",
            "rollout_mode": "online_current_policy", "model_checkpoint": sft,
            "reference_checkpoint": sft, "reward_version": "native_outcome_first",
            "verifier_version": str(evidence[0]["audit_version"]),
            "harnesses": ["hermes"], "train_split_hash": hashlib.sha256(train_file.read_bytes()).hexdigest(),
        })
        journal.log({
            "step": int(self.global_steps), "phase": "train",
            **admitted_batch_counts(admission, len(evidence)),
            "rewards_by_cell": admission["rewards_by_cell"],
            "optimizer": metrics,
            "harness_metrics": {"hermes": arms},
        })
        return result
