"""CHS-PO must distinguish a useless GRPO batch from safety-only learning."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "verl"))
sys.path.insert(0, str(ROOT / "scripts"))

from rl_guarded_chspo_trainer import GuardedCHSPOTrainer
from rl_multiharness_batch_gate import MultiharnessAdmissionError, require_online_batch
from rl_multiharness_stratified_sampler import EXPECTED_STRATA, GROUPS_PER_STRATUM


def constant_reward_rows():
    cells, rows = {}, []
    for benchmark, harness, arm in sorted(EXPECTED_STRATA):
        for group in range(GROUPS_PER_STRATUM):
            cell_id = f"{benchmark}/{harness}/{arm}/{group}"
            cells[cell_id] = {"benchmark": benchmark, "harness_name": harness,
                              "arm": arm, "source_family_id": f"family-{group}",
                              "task_id": f"task-{group}"}
            for seed in range(4):
                rows.append({"rollout_id": f"{cell_id}/{seed}", "cell_id": cell_id,
                             "split": "train", "benchmark": benchmark, "harness": harness,
                             "arm": arm, "source_family_id": f"family-{group}",
                             "task_id": f"task-{group}", "seed": seed,
                             "policy_snapshot": "step:67", "on_policy_rollout": True,
                             "reset_verified": True, "harness_contract_verified": True,
                             "audit_complete": True, "initial_state_sha256": "a" * 64,
                             "reward": 1.0, "reward_path_signal": 0.0,
                             "response_ids": [7], "response_mask": [1],
                             "token_origins": ["assistant"],
                             "rollout_log_probs": [-0.5], "old_log_probs": [-0.5]})
    return rows, cells


def test_standard_grpo_still_rejects_constant_batch_but_chspo_can_inspect_it():
    rows, cells = constant_reward_rows()
    with pytest.raises(MultiharnessAdmissionError, match="zero reward advantage"):
        require_online_batch(rows, policy_snapshot="step:67", cells=cells)
    admitted = require_online_batch(rows, policy_snapshot="step:67", cells=cells,
                                    allow_zero_reward_variance=True)
    assert admitted["variable_reward_groups"] == 0
    assert admitted["rollouts"] == 128


def test_chspo_skips_only_when_adjusted_policy_advantage_is_zero(tmp_path, monkeypatch):
    import rl_guarded_chspo_trainer as module

    monkeypatch.setattr(module, "require_verl_update_batch", lambda *a, **k:
                        {"groups": 32, "rollouts": 128, "variable_reward_groups": 0})
    monkeypatch.setattr(module.pd if hasattr(module, "pd") else __import__("pandas"),
                        "read_parquet", lambda _: SimpleNamespace(to_dict=lambda _: []))
    trainer = object.__new__(GuardedCHSPOTrainer)
    trainer.config = SimpleNamespace(data=SimpleNamespace(train_files="unused"),
                                     trainer=SimpleNamespace(pilot_metrics_path=str(tmp_path / "metrics.jsonl")))
    trainer.global_steps = 68
    trainer.train_dataloader = SimpleNamespace(sampler=SimpleNamespace(epoch=68))
    evidence = [{"benchmark": "agentharm", "harness": "hermes", "arm": arm,
                 "official_attack_success": False, "prohibited_persistence": False,
                 "invalid_tool_dispatched": False, "blocked_tool_calls": 0,
                 "termination": "completed", "reward_outcome": "safe_complete"}
                for arm in ("risk", "benign") for _ in range(4)]
    batch = SimpleNamespace(batch={"advantages": torch.zeros(8, 2),
                                   "response_mask": torch.ones(8, 2)},
                            non_tensor_batch={"multiharness_evidence": evidence})
    assert trainer._skip_uninformative_batch(batch) is True
    skipped = json.loads((tmp_path / "chspo_skipped_batches.jsonl").read_text().strip())
    assert skipped["step_attempted"] == 68 and skipped["sampler_epoch"] == 68
    evidence[0]["official_attack_success"] = True
    assert trainer._skip_uninformative_batch(batch) is False
