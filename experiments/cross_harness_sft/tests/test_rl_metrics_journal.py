from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _metadata(rollout_mode: str = "online_current_policy"):
    return {
        "run_id": "journal-test", "algorithm": "GRPO", "model_checkpoint": "sft-step-267",
        "reference_checkpoint": "sft-step-267", "reward_version": "rlvr-v1",
        "verifier_version": "audit-v1", "harnesses": ["codex"],
        "train_split_hash": "fixture-hash", "rollout_mode": rollout_mode,
    }


def _row(step: int = 1):
    return {
        "step": step, "phase": "train", "sampled_groups": 2, "sampled_rollouts": 4,
        "scored_rollouts": 4, "unscorable_rollouts": 0,
        "optimizer": {"entropy": 0.6, "mean_token_probability": 0.7,
                      "mean_response_tokens": 180, "p95_response_tokens": 320,
                      "approx_kl": 0.01, "clip_fraction": 0.08, "grad_norm": 0.9},
        "harness_metrics": {"codex": {
            "benign": {"mean_reward": 0.5, "utility": 0.5, "false_refusal_rate": 0.0,
                       "group_zero_variance_rate": 0.0},
            "risk": {"mean_reward": 0.0, "utility": 0.5, "attack_success_rate": 0.0,
                     "persistence_rate": 0.0, "contract_violation_rate": 0.0,
                     "group_zero_variance_rate": 0.0},
        }},
    }


class RLMetricsJournalTest(unittest.TestCase):
    def test_journal_requires_online_rollouts_and_rejects_invalid_counts(self):
        journal_module = _load("rl_metrics_journal")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            with self.assertRaisesRegex(ValueError, "online_current_policy"):
                journal_module.RLMetricsJournal(path, _metadata("teacher_replay"))
            journal = journal_module.RLMetricsJournal(path, _metadata())
            row = _row()
            row["scored_rollouts"] = 5
            with self.assertRaisesRegex(ValueError, "exceeds"):
                journal.log(row)

    def test_metrics_journal_and_dashboard_roundtrip(self):
        journal_module = _load("rl_metrics_journal")
        plot_module = _load("plot_rl_training_metrics")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            journal = journal_module.RLMetricsJournal(path, _metadata())
            journal.log(_row())
            row = _row(2)
            row["phase"] = "validation"
            row["sampled_groups"] = row["sampled_rollouts"] = 0
            row["scored_rollouts"] = row["unscorable_rollouts"] = 0
            journal.log(row)
            metadata, records = plot_module.load_journal(path)
            self.assertEqual(metadata["rollout_mode"], "online_current_policy")
            self.assertEqual(len(records), 2)
            output = Path(tmp) / "dashboard.png"
            subprocess.run([
                "/data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/python",
                str(SCRIPTS / "plot_rl_training_metrics.py"), str(path), "--output", str(output),
            ], check=True, capture_output=True, text=True)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)

    def test_dashboard_refuses_offline_replay_data(self):
        plot_module = _load("plot_rl_training_metrics")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "offline.jsonl"
            path.write_text('{"record_type":"run","metadata":{"rollout_mode":"teacher_replay"}}\n')
            with self.assertRaisesRegex(ValueError, "offline/replay"):
                plot_module.load_journal(path)


if __name__ == "__main__":
    unittest.main()
