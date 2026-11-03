import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_chspo_resume import prepare


def test_v4_resume_requires_checkpoint_matched_priority_snapshot(tmp_path):
    source = tmp_path / "source"
    checkpoint = source / "checkpoints/global_step_140"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"dataloader")
    for rank in (0, 1):
        for kind in ("model", "optim", "extra_state"):
            (actor / f"{kind}_world_size_2_rank_{rank}.pt").write_bytes(b"checkpoint")
    train = tmp_path / "train.parquet"
    train.write_bytes(b"frozen train cells")
    metrics = source / "metrics.jsonl"
    metrics.write_text(
        json.dumps({"record_type": "run", "metadata": {"algorithm": "CHS-PO-harness-safety-dual-v3"}})
        + "\n" + json.dumps({"record_type": "metrics", "phase": "train", "step": 140,
                              "scored_rollouts": 128, "unscorable_rollouts": 0}) + "\n"
    )
    (source / "chspo_safety.jsonl").write_text(json.dumps(
        {"step": 140, "sampler_epoch": 140, "duals": {"agentharm/codex": {"attack": 0.25}},
         "safety_streaks": {"agentharm/codex": {"attack": 0}}}) + "\n")
    snapshot = tmp_path / "priority.json"
    snapshot.write_text(json.dumps({"last_step": 140,
                                    "source_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
                                    "weights": {}, "unseen_weight": 2.0}))
    destination = tmp_path / "v4"
    prepare(source, checkpoint, destination, train,
            objective_version="CHS-PO-harness-safety-dual-v4",
            allow_objective_migration=True, priority_snapshot=snapshot)
    provenance = json.loads((destination / "chspo_resume_provenance.json").read_text())
    assert provenance["step"] == 140
    assert provenance["objective_migration"] is True
    assert hashlib.sha256((destination / "chspo_priority_snapshot.json").read_bytes()).hexdigest() == provenance["priority_snapshot_sha256"]
