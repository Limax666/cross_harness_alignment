"""Stage the exact CHS-PO dual/gate state matching a VeRL checkpoint.

The actor and AdamW state stay in the source checkpoint. A fresh run directory
gets only the small, checkpoint-matched constraint state and provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def prepare(
    source_run: Path,
    checkpoint: Path,
    destination: Path,
    train_data: Path,
    objective_version: str | None = None,
    allow_objective_migration: bool = False,
    priority_snapshot: Path | None = None,
) -> None:
    source_run, checkpoint, destination, train_data = (
        path.resolve() for path in (source_run, checkpoint, destination, train_data)
    )
    if checkpoint.parent != source_run / "checkpoints" or not checkpoint.name.startswith("global_step_"):
        raise ValueError("checkpoint must belong to source run's checkpoints directory")
    step = int(checkpoint.name.removeprefix("global_step_"))
    if step < 1 or not (checkpoint / "data.pt").is_file():
        raise ValueError("missing checkpoint or dataloader state")
    actor = checkpoint / "actor"
    for rank in (0, 1):
        for kind in ("model", "optim", "extra_state"):
            if not (actor / f"{kind}_world_size_2_rank_{rank}.pt").is_file():
                raise ValueError(f"missing {kind} checkpoint for rank {rank}")
    if destination.exists():
        raise FileExistsError(f"resume run directory already exists: {destination}")

    metrics = [json.loads(line) for line in (source_run / "metrics.jsonl").read_text().splitlines()]
    headers = [row for row in metrics if row.get("record_type") == "run"]
    if len(headers) != 1 or not isinstance(headers[0].get("metadata", {}).get("algorithm"), str):
        raise ValueError("source run lacks one objective-version header")
    source_objective_version = headers[0]["metadata"]["algorithm"]
    target_objective_version = objective_version or source_objective_version
    if target_objective_version != source_objective_version and not allow_objective_migration:
        raise ValueError(
            "objective version migration requires --allow-objective-migration; "
            f"source={source_objective_version!r}, target={target_objective_version!r}"
        )
    completed = [row for row in metrics if row.get("record_type") == "metrics" and row.get("step") == step]
    safety = [json.loads(line) for line in (source_run / "chspo_safety.jsonl").read_text().splitlines()]
    matched = [row for row in safety if row.get("step") == step]
    if len(completed) != 1 or len(matched) != 1:
        raise ValueError("checkpoint step lacks exactly one committed metrics and safety record")
    if completed[0].get("scored_rollouts") != 128 or completed[0].get("unscorable_rollouts") != 0:
        raise ValueError("checkpoint step did not have a fully admitted online batch")
    row = matched[0]
    if not isinstance(row.get("duals"), dict) or not isinstance(row.get("safety_streaks"), dict):
        raise ValueError("checkpoint step lacks CHS-PO constraint state")
    priority_bytes = None
    if priority_snapshot is not None:
        priority_bytes = priority_snapshot.read_bytes()
        priority = json.loads(priority_bytes)
        if target_objective_version != "CHS-PO-harness-safety-dual-v4" or priority.get("last_step") != step:
            raise ValueError("v4 priority snapshot must match the checkpoint step")
        if priority.get("source_sha256") != hashlib.sha256((source_run / "metrics.jsonl").read_bytes()).hexdigest():
            raise ValueError("v4 priority source metrics changed")
    provenance = {
        "checkpoint": str(checkpoint),
        "source_run": str(source_run),
        "step": step,
        "sampler_epoch": int(row.get("sampler_epoch", step)),
        "train_sha256": hashlib.sha256(train_data.read_bytes()).hexdigest(),
        "source_objective_version": source_objective_version,
        "objective_version": target_objective_version,
        "objective_migration": target_objective_version != source_objective_version,
        "duals": row["duals"],
        "safety_streaks": row["safety_streaks"],
    }
    if priority_bytes is not None:
        provenance["priority_snapshot_sha256"] = hashlib.sha256(priority_bytes).hexdigest()
    destination.mkdir(parents=True)
    for name, value in (("chspo_duals.json", row["duals"]),
                        ("chspo_safety_streaks.json", row["safety_streaks"]),
                        ("chspo_resume_provenance.json", provenance)):
        (destination / name).write_text(json.dumps(value, sort_keys=True) + "\n")
    if priority_bytes is not None:
        (destination / "chspo_priority_snapshot.json").write_bytes(priority_bytes)
    print(f"CHS-PO resume staged: step={step}, source={checkpoint}, destination={destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--objective-version")
    parser.add_argument("--allow-objective-migration", action="store_true")
    parser.add_argument("--priority-snapshot", type=Path)
    args = parser.parse_args()
    prepare(args.source_run, args.checkpoint, args.destination, args.train_data,
            objective_version=args.objective_version,
            allow_objective_migration=args.allow_objective_migration,
            priority_snapshot=args.priority_snapshot)
