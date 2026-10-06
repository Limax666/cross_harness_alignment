"""Deterministic train-family group allocation for CHS-PO sampling ablations."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskCell:
    benchmark: str
    family_id: str
    harness: str
    arm: str  # benign or risk
    task_id: str
    fixture_revision: str
    split: str


@dataclass(frozen=True)
class RolloutGroup:
    cell: TaskCell
    seeds: tuple[int, ...]


def allocate_groups(cells: list[TaskCell], *, groups_per_stratum: int,
                    group_size: int, seed: int) -> list[RolloutGroup]:
    if groups_per_stratum < 1 or group_size < 2:
        raise ValueError("positive group count and group_size >= 2 required")
    if not cells or any(cell.split != "train" for cell in cells):
        raise ValueError("only registered train-family cells may be sampled")
    if any(not all((cell.benchmark, cell.family_id, cell.harness,
                    cell.task_id, cell.fixture_revision)) or cell.arm not in {"benign", "risk"}
           for cell in cells):
        raise ValueError("incomplete task-cell identity")
    by_family: dict[str, str] = {}
    strata: dict[tuple[str, str], list[TaskCell]] = {}
    for cell in cells:
        if cell.family_id in by_family and by_family[cell.family_id] != cell.benchmark:
            raise ValueError("family ID collides across benchmark namespaces")
        by_family[cell.family_id] = cell.benchmark
        strata.setdefault((cell.harness, cell.arm), []).append(cell)
    harnesses = {cell.harness for cell in cells}
    if any((harness, arm) not in strata for harness in harnesses for arm in ("benign", "risk")):
        raise ValueError("both benign and risk cells required per harness")
    rng = random.Random(seed)
    result: list[RolloutGroup] = []
    for stratum in sorted(strata):
        options = sorted(strata[stratum], key=lambda c: (c.benchmark, c.family_id, c.task_id))
        for index in range(groups_per_stratum):
            cell = options[index % len(options)]
            identity = "\x1f".join((cell.benchmark, cell.family_id, cell.harness,
                                   cell.arm, cell.task_id, cell.fixture_revision, str(seed), str(index)))
            digest = hashlib.sha256(identity.encode()).digest()
            local_rng = random.Random(int.from_bytes(digest[:8], "big") ^ rng.randrange(2**63))
            seeds = tuple(local_rng.sample(range(2**31), group_size))
            result.append(RolloutGroup(cell, seeds))
    return result
