"""Rotate train-family cells while keeping every online batch source/arm balanced."""
from __future__ import annotations

import random
from collections import defaultdict
from torch.utils.data import Sampler


GROUPS_PER_STRATUM = 4
EXPECTED_STRATA = {(benchmark, harness, arm)
                   for benchmark, harness in (("agentdojo", "codex"),
                                              ("agentdojo", "claude_code"),
                                              ("agentharm", "codex"),
                                              ("agentharm", "hermes"))
                   for arm in ("benign", "risk")}


class StratifiedTaskSampler(Sampler[int]):
    def __init__(self, dataframe, *, seed: int = 267):
        self.seed = seed
        self.epoch = 0
        self.pools = defaultdict(list)
        for index, row in enumerate(dataframe):
            if row["split"] != "train" or row["policy_snapshot"] != "step:0":
                raise ValueError("sampler received non-train or stale task")
            key = (row["benchmark"], row["harness_name"], row["arm"])
            self.pools[key].append((int(index), row["source_family_id"], row["task_id"]))
        if set(self.pools) != EXPECTED_STRATA or any(len(pool) < GROUPS_PER_STRATUM for pool in self.pools.values()):
            raise ValueError("sampler requires four source/harness strata with both arms and four cells per arm")

    def __len__(self):
        return len(EXPECTED_STRATA) * GROUPS_PER_STRATUM

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        result = []
        for stratum in sorted(self.pools):
            pool = self.pools[stratum]
            # Each pool is traversed before restarting; the shuffle changes
            # between cycles, but an epoch never contains duplicate task cells.
            start = epoch * GROUPS_PER_STRATUM
            chosen = []
            for position in range(start, start + GROUPS_PER_STRATUM):
                cycle, offset = divmod(position, len(pool))
                order = list(range(len(pool)))
                random.Random(f"{self.seed}:{stratum}:{cycle}").shuffle(order)
                chosen.append(pool[order[offset]][0])
            if len(set(chosen)) != GROUPS_PER_STRATUM:
                # A cycle boundary may select the same task twice. Shift the
                # second occurrence to the first unused task in the pool.
                used = set()
                for i, index in enumerate(chosen):
                    if index in used:
                        chosen[i] = next(item[0] for item in pool if item[0] not in used and item[0] not in chosen[i + 1:])
                    used.add(chosen[i])
            result.extend(chosen)
        random.Random(f"{self.seed}:batch:{epoch}").shuffle(result)
        return iter(result)
