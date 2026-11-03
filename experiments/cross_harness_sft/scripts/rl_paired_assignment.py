"""Immutable exact-task K=2 assignment; no model or teacher replay is sampled here.

Only fully attested, train-only cells may form the ledger. This module is
independent of the historical 2B sampler and does not qualify a runtime merely
because its display name appears in an input row.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import defaultdict, deque

HARNESSES = ("codex", "claude_code", "hermes")
BENCHMARKS = ("agentdojo", "agentharm")
ARMS = ("benign", "risk")
SEMANTIC_FIELDS = ("task_id", "task_semantic_sha256", "attack_payload_sha256",
                   "initial_state_sha256", "fixture_revision", "semantic_fixture_seed")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def cell_identity_digest(cells):
    # Prompt/tool payloads may be Arrow/numpy objects. Their independent source
    # and live-prompt audit belongs to the environment admission gate. Here we
    # bind the exact row ordering and attested semantic identities used by indices.
    keys = SEMANTIC_FIELDS + ("cell_id", "benchmark", "source_family_id", "arm",
                              "harness_name", "split")
    return digest([{key: row.get(key) for key in keys} for row in cells])


def _qualified_pairs(cells):
    """Return complete semantic-task intersections, never fill a missing arm."""
    families = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    ids = set()
    for index, row in enumerate(cells):
        if row.get("split") != "train":
            raise ValueError("paired assignment accepts only train families")
        if row.get("benchmark") not in BENCHMARKS or row.get("harness_name") not in HARNESSES:
            raise ValueError("unregistered benchmark/harness in train ledger")
        if row.get("arm") not in ARMS or not row.get("source_family_id"):
            raise ValueError("missing family/arm identity")
        if not row.get("cell_id") or row["cell_id"] in ids:
            raise ValueError("missing or duplicate task cell")
        ids.add(row["cell_id"])
        if any(not isinstance(row.get(key), str) or not row[key] for key in SEMANTIC_FIELDS[:-1]):
            raise ValueError("missing independently attested semantic task contract")
        if type(row.get("semantic_fixture_seed")) is not int or row["semantic_fixture_seed"] < 0:
            raise ValueError("missing independently attested fixture seed")
        for key in SEMANTIC_FIELDS[1:4]:
            value = row[key]
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"invalid {key}")
        family = (row["benchmark"], row["source_family_id"])
        signature = tuple(row[key] for key in SEMANTIC_FIELDS)
        slot = families[family][row["arm"]][signature]
        if row["harness_name"] in slot:
            raise ValueError("ambiguous duplicate semantic task/runtime")
        slot[row["harness_name"]] = index
    eligible = defaultdict(dict)
    excluded = []
    for (benchmark, family), arms in sorted(families.items()):
        matches = {}
        for arm in ARMS:
            complete = [(signature, variants) for signature, variants in arms[arm].items()
                        if set(variants) == set(HARNESSES)]
            if complete:
                matches[arm] = sorted(complete)
        if set(matches) == set(ARMS):
            eligible[benchmark][family] = matches
        else:
            excluded.append({"benchmark": benchmark, "family_id": family,
                             "reason": "missing_exact_task_cross_runtime_arm"})
    if any(len(eligible[b]) < 6 for b in BENCHMARKS):
        sizes = {b: len(eligible[b]) for b in BENCHMARKS}
        raise ValueError(f"need six exact paired train families per benchmark; got {sizes}")
    return eligible, excluded


def _coverage_schedule(options, blocks, seed, benchmark, width=6):
    queue = deque()
    cycle = 0
    cohorts = []
    for _ in range(blocks):
        selected = []
        while len(selected) < width:
            if not queue:
                order = sorted(options)
                random.Random(f"{seed}:{benchmark}:coverage:{cycle}").shuffle(order)
                # Retain a full permutation, but postpone entries that would
                # duplicate the current cohort across a cycle boundary.
                queue.extend([x for x in order if x not in selected] +
                             [x for x in order if x in selected])
                cycle += 1
            selected.append(queue.popleft())
        cohorts.append(selected)
    return cohorts


def build_assignment_ledger(cells, *, iterations=300, seed=267):
    if type(iterations) is not int or iterations < 2 or iterations % 2:
        raise ValueError("K=2 requires a positive even iteration budget")
    if type(seed) is not int or seed < 0:
        raise ValueError("invalid assignment seed")
    eligible, excluded = _qualified_pairs(cells)
    coverage = {b: _coverage_schedule(eligible[b], iterations // 2, seed, b)
                for b in BENCHMARKS}
    rounds = []
    for block in range(iterations // 2):
        phases = [[], []]
        for benchmark in BENCHMARKS:
            selected = coverage[benchmark][block]
            order = list(HARNESSES)
            random.Random(f"{seed}:{benchmark}:anchors:{block}").shuffle(order)
            for slot, family in enumerate(selected):
                anchor_index = slot // 2
                anchor = order[anchor_index]
                # Alternate directions in the fixed product order, independent
                # of the shuffled allocation order, to cover all directed pairs.
                partner = HARNESSES[(HARNESSES.index(anchor) +
                                     (1 if block % 2 == 0 else -1)) % 3]
                for arm in ARMS:
                    options = eligible[benchmark][family][arm]
                    rng = random.Random(f"{seed}:{benchmark}:{family}:{arm}:{block}")
                    signature, variants = rng.choice(options)
                    semantic = dict(zip(SEMANTIC_FIELDS, signature))
                    assignment_id = digest([seed, block, benchmark, family, arm, signature])
                    for phase, harness in enumerate((anchor, partner)):
                        cell_index = variants[harness]
                        phases[phase].append({
                            **semantic, "benchmark": benchmark, "source_family_id": family,
                            "arm": arm, "harness_name": harness,
                            "cell_id": cells[cell_index]["cell_id"], "cell_index": cell_index,
                            "assignment_id": assignment_id,
                            "fixture_seed": semantic["semantic_fixture_seed"],
                        })
        for phase, assignments in enumerate(phases):
            rounds.append({"iteration": 2 * block + phase + 1, "block_id": block,
                           "phase": phase, "groups": 24, "rollouts": 96,
                           "assignments": assignments})
    core = {"version": "9b-exact-task-k2-v1", "seed": seed, "group_size": 4,
            "iterations": iterations, "cells_sha256": cell_identity_digest(cells),
            "excluded_families": excluded, "rounds": rounds}
    return {**core, "ledger_sha256": digest(core)}


class AssignmentCursor:
    """Advance a fixed sampling round only after an admitted objective/no-op.

    Retrying an infrastructure failure does not call commit. A no-op advances
    the assignment phase but not Adam's update count. The trainer, rather than
    this cursor, owns gradient and independent admission checks.
    """
    def __init__(self, ledger):
        core = {k: v for k, v in ledger.items() if k != "ledger_sha256"}
        if digest(core) != ledger.get("ledger_sha256"):
            raise ValueError("assignment ledger hash mismatch")
        self.ledger = copy.deepcopy(ledger)
        self.completed_iterations = self.optimizer_updates = self.kl_only_updates = 0

    def batch(self):
        if self.completed_iterations >= self.ledger["iterations"]:
            raise StopIteration("fixed rollout budget exhausted")
        return copy.deepcopy(self.ledger["rounds"][self.completed_iterations])

    def validate_cells(self, cells):
        if cell_identity_digest(cells) != self.ledger["cells_sha256"]:
            raise ValueError("live dataset order/semantic identities differ from frozen ledger")

    def commit(self, *, optimizer_updated, kl_only=False):
        self.batch()
        if type(optimizer_updated) is not bool or type(kl_only) is not bool:
            raise ValueError("update counters require explicit booleans")
        if kl_only and not optimizer_updated:
            raise ValueError("KL-only update cannot be a no-op")
        self.completed_iterations += 1
        self.optimizer_updates += int(optimizer_updated)
        self.kl_only_updates += int(kl_only)

    def state_dict(self):
        return {"ledger_sha256": self.ledger["ledger_sha256"],
                "completed_iterations": self.completed_iterations,
                "optimizer_updates": self.optimizer_updates,
                "kl_only_updates": self.kl_only_updates}

    def load_state_dict(self, state):
        if state.get("ledger_sha256") != self.ledger["ledger_sha256"]:
            raise ValueError("cannot resume a different assignment ledger")
        keys = ("completed_iterations", "optimizer_updates", "kl_only_updates")
        if any(type(state.get(key)) is not int for key in keys):
            raise ValueError("invalid cursor counters")
        done, updates, kl = (state[key] for key in keys)
        if not 0 <= kl <= updates <= done <= self.ledger["iterations"]:
            raise ValueError("cursor counters exceed admitted iteration budget")
        self.completed_iterations, self.optimizer_updates, self.kl_only_updates = done, updates, kl
