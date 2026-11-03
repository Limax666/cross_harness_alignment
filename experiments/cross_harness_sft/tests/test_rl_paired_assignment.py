"""Pin the task identity, budget and recovery rules of the 9B K=2 design."""
import copy
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rl_paired_assignment import AssignmentCursor, build_assignment_ledger


def cells():
    rows = []
    for benchmark in ("agentdojo", "agentharm"):
        for family in range(8):
            for arm in ("benign", "risk"):
                for harness in ("codex", "claude_code", "hermes"):
                    identity = f"{benchmark}/{family}/{arm}"
                    rows.append(dict(
                        benchmark=benchmark, source_family_id=f"{benchmark}:{family}",
                        arm=arm, harness_name=harness, split="train",
                        task_id=identity, cell_id=f"{harness}/{identity}",
                        task_semantic_sha256=f"{family + 1:064x}",
                        attack_payload_sha256=("a" if arm == "risk" else "0") * 64,
                        initial_state_sha256=f"{family + 30:064x}", fixture_revision="fixed-v1",
                        semantic_fixture_seed=73,
                    ))
    return rows


def test_rounds_balance_and_reuse_exact_tasks_with_different_runtime():
    ledger = build_assignment_ledger(cells(), iterations=8, seed=267)
    assert ledger == build_assignment_ledger(cells(), iterations=8, seed=267)
    directed = set()
    for first, second in zip(ledger["rounds"][::2], ledger["rounds"][1::2]):
        assert len(first["assignments"]) == len(second["assignments"]) == 24
        assert first["rollouts"] == second["rollouts"] == 96
        strata = Counter((r["benchmark"], r["harness_name"], r["arm"])
                         for r in first["assignments"])
        assert len(strata) == 12 and set(strata.values()) == {2}
        for a, b in zip(first["assignments"], second["assignments"]):
            for field in ("task_id", "source_family_id", "arm", "task_semantic_sha256",
                          "attack_payload_sha256", "initial_state_sha256", "fixture_seed"):
                assert a[field] == b[field], field
            assert a["harness_name"] != b["harness_name"]
            directed.add((a["harness_name"], b["harness_name"]))
    assert len(directed) == 6


@pytest.mark.parametrize("field,value", [("split", "validation"),
    ("attack_payload_sha256", "b" * 64), ("initial_state_sha256", "c" * 64)])
def test_rejects_changed_split_or_semantic_contract(field, value):
    rows = cells()
    # Fewer than six admissible families must not silently become K=1.
    rows = [r for r in rows if int(r["source_family_id"].split(":")[-1]) < 6]
    rows[0][field] = value
    with pytest.raises(ValueError):
        build_assignment_ledger(rows, iterations=2)


def test_noop_and_kl_only_keep_fixed_ledger_and_survive_resume():
    ledger = build_assignment_ledger(cells(), iterations=4)
    cursor = AssignmentCursor(ledger)
    first = cursor.batch()
    cursor.commit(optimizer_updated=False)
    second = cursor.batch()
    assert second["phase"] == 1 and second["block_id"] == first["block_id"]
    assert cursor.state_dict()["optimizer_updates"] == 0
    restored = AssignmentCursor(ledger)
    restored.load_state_dict(cursor.state_dict())
    assert restored.batch() == second
    restored.commit(optimizer_updated=True, kl_only=True)
    assert restored.state_dict()["optimizer_updates"] == 1
    assert restored.state_dict()["kl_only_updates"] == 1
    assert restored.batch()["iteration"] == 3


def test_resume_rejects_other_ledger_and_corrupt_counters():
    cursor = AssignmentCursor(build_assignment_ledger(cells(), iterations=4))
    state = cursor.state_dict()
    state["ledger_sha256"] = "x" * 64
    with pytest.raises(ValueError):
        cursor.load_state_dict(state)
    state = copy.deepcopy(cursor.state_dict())
    state["optimizer_updates"] = 8
    with pytest.raises(ValueError):
        cursor.load_state_dict(state)


def test_no_sampling_past_fixed_budget():
    cursor = AssignmentCursor(build_assignment_ledger(cells(), iterations=2))
    cursor.commit(optimizer_updated=False)
    cursor.commit(optimizer_updated=True)
    with pytest.raises(StopIteration):
        cursor.batch()


def test_nonzero_audited_fixture_seed_is_preserved():
    ledger = build_assignment_ledger(cells(), iterations=2)
    assert all(r["fixture_seed"] == 73 for step in ledger["rounds"] for r in step["assignments"])


def test_cross_runtime_fixture_seed_mismatch_is_not_paired():
    rows = [r for r in cells() if int(r["source_family_id"].split(":")[-1]) < 6]
    rows[0]["semantic_fixture_seed"] = 91
    with pytest.raises(ValueError):
        build_assignment_ledger(rows, iterations=2)


@pytest.mark.parametrize("n", [7, 8])
def test_cycle_boundary_preserves_every_family_once_per_cycle(n):
    rows = [r for r in cells() if int(r["source_family_id"].split(":")[-1]) < n]
    ledger = build_assignment_ledger(rows, iterations=16)
    sampled = []
    for step in ledger["rounds"][::2]:
        part = [r["source_family_id"] for r in step["assignments"]
                if r["benchmark"] == "agentdojo" and r["arm"] == "benign"]
        assert len(part) == len(set(part)) == 6
        sampled.extend(part)
    for start in range(0, len(sampled) - n + 1, n):
        assert len(set(sampled[start:start+n])) == n


def test_ledger_rejects_reordered_live_dataset_before_execution():
    rows = cells()
    cursor = AssignmentCursor(build_assignment_ledger(rows, iterations=2))
    cursor.validate_cells(rows)
    rows.reverse()
    with pytest.raises(ValueError):
        cursor.validate_cells(rows)


def test_nonsemantic_arrays_do_not_enter_identity_hash():
    import numpy as np
    rows = cells()
    first = build_assignment_ledger(rows, iterations=2)
    for row in rows:
        row["prompt"] = np.array([{"role": "user", "content": "not a fixture identity"}], dtype=object)
    assert build_assignment_ledger(rows, iterations=2) == first
