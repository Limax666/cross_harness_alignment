from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rl_group_sampler import TaskCell, allocate_groups


def _cells():
    return [TaskCell("trainbench", f"f-{arm}", harness, arm, f"t-{arm}", "rev1", "train")
            for harness in ("hermes", "nanobot", "openclaw") for arm in ("benign", "risk")]


def test_equal_harness_arm_allocation_and_seed_reproducibility():
    groups = allocate_groups(_cells(), groups_per_stratum=2, group_size=4, seed=42)
    assert groups == allocate_groups(_cells(), groups_per_stratum=2, group_size=4, seed=42)
    assert len(groups) == 12
    assert len({(g.cell.harness, g.cell.arm) for g in groups}) == 6
    assert all(len(set(g.seeds)) == 4 for g in groups)


def test_heldout_family_is_rejected():
    cells = _cells()
    cells[0] = TaskCell("trainbench", "f-benign", "hermes", "benign", "t-benign", "rev1", "test")
    try:
        allocate_groups(cells, groups_per_stratum=1, group_size=2, seed=1)
    except ValueError as exc:
        assert "train-family" in str(exc)
    else:
        raise AssertionError("test split was admitted")


def test_missing_risk_arm_is_rejected():
    try:
        allocate_groups([c for c in _cells() if not (c.harness == "hermes" and c.arm == "risk")],
                        groups_per_stratum=1, group_size=2, seed=1)
    except ValueError as exc:
        assert "both benign and risk" in str(exc)
    else:
        raise AssertionError("unpaired harness was admitted")
