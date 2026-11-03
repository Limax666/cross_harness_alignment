import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rl_informative_sampler import InformativeTaskSampler, priority_snapshot
from rl_multiharness_stratified_sampler import EXPECTED_STRATA


def frame():
    rows = []
    for benchmark, harness, arm in sorted(EXPECTED_STRATA):
        for i in range(8):
            rows.append({"benchmark": benchmark, "harness_name": harness, "arm": arm,
                         "split": "train", "policy_snapshot": "step:0",
                         "cell_id": f"{benchmark}/{harness}/{arm}/{i}",
                         "source_family_id": f"family-{i}", "task_id": f"task-{i}"})
    return rows


def test_balanced_deterministic_and_no_duplicate_cells():
    data = frame()
    snapshot = {"last_step": 140, "weights": {row["cell_id"]: 8.0 if row["cell_id"].endswith("/0") else 1.0
                                              for row in data}, "unseen_weight": 2.0}
    first = InformativeTaskSampler(data, snapshot)
    second = InformativeTaskSampler(data, snapshot)
    for _ in range(12):
        selected = list(first)
        assert selected == list(second)
        assert len(selected) == len(set(selected)) == 32
        strata = {(data[index]["benchmark"], data[index]["harness_name"], data[index]["arm"])
                  for index in selected}
        assert strata == EXPECTED_STRATA
        assert all(sum((data[index]["benchmark"], data[index]["harness_name"], data[index]["arm"]) == key
                       for index in selected) == 4 for key in EXPECTED_STRATA)


def test_snapshot_rejects_uncommitted_checkpoint_step(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"record_type":"metrics","phase":"train","step":139,"rewards_by_cell":{}}\n')
    try:
        priority_snapshot(path, last_step=140)
    except ValueError as error:
        assert "checkpoint" in str(error)
    else:
        raise AssertionError("accepted a future checkpoint step")
