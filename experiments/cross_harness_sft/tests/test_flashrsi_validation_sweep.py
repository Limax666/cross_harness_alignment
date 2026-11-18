import pytest

from flashrsi_validation_sweep import validation_cases


def test_only_frozen_validation_and_registered_suites():
    manifest = {"cases": [
        {"task_id": "workspace:user_task_3:clean", "family": "workspace:user_task_3", "split": "validation", "seed": 0},
        {"task_id": "workspace:user_task_4:clean", "family": "workspace:user_task_4", "split": "test", "seed": 0},
    ]}
    assert [x["task_id"] for x in validation_cases(manifest)] == ["workspace:user_task_3:clean"]


def test_rejects_validation_duplicate_or_family_mismatch():
    row = {"task_id": "workspace:user_task_3:clean", "family": "workspace:user_task_3", "split": "validation", "seed": 0}
    with pytest.raises(ValueError):
        validation_cases({"cases": [row, row]})
    with pytest.raises(ValueError):
        validation_cases({"cases": [{**row, "family": "workspace:user_task_4"}]})
