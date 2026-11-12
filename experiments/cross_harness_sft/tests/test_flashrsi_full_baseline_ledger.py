from flashrsi_full_baseline_ledger import make_ledger


def test_full_matrix_preserves_cases_splits_and_unlisted():
    cases = [
        {"task_id": "workspace:user_task_1:clean", "semantic_task_family_id": "workspace:user_task_1"},
        {"task_id": "workspace:user_task_1:injection_task_1", "semantic_task_family_id": "workspace:user_task_1"},
        {"task_id": "slack:user_task_0:clean", "semantic_task_family_id": "slack:user_task_0"},
    ]
    splits = {"agentdojo:workspace:user_task_1": "test"}
    rows = make_ledger(cases, splits)
    assert len(rows) == 3
    assert [r["split"] for r in rows] == ["test", "test", "unlisted"]
    assert [r["arm"] for r in rows] == ["clean", "attack", "clean"]
    assert len({r["task_id"] for r in rows}) == len(rows)


def test_duplicate_case_ids_rejected():
    case = {"task_id": "workspace:user_task_1:clean", "semantic_task_family_id": "workspace:user_task_1"}
    import pytest
    with pytest.raises(ValueError):
        make_ledger([case, case], {})
