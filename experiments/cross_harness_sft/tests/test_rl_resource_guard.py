from rl_resource_guard import select_idle_pair


def test_select_idle_pair_requires_empty_memory_and_no_compute_process():
    used = {0: 20, 1: 80, 2: 20, 3: 19}
    uuids = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    assert select_idle_pair((0, 2, 3), used, uuids, {'a'}) == (2, 3)
    assert select_idle_pair((0, 2, 3), used, uuids, {'a', 'c'}) is None
    assert select_idle_pair((0, 2, 3), {**used, 2: 256}, uuids, set()) == (0, 3)


def test_select_idle_pair_is_deterministic_and_rejects_duplicate_candidates():
    used = {0: 1, 2: 1, 3: 1}
    uuids = {0: 'a', 2: 'c', 3: 'd'}
    assert select_idle_pair((3, 2, 0), used, uuids, set()) == (3, 2)
    import pytest
    with pytest.raises(ValueError, match='distinct'):
        select_idle_pair((0, 0, 3), used, uuids, set())
