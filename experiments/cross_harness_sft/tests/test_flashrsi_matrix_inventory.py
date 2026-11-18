from flashrsi_matrix_inventory import build_matrix


def test_inventory_accounts_for_every_method_harness_and_source():
    rows = build_matrix()
    assert len(rows) == 7 * 13 * 7
    assert len({(r['harness'], r['method'], r['benchmark']) for r in rows}) == len(rows)
    assert {r['status'] for r in rows} == {'not_qualified'}
    assert len({r['method'] for r in rows}) == 13
    assert len({r['harness'] for r in rows}) == 7
