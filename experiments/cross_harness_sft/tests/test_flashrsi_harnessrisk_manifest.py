import json
from pathlib import Path
import pytest
from flashrsi_harnessrisk_manifest import build_manifest


def case(path, case_id, phase='runtime'):
    path.write_text(json.dumps({'case_id':case_id,'phase':phase,'task':{'user_messages':['first','second']},
                                'world':{},'oracle_scorer':{}}))


def test_inventory_hashes_full_cases_and_preserves_order(tmp_path):
    case(tmp_path/'b.json','b');case(tmp_path/'a.json','a')
    x=build_manifest(tmp_path,expected_count=2)
    assert [r['case_id'] for r in x['cases']]==['a','b']
    assert all(len(r['sha256'])==64 and r['owner_turns']==2 for r in x['cases'])
    assert x['full_evaluation_count']==2


def test_inventory_rejects_missing_or_duplicate_identity(tmp_path):
    case(tmp_path/'a.json','b')
    with pytest.raises(ValueError):build_manifest(tmp_path,expected_count=1)
    with pytest.raises(ValueError):build_manifest(tmp_path,expected_count=128)
