import json
import pytest
from rl_curriculum_schedule import make_snapshot,curriculum_cohorts

def test_priority_train_only_and_coverage(tmp_path):
    families={f'agentdojo:workspace:user_task_{i}' for i in range(18)}
    p=tmp_path/'metrics.jsonl';p.write_text(json.dumps({'record_type':'metrics','phase':'train','rewards_by_cell':{'agentdojo/cfg/workspace:user_task_0:clean':[1,1,0,1]}})+'\n')
    snap=make_snapshot([p],families);assert snap['family_weights']['agentdojo:workspace:user_task_0']==3
    rounds=curriculum_cohorts(families,10,267,snap)
    assert rounds==curriculum_cohorts(families,10,267,snap) and all(len(x)==len(set(x))==12 for x in rounds)
    assert set.union(*map(set,rounds))==families
    assert snap['family_observations']['agentdojo:workspace:user_task_0']==1
    p.write_text(json.dumps({'record_type':'metrics','phase':'train','rewards_by_cell':{'agentdojo/cfg/workspace:user_task_100:clean':[1,1,0,1]}})+'\n')
    with pytest.raises(ValueError,match='non-train'):make_snapshot([p],families)

def test_rejects_nonfinite_and_unknown_weights(tmp_path):
    families={f'f{i}' for i in range(18)};snap={'family_weights':{f:1.5 for f in families}}
    snap['family_weights']['f0']=float('nan')
    with pytest.raises(ValueError):curriculum_cohorts(families,3,1,snap)

def test_custom_ledger_requires_provenance_and_preserves_k2():
    from rl_whitebox_harness import TRAIN_CONFIGS,config_name,build_whitebox_ledger
    rows=[]
    for f in range(18):
        for arm in ('benign','risk'):
            for config in TRAIN_CONFIGS:
                name=config_name(config)
                rows.append(dict(benchmark='agentdojo',source_family_id=f'family-{f}',arm=arm,harness_name=name,split='train',task_id=f'task-{f}-{arm}',cell_id=f'{name}/{f}/{arm}',task_semantic_sha256=f'{f+1:064x}',attack_payload_sha256=('a' if arm=='risk' else '0')*64,initial_state_sha256=f'{f+20:064x}',fixture_revision='pinned',semantic_fixture_seed=11))
    families={f'family-{f}' for f in range(18)};snap={'family_weights':{f:1.5 for f in families}}
    schedule=curriculum_cohorts(families,2,267,snap)
    with pytest.raises(ValueError,match='provenance'):build_whitebox_ledger(rows,iterations=4,family_schedule=schedule)
    ledger=build_whitebox_ledger(rows,iterations=4,family_schedule=schedule,schedule_metadata={'version':'candidate'})
    for x,y in zip(ledger['rounds'][::2],ledger['rounds'][1::2]):
        assert {r['assignment_id'] for r in x['assignments']}=={r['assignment_id'] for r in y['assignments']}
        assert len(x['assignments'])==len(y['assignments'])==24
        assert all(a['config']!=b['config'] for a,b in zip(x['assignments'],y['assignments']))

def test_frozen_curriculum_checks_all_three_artifacts(tmp_path):
    import hashlib
    from rl_curriculum_schedule import verify_frozen_curriculum
    from rl_paired_assignment import digest
    snap={'family_weights':{'a':1.5}};sh=hashlib.sha256(json.dumps(snap,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    ledger={'family_schedule_metadata':{'snapshot_sha256':sh},'rounds':[]};ledger['ledger_sha256']=digest(ledger)
    (tmp_path/'family_priority_snapshot.json').write_text(json.dumps(snap));(tmp_path/'assignment_ledger.json').write_text(json.dumps(ledger));(tmp_path/'train.parquet').write_bytes(b'fixture')
    (tmp_path/'manifest.json').write_text(json.dumps({'data_sha256':hashlib.sha256(b'fixture').hexdigest(),'snapshot_sha256':sh,'ledger_sha256':ledger['ledger_sha256']}))
    assert verify_frozen_curriculum(tmp_path)
    snap['family_weights']['a']=2;(tmp_path/'family_priority_snapshot.json').write_text(json.dumps(snap))
    with pytest.raises(ValueError,match='snapshot'):verify_frozen_curriculum(tmp_path)
