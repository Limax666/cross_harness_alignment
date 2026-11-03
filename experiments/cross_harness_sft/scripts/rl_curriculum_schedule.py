"""Frozen, train-only balanced coverage plus bounded difficulty prioritization."""
import hashlib,json,math,random
from collections import defaultdict
from pathlib import Path
from rl_paired_assignment import _coverage_schedule

def make_snapshot(metrics_paths, train_families):
    families=set(train_families); history=defaultdict(list); provenance=[]
    if not families:raise ValueError('empty train family whitelist')
    for p in map(Path,metrics_paths):
        raw=p.read_bytes();provenance.append({'path':str(p.resolve()),'sha256':hashlib.sha256(raw).hexdigest()})
        for line in raw.splitlines():
            row=json.loads(line)
            if row.get('phase')!='train' or row.get('record_type')!='metrics':continue
            for cell,scores in row.get('rewards_by_cell',{}).items():
                parts=cell.split('/',2)
                if len(parts)!=3 or parts[0]!='agentdojo' or not isinstance(scores,list) or len(scores)!=4:raise ValueError('invalid training evidence')
                family='agentdojo:'+parts[2].rpartition(':')[0]
                if family not in families:raise ValueError('non-train family in history')
                if any(type(x) not in (int,float) or not math.isfinite(x) for x in scores):raise ValueError('invalid group reward')
                history[family].append((len(set(scores))>1,any(x < .95 for x in scores)))
    weights={}
    for f in sorted(families):
        h=history[f]
        weights[f]=round(1+1.5*sum(x[0] for x in h)/len(h)+.5*sum(x[1] for x in h)/len(h),6) if h else 1.5
    return {'version':'train-coverage-difficulty-v1','provenance':provenance,'family_weights':weights,'family_observations':{f:len(history[f]) for f in sorted(families)},'formula':'1 + 1.5 * variable_group_rate + 0.5 * nonmax_group_rate; unseen 1.5','sampling':'8 deterministic coverage, 4 weighted without replacement; all G=4 rollouts retained'}

def curriculum_cohorts(families,blocks,seed,snapshot,width=12,coverage=8):
    families=set(families); weights=snapshot.get('family_weights',{})
    if set(weights)!=families or not 0<coverage<width<=len(families):raise ValueError('curriculum whitelist or width mismatch')
    if any(type(w) not in (int,float) or not math.isfinite(w) or not 1<=w<=3 for w in weights.values()):raise ValueError('invalid priority weight')
    core=_coverage_schedule(families,blocks,seed,'curriculum-coverage',width=coverage)
    result=[]
    for block,selected in enumerate(core):
        selected=list(selected); candidates=sorted(families-set(selected));rng=random.Random(f'{seed}:curriculum:{block}')
        for _ in range(width-coverage):
            pick=rng.choices(candidates,weights=[weights[x] for x in candidates],k=1)[0]
            selected.append(pick);candidates.remove(pick)
        result.append(selected)
    return result

def verify_frozen_curriculum(pool):
    """Require matching dataset, priority snapshot and ledger provenance."""
    pool=Path(pool);manifest=json.loads((pool/'manifest.json').read_text())
    data=(pool/'train.parquet').read_bytes()
    snap=json.loads((pool/'family_priority_snapshot.json').read_text())
    digest=hashlib.sha256(json.dumps(snap,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    ledger=json.loads((pool/'assignment_ledger.json').read_text())
    from rl_paired_assignment import digest as ledger_digest
    if hashlib.sha256(data).hexdigest()!=manifest.get('data_sha256'):raise ValueError('curriculum data hash mismatch')
    if digest!=manifest.get('snapshot_sha256') or ledger.get('family_schedule_metadata',{}).get('snapshot_sha256')!=digest:raise ValueError('curriculum priority snapshot hash mismatch')
    if ledger.get('ledger_sha256')!=manifest.get('ledger_sha256') or ledger_digest({k:v for k,v in ledger.items() if k!='ledger_sha256'})!=ledger['ledger_sha256']:raise ValueError('curriculum ledger hash mismatch')
    return True
