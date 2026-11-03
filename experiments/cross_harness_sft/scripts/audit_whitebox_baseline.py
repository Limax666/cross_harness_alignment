#!/usr/bin/env python3
"""Audit frozen-policy rollout evidence; thresholds must be supplied before collection."""
import argparse
import collections
import gzip
import json
from pathlib import Path


def audit(records, min_valid, max_timeout, min_per_group=4):
    failures=[]; groups=collections.defaultdict(list); families=collections.defaultdict(set)
    if not 0 <= min_valid <= 1 or not 0 <= max_timeout <= 1:
        raise ValueError('thresholds must be in [0,1]')
    if min_per_group < 4:
        raise ValueError('at least four independent rollouts per group are required')
    for record in records:
        e=record['evidence']; split=e.get('split'); harness=e.get('harness'); arm=e.get('arm')
        if split not in ('train','validation') or not harness or arm not in ('benign','risk'):
            failures.append('missing or invalid split/harness/arm'); continue
        if e.get('policy_snapshot') != 'step:0': failures.append('policy snapshot is not frozen SFT step:0')
        if not all(e.get(k) is True for k in ('reset_verified','final_state_verified','audit_complete')):
            failures.append('missing reset, final state, or audit evidence')
        if not e.get('source_family_id'): failures.append('missing family identity')
        else: families[split].add(e['source_family_id'])
        groups[(split,harness,arm)].append(e)
    for split in ('train','validation'):
        if not families[split]: failures.append(f'missing {split} evidence')
    if families['train'] & families['validation']: failures.append('train and validation family overlap')
    for harness in {k[1] for k in groups}:
        for split in ('train','validation'):
            for arm in ('benign','risk'):
                if (split,harness,arm) not in groups:
                    failures.append(f'missing {split}/{harness}/{arm} group')
    summary=[]
    for (split,harness,arm), rows in sorted(groups.items()):
        n=len(rows); valid=sum(bool(x.get('valid_tool_dispatches',0)) for x in rows)/n
        timeout=sum(x.get('termination')=='model_timeout' for x in rows)/n
        summary.append(dict(split=split,harness=harness,arm=arm,attempted=n,valid_tool_rate=valid,model_timeout_rate=timeout))
        if n < min_per_group: failures.append(f'{split}/{harness}/{arm} has only {n} rollouts')
        if valid < min_valid: failures.append(f'{split}/{harness}/{arm} valid tool rate {valid:.3f} < {min_valid}')
        if timeout > max_timeout: failures.append(f'{split}/{harness}/{arm} timeout rate {timeout:.3f} > {max_timeout}')
    return dict(passed=not failures,failures=sorted(set(failures)),groups=summary,thresholds=dict(min_valid=min_valid,max_timeout=max_timeout,min_per_group=min_per_group),family_counts={k:len(v) for k,v in families.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('files',nargs='+',type=Path);p.add_argument('--min-valid',type=float,required=True);p.add_argument('--max-timeout',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    records=[]
    for path in a.files:
        with gzip.open(path,'rt') as f: records.extend(json.loads(line) for line in f if line.strip())
    result=audit(records,a.min_valid,a.max_timeout);result['sources']=[str(p) for p in a.files]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['passed'] else 2

if __name__=='__main__': raise SystemExit(main())
