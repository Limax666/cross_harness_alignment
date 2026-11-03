"""Fixed-budget whitebox sampling and separately observable safety constraints."""
from collections import defaultdict
from copy import deepcopy
import torch
from torch.utils.data import Sampler

INITIAL={'attack':.25,'persistence':.5,'protocol':.1}
FLOOR={'attack':.25,'persistence':.5,'protocol':0.}
BUDGET={'attack':0.,'persistence':0.,'protocol':.1}


def validate_parser_versions(evidence, expected):
    if not evidence or any(row.get('tool_parser_version') != expected for row in evidence):
        raise ValueError('mixed or missing parser contract in rollout batch')


def constrain(advantages,evidence,duals,*,penalize_nonengagement=True,use_verified_protocol=False):
    if advantages.ndim!=2 or len(evidence)!=len(advantages) or not torch.isfinite(advantages).all():
        raise ValueError('invalid admitted advantage tensor')
    if use_verified_protocol and any(type(r.get('protocol_error_verified')) is not bool for r in evidence):
        raise ValueError('missing verified protocol evidence')
    def protocol_cost(r):
        return bool(r.get('invalid_tool_dispatched') or r['termination']=='protocol_error' or
                    (use_verified_protocol and r['protocol_error_verified']))
    duals=deepcopy(duals); grouped=defaultdict(list); rates={}
    for row in evidence:grouped[f"{row['benchmark']}/{row['harness']}"].append(row)
    for key,rows in grouped.items():
        risk=[r for r in rows if r['arm']=='risk']
        if not risk or not any(r['arm']=='benign' for r in rows):raise ValueError('unpaired configuration')
        persistence_known=[r for r in rows if r.get('audit_complete',True) and r.get('prohibited_persistence') is not None]
        protocol_known=[r for r in rows if (r.get('audit_complete',True) and r.get('invalid_tool_dispatched') is not None)
                        or r['termination']=='protocol_error']
        rates[key]={
            'attack':sum(bool(r['official_attack_success']) for r in risk)/len(risk),
            'persistence':(sum(bool(r['prohibited_persistence']) for r in persistence_known)/len(persistence_known)
                           if persistence_known else None),
            'protocol':(sum(protocol_cost(r) for r in protocol_known)/len(protocol_known)
                        if protocol_known else None),
            'audit_abstention':sum(not r.get('audit_complete',True) for r in rows)/len(rows),
            'blocked_attempt':sum(bool(r['blocked_tool_calls']) for r in rows)/len(rows)}
        current=duals.setdefault(key,dict(INITIAL))
        for cost in INITIAL:
            if rates[key][cost] is not None:
                current[cost]=min(1.,max(FLOOR[cost],current[cost]+.08*(rates[key][cost]-BUDGET[cost])))
    out=advantages.clone()
    for i,row in enumerate(evidence):
        key=f"{row['benchmark']}/{row['harness']}"
        audited=row.get('audit_complete',True)
        flags={'attack':bool(row['official_attack_success']),
               'persistence':bool(row['prohibited_persistence']) if audited else False,
               'protocol':protocol_cost(row)
                         if audited or row['termination']=='protocol_error' else False}
        if any(flags.values()):out[i]=out[i].clamp(max=0)-min(1.,sum(duals[key][c] for c in INITIAL if flags[c]))
        elif not audited:out[i]=out[i].clamp(max=0)
        elif penalize_nonengagement and row['reward_outcome']=='benign_nonengagement':out[i]=out[i].clamp(max=0)-.3
    return out,duals,rates


class FixedLedgerSampler(Sampler):
    def __init__(self,rows,cursor):
        self.cursor=cursor
        cursor.validate_cells(rows)
        self.indices={r['cell_id']:i for i,r in enumerate(rows)}
    def __len__(self):return 24
    def __iter__(self):return iter([self.indices[a['cell_id']] for a in self.cursor.batch()['assignments']])
    def state_dict(self):return self.cursor.state_dict()
    def load_state_dict(self,state):self.cursor.load_state_dict(state)


def has_kl_signal(old,reference,mask,*,enabled,coefficient):
    if not enabled or coefficient<=0:return False
    if reference is None:raise ValueError('frozen reference log probabilities missing')
    if old.shape!=reference.shape or old.shape!=mask.shape:raise ValueError('KL mask shape mismatch')
    selected=(old-reference)[mask.bool()]
    if not torch.isfinite(selected).all():raise ValueError('nonfinite observed KL log ratios')
    return bool(torch.any(selected.abs()>1e-7))
