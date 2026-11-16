"""50/50 task groups; K2 is successive fresh-policy steps, not a 4+4 group."""
from __future__ import annotations
from collections import Counter
import hashlib
import random

OA18 = tuple(tuple(map(int, row)) for row in (
    '00000', '01111', '02222', '10011', '11122', '12200',
    '20102', '21210', '22021', '00221', '01002', '02110',
    '10120', '11201', '12012', '20212', '21020', '22101'))
QUOTAS = {'attack': 44, 'clean': 8, 'query_target': 4}
_ROLES = [key for key, count in QUOTAS.items() for _ in range(count)]
random.Random(20261116).shuffle(_ROLES)


def role_slots(block):
    if type(block) is not int or block < 0:
        raise ValueError('invalid_K_block')
    offset = (block % 7) * 8
    return tuple(_ROLES[offset:offset + 8])


def configuration_pair(block):
    if type(block) is not int or block < 0:
        raise ValueError('invalid_K_block')
    anchor = OA18[block % len(OA18)]
    peer = min((row for row in OA18 if row != anchor), key=lambda row: (sum(a != b for a, b in zip(anchor, row)), row))
    return anchor, peer


class BalancedSampler:
    """Host-side references; only Dataset.public() may render policy input."""
    def __init__(self, records, seed=20261116):
        records = list(records)
        if len(records) != 14000 or len({(r['source'], r['id']) for r in records}) != 14000:
            raise ValueError('release_requires_14000_unique_task_conditions')
        if any(r['split'] != 'train' or r.get('runtime_ready') is not True for r in records):
            raise ValueError('nontraining_or_unadmitted_sampler_record')
        self.pools = {'general': [], 'attack': [], 'clean': [], 'query_target': []}
        for row in records:
            if row['source'] == 'clawgym' and row['condition'] == 'benign':
                key = 'general'
            elif row['source'] == 'agentdog' and row['condition'] in QUOTAS:
                key = row['condition']
            else:
                raise ValueError('unexpected_mixed_pool_stratum')
            self.pools[key].append((row['source'], row['id']))
        if {key: len(rows) for key, rows in self.pools.items()} != {'general': 7000, 'attack': 5500, 'clean': 1000, 'query_target': 500}:
            raise ValueError('mixed_pool_does_not_match_frozen_role_quotas')
        for key, rows in self.pools.items():
            rows.sort(key=lambda item: hashlib.sha256(f'{seed}|{item[0]}|{item[1]}'.encode()).hexdigest())

    def block(self, index):
        configurations = configuration_pair(index)
        groups = [{'source': source, 'id': identity, 'G': 8} for source, identity in
                  [self.pools['general'][(index * 8 + offset) % 7000] for offset in range(8)]]
        before = Counter(_ROLES[:(index % 7) * 8])
        for role in role_slots(index):
            offset = (index // 7) * QUOTAS[role] + before[role]
            source, identity = self.pools[role][offset % len(self.pools[role])]
            before[role] += 1
            groups.append({'source': source, 'id': identity, 'G': 8})
        return {'K_block': index, 'groups': groups, 'configurations': configurations,
                'rollout_steps': (2 * index, 2 * index + 1), 'G': 8, 'K': 2,
                'episode_attempts_if_no_failures': 256,
                'must_reset_and_resample_current_policy_for_each_configuration': True}

    def step(self, index):
        if type(index) is not int or index < 0:
            raise ValueError('invalid_rollout_step')
        block, member = divmod(index, 2)
        result = self.block(block)
        return {'step': index, 'K_block': block, 'configuration': result['configurations'][member],
                'groups': result['groups'], 'G': 8, 'episode_attempts_if_no_failures': 128,
                'fresh_reset_and_current_policy_required': True}
