"""CHS-PO v1 batch arithmetic for admitted single-configuration READY14K groups.

The collector owns episode authentication, per-slot retry accounting, reset and
cleanup receipts. Extra audit fields must come from the trusted evaluator.
Commit next_duals with the resulting optimizer checkpoint, never on collection.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from statistics import fmean, pstdev

from hsrl_v3.components import config_name
from hsrl_v5.io import digest
from hsrl_v5.reward import components
from hsrl_v5.sampler import OA18

# name -> (initial multiplier, lower bound, budget), per research protocol.
CONSTRAINTS = {'attack': (.25, .25, 0.), 'persistence': (.50, .50, 0.),
               'protocol': (.10, 0., .10)}


@dataclass(frozen=True)
class Audit:
    protocol_executed: bool | None
    legal_nonparticipation: bool | None
    complete: bool

    def __post_init__(self):
        if type(self.complete) is not bool:
            raise ValueError('invalid audit completeness')
        for field in (self.protocol_executed, self.legal_nonparticipation):
            if field is not None and type(field) is not bool:
                raise ValueError('invalid audit verdict')
        if self.complete and self.protocol_executed is None:
            raise ValueError('complete audit requires a protocol verdict')


def prepare_update(schedule, episodes, audits, *, policy_version, duals, groups_per_source=8):
    """Compute slot-ordered advantages with this batch's projected multipliers."""
    groups = schedule['groups']
    if type(groups_per_source) is not int or groups_per_source not in (1, 8):
        raise ValueError('expected 1 or 8 complete G8 groups per source')
    n_groups = 2 * groups_per_source
    if (schedule['G'] != 8 or len(episodes) != n_groups * 8 or len(audits) != n_groups * 8
            or Counter(g['source'] for g in groups) != {'clawgym': groups_per_source, 'agentdog': groups_per_source}
            or len({(g['source'], g['id']) for g in groups}) != n_groups
            or any(g['G'] != 8 for g in groups) or tuple(schedule['configuration']) not in OA18):
        raise ValueError(f'expected {n_groups} distinct scheduled G8 groups, balanced by source')
    configuration = config_name(schedule['configuration'])
    measurements = defaultdict(lambda: {name: [] for name in CONSTRAINTS})
    advantages = []
    costs_by_slot = []
    strata = []
    zero_variance_groups = 0
    for group_index, group in enumerate(groups):
        source, task = group['source'], group['id']
        stratum = source + '/' + configuration
        utilities = []
        for slot in range(group_index * 8, (group_index + 1) * 8):
            row, audit = episodes[slot], audits[slot]
            if (row.get('valid') is not True or row.get('policy_version') != policy_version
                    or row.get('configuration') != configuration or row.get('key') != [source, task]
                    or type(row.get('slot')) is not int or row['slot'] != slot):
                raise ValueError('invalid episode identity, policy version or slot')
            v = components(row['source_result'])
            profile = {'clawgym': 'clawgym_official_rl_code_only_v1',
                       'agentdog': 'hsrl-v5-agentdog-registered-call-goal-v1'}[source]
            if v['source_profile'] != profile or not isinstance(audit, Audit):
                raise ValueError('source profile or trusted audit mismatch')
            if audit.legal_nonparticipation and v['utility'] is None:
                raise ValueError('nonparticipation penalty requires an applicable lawful goal')
            if audit.complete and v['utility'] is not None and audit.legal_nonparticipation is None:
                raise ValueError('lawful participation verdict missing')
            utilities.append(v['utility'])
            # c_protocol also counts invalid or blocked proposals. Use only
            # the separate executed-protocol verdict for this constraint.
            costs = {'attack': v['source_cost'], 'persistence': v['persistence_cost'],
                     'protocol': audit.protocol_executed}
            for channel, observation in costs.items():
                if observation is not None:
                    measurements[stratum][channel].append(float(observation))
            costs_by_slot.append(costs)
            strata.append(stratum)
        known = [u for u in utilities if u is not None]
        if known and len(known) != 8:
            raise ValueError('utility applicability changed within a task-condition')
        if known:
            center, scale = fmean(known), pstdev(known)
            zero_variance_groups += int(scale == 0)
            advantages.extend((u - center) / (scale + 1e-6) for u in known)
        else:
            advantages.extend([0.] * 8)

    next_duals = deepcopy(duals)
    observations = {}
    for key in set(strata):
        multipliers = next_duals.setdefault(key, {name: c[0] for name, c in CONSTRAINTS.items()})
        if set(multipliers) != set(CONSTRAINTS):
            raise ValueError('incomplete multiplier checkpoint')
        observations[key] = {}
        for name, (_, lower, budget) in CONSTRAINTS.items():
            old = multipliers[name]
            if type(old) not in (float, int) or not isfinite(old) or not lower <= old <= 1:
                raise ValueError('invalid multiplier checkpoint')
            measured = measurements[key][name]
            rate = fmean(measured) if measured else None
            observations[key][name] = {'count': len(measured), 'rate': rate}
            if rate is not None:
                multipliers[name] = min(1., max(lower, old + .08 * (rate - budget)))

    for i, (key, costs, audit) in enumerate(zip(strata, costs_by_slot, audits)):
        penalty = sum(next_duals[key][name] * cost for name, cost in costs.items() if cost is not None)
        if any(cost for cost in costs.values()):
            advantages[i] = min(advantages[i], 0.) - min(1., penalty)
        elif not audit.complete:
            advantages[i] = min(advantages[i], 0.)
        elif audit.legal_nonparticipation:
            advantages[i] = min(advantages[i], 0.) - .3
    prepared = dict(advantages=advantages, observations=observations, next_duals=next_duals,
                    zero_variance_utility_groups=zero_variance_groups,
                    audit_abstentions=sum(not a.complete for a in audits))
    if groups_per_source == 1:
        prepared['episode_sha256'] = [digest(ep) for ep in episodes]
    return prepared


def optimize_admitted_batch(actor, optimizer, schedule, episodes, audits, *, policy_version, duals, lr):
    """Run one in-memory CHS-PO step after admission, with one global token reduction.

    A caller must durably commit actor/master weights, moments, step, multipliers,
    RNG and the sampling journal together before admitting the next batch. Discard
    the in-memory actor and optimizer after any exception during the update.
    """
    from hsrl_a6000_fullparam import backward_segment
    from hsrl_v3.policy import LOGPROB_CONTRACT
    from hsrl_v5.reward import VERSION
    prepared = prepare_update(schedule, episodes, audits, policy_version=policy_version, duals=duals)
    parameters = dict(actor.named_parameters())
    if (set(optimizer.params) != set(parameters) or optimizer.gradients
            or any(optimizer.params[n] is not p or not p.requires_grad for n, p in parameters.items())):
        raise ValueError('optimizer must own every actor parameter and start with no pending gradients')
    token_count = 0
    vocab = actor.config.text_config.vocab_size
    for ep in episodes:
        if ep.get('reward_contract') != VERSION or not ep.get('segments'):
            raise ValueError('missing native action segments or reward identity')
        episode_tokens = 0
        for seg in ep['segments']:
            prompt, response = seg['prompt_ids'], seg['response_ids']
            old, ref, mask = seg['old_logprobs'], seg['reference_logprobs'], seg['action_mask']
            if (seg['policy_version'] != policy_version or seg['logprob_contract'] != LOGPROB_CONTRACT
                    or not prompt or not response or len(response) > 2048 or len(prompt)+len(response) > 16384
                    or not (len(response) == len(old) == len(ref) == len(mask))
                    or any(type(v) is not int or not 0 <= v < vocab for ids in (prompt,response) for v in ids)
                    or any(type(x) not in (float,int) or not isfinite(x) or x > 0 for col in (old,ref) for x in col)
                    or any(type(x) is not bool for x in mask)):
                raise ValueError('stale, invalid or unscored action segment')
            token_count += sum(mask)
            episode_tokens += len(response)
        if episode_tokens > 8192 or len(ep['segments']) > 12 or not any(any(s['action_mask']) for s in ep['segments']):
            raise ValueError('invalid episode token coverage or budget')

    loss = 0.
    for ep, advantage in zip(episodes, prepared['advantages']):
        for seg in ep['segments']:
            loss += backward_segment(actor, seg, version=policy_version, advantage=advantage,
                                     pg_tokens=token_count, kl_tokens=token_count, pg_enabled=True)
    expected = {n for n in parameters if n.startswith(('model.language_model.', 'lm_head.'))}
    if not expected or set(optimizer.gradients) != expected:
        raise RuntimeError('incomplete text-policy gradients')
    step = optimizer.step(lr)
    return {**prepared, 'optimizer': step, 'action_tokens': token_count, 'loss': loss}
