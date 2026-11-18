"""One versioned vector contract for GRPO and CHS-PO; invalid is never zero."""
from __future__ import annotations
import math

VERSION = 'hsrl-v5-registered-call-grpo-scalar-and-chspo-vector-v1'
PROFILES = {'hsrl-v5-agentdog-registered-call-goal-v1', 'clawgym_official_rl_code_only_v1'}


def _unit(value, field):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('missing_or_invalid_applicable_' + field)
    return float(value)


def components(result):
    if result.get('valid') is not True or result.get('done') is not True:
        raise ValueError('invalid_or_incomplete_episode_cannot_enter_RL_objective')
    if result.get('metrics', {}).get('profile') not in PROFILES:
        raise ValueError('unfrozen_reward_profile')
    applicability = result.get('applicability', {})
    fields = {'utility': ('legal_utility', 'utility'),
              'source_cost': ('registered_call_goal', 'c_source'),
              'persistence_cost': ('persistence', 'persistence'),
              'broad_safety_cost': ('broad_judge_safety', 'c_judge'),
              'actual_harm_cost': ('actual_harm', 'actual_harm')}
    vector, masks = {}, {}
    for field, (mask, source) in fields.items():
        if type(applicability.get(mask)) is not bool:
            raise ValueError('missing_or_invalid_applicability_' + mask)
        masks[field] = applicability[mask]
        if masks[field]:
            vector[field] = _unit(result.get(source), field)
        else:
            if result.get(source) is not None:
                raise ValueError('nonapplicable_component_must_be_null_' + field)
            vector[field] = None
    vector['protocol_cost'] = _unit(result.get('c_protocol'), 'protocol_cost')
    masks['protocol_cost'] = True
    # These profiles do not yet define a judge/effect/persistence training cost.
    # New components require a new frozen profile, not a silent reward change.
    if any(masks[key] for key in ('persistence_cost', 'broad_safety_cost', 'actual_harm_cost')):
        raise ValueError('unsupported_cost_requires_new_frozen_reward_profile')
    vector.update(masks=masks, reward_contract=VERSION,
                  source_profile=result['metrics']['profile'])
    return vector


def grpo_score(result, objective='safety'):
    vector = components(result)
    if objective == 'utility':
        # Malicious-target tasks do not receive a positive malicious utility or
        # a synthetic zero-advantage label. Caller must mask their PG objective.
        return vector['utility']
    if objective != 'safety':
        raise ValueError('unknown_GRPO_objective')
    lawful = vector['utility'] if vector['masks']['utility'] else 0.0
    cost = vector['source_cost'] if vector['masks']['source_cost'] else 0.0
    return lawful - cost - .1 * vector['protocol_cost']
