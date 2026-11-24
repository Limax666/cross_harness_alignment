import json

import pytest

from hsrl_a6000_chspo import Audit
from hsrl_v5.agentdog_backend import PROFILE
from hsrl_v5.io import digest
from hsrl_v5.release import GRADER_SHA256
from hsrl_v5.runtime import VERSION


def put(path, value):
    path.write_text(json.dumps(value))


def dog(directory, *, legal=True, risk=False, utility=False):
    backend = directory / 'private/backend'
    backend.mkdir(parents=True)
    h = digest({})
    grade = dict(ok=True, grader_sha256=GRADER_SHA256, registered_call_goal=False,
                 source_utility=utility)
    put(backend/'private_grade_0.json', grade)
    put(backend/'private_final_state.json', {})
    put(backend/'provenance.json', dict(source='agentdog', source_id='fixture',
        grader_sha256=GRADER_SHA256, registered_call_goal_profile=PROFILE, initial_state_sha256=h))
    put(backend/'reset.json', dict(initial_state_sha256=h))
    result = dict(done=True, valid=True, c_protocol=0., c_source=0. if risk else None,
        utility=float(utility) if legal else None, persistence=None,
        applicability=dict(legal_utility=legal, registered_call_goal=risk, persistence=False,
                           actual_harm=False, broad_judge_safety=False),
        evidence=dict(initial_state_sha256=h, final_state_sha256=h, history_sha256=digest([])),
        metrics=dict(profile=PROFILE, canonical_actions=0, source_argument_errors=0,
                     original_source_utility=utility, registered_call_goal=False if risk else None))
    put(backend/'result.json', result)
    return dict(valid=True, key=['agentdog','fixture'], runtime_version=VERSION,
                actions=[], tool_steps=0, segments=[dict(turn=0,text='')], protocol_errors=[],
                stop_reason='completed', source_result=result)


def test_only_state_goal_with_no_attempt_is_confirmed(tmp_path):
    from a6000_vm.chspo_audit import audit_episode
    assert audit_episode(dog(tmp_path), tmp_path) == Audit(False, True, True)


def test_attack_conditions_do_not_turn_no_action_into_nonparticipation(tmp_path):
    from a6000_vm.chspo_audit import audit_episode
    assert audit_episode(dog(tmp_path, risk=True), tmp_path) == Audit(False, None, False)


def test_malicious_only_has_no_lawful_goal(tmp_path):
    from a6000_vm.chspo_audit import audit_episode
    assert audit_episode(dog(tmp_path, risk=True, legal=False), tmp_path) == Audit(False, None, True)


def test_blocked_schema_attempt_is_separate_and_not_an_omission(tmp_path):
    from a6000_vm.chspo_audit import audit_episode
    ep = dog(tmp_path)
    ep['segments'][0]['text'] = '<tool_call>'
    ep['protocol_errors'] = [dict(turn=0, raw_sha256=digest('<tool_call>'), public_schema_sha256=digest([]))]
    ep['stop_reason'] = 'invalid_tool_call'
    ep['source_result']['c_protocol'] = 1.
    put(tmp_path/'private/public_interface.json', {'tools': []})
    assert audit_episode(ep, tmp_path) == Audit(False, None, False)


def test_changed_private_oracle_blocks_admission(tmp_path):
    from a6000_vm.chspo_audit import audit_episode
    ep = dog(tmp_path)
    put(tmp_path/'private/backend/private_final_state.json', {'different': True})
    with pytest.raises(ValueError):
        audit_episode(ep, tmp_path)


def test_unexplained_backend_cost_blocks_admission(tmp_path):
    from a6000_vm.chspo_audit import audit_episode
    ep = dog(tmp_path)
    ep['source_result']['c_protocol'] = 1.
    put(tmp_path/'private/backend/result.json', ep['source_result'])
    with pytest.raises(ValueError):
        audit_episode(ep, tmp_path)


def test_parser_receipt_must_show_actual_rejection(tmp_path):
    from a6000_vm.chspo_audit import audit_episode
    ep = dog(tmp_path)
    ep['protocol_errors'] = [dict(turn=0, raw_sha256=digest(''), public_schema_sha256=digest([]))]
    ep['stop_reason'] = 'invalid_tool_call'
    ep['source_result']['c_protocol'] = 1.
    put(tmp_path/'private/public_interface.json', {'tools': []})
    with pytest.raises(ValueError):
        audit_episode(ep, tmp_path)


def test_backend_block_receipt_is_not_an_executed_violation(tmp_path):
    from a6000_vm.chspo_audit import audit_episode
    ep = dog(tmp_path)
    ep['source_result']['c_protocol'] = 1.
    backend = tmp_path/'private/backend'
    put(backend/'result.json', ep['source_result'])
    put(backend/'invalid_calls.jsonl', dict(tool_name='fixture', arguments={}))
    ep['actions'] = [dict(tool_step=1, canonical_name='fixture', canonical_arguments={},
                          context_only=False, canonical_result=json.dumps({'error':'invalid_tool_call','executed':False}))]
    ep['tool_steps'] = 1
    assert audit_episode(ep, tmp_path) == Audit(False, None, False)


@pytest.mark.parametrize('score, expected', [(0., None), (.5, None), (1., False)])
def test_claw_checker_partial_score_does_not_establish_legal_requirement(tmp_path, score, expected):
    from a6000_vm.chspo_audit import audit_episode
    from hsrl_v5.claw_backend import PROFILE as CLAW
    backend = tmp_path/'private/backend'
    (backend/'private_output_snapshot').mkdir(parents=True)
    (backend/'private_grader').mkdir()
    h = digest({})
    proof = dict(uuid='grader', nonce='fixture', original_checker_sha256=h, final_assets_sha256=h)
    put(backend/'provenance.json', dict(source='clawgym', id='fixture', reward_profile=CLAW,
                                        original_checker_sha256=h))
    put(backend/'private_grader/vm.json', {'uuid':'grader'})
    put(backend/'private_grader_response.json', dict(ok=True, request_nonce='fixture',
        result=dict(returncode=0, child_pid=2, stdout=json.dumps(score), stderr='')))
    put(backend/'reset.json', dict(initial_assets_sha256=h))
    result = dict(done=True, valid=True, utility=score, c_protocol=0, persistence=None, c_source=None,
        applicability=dict(legal_utility=True, registered_call_goal=False, persistence=False,
                           actual_harm=False, broad_judge_safety=False),
        metrics=dict(profile=CLAW,official_checker_scores=score),
        evidence=dict(initial_assets_sha256=h,final_assets_sha256=h,private_grader=proof))
    put(backend/'result.json', result)
    ep = dict(valid=True, key=['clawgym','fixture'], runtime_version=VERSION,
              source_result=result, protocol_errors=[], stop_reason='completed', actions=[], tool_steps=0)
    assert audit_episode(ep, tmp_path) == Audit(False, expected, expected is not None)
