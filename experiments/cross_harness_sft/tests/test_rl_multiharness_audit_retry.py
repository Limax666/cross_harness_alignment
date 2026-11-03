"""An ambiguous native audit triggers a fresh rollout, never a made-up reward."""

import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rl_verl_multiharness_loop import (
    AuditAbstention, NativeMultiharnessVeRLLoop, validate_reward_evidence,
)


def _loop_with_attempts(outcomes):
    loop = object.__new__(NativeMultiharnessVeRLLoop)
    seen = []

    async def run_once(sampling_params, priority=0, **kwargs):
        seen.append((sampling_params, priority, kwargs))
        outcome = outcomes[len(seen) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    loop._run_once = run_once
    return loop, seen


def test_resettable_audit_abstention_regenerates_same_cell():
    accepted = SimpleNamespace(extra_fields={"multiharness_evidence": {"reward": -1.0}})
    loop, seen = _loop_with_attempts([AuditAbstention("path_audit_abstained"), accepted])
    result = asyncio.run(loop.run({"temperature": 0.8}, priority=2, cell_id="dojo/cell"))
    assert result is accepted
    assert result.extra_fields["multiharness_evidence"]["audit_retry_count"] == 1
    assert len(seen) == 2
    assert all(item == ({"temperature": 0.8}, 2, {"cell_id": "dojo/cell"}) for item in seen)


def test_systemic_audit_abstention_still_fails_closed():
    loop, seen = _loop_with_attempts([AuditAbstention("path_audit_abstained")] * 8)
    with pytest.raises(ValueError, match="eight fresh resettable rollouts"):
        asyncio.run(loop.run({}, cell_id="dojo/cell"))
    assert len(seen) == 8


def test_non_audit_failure_is_never_retried():
    loop, seen = _loop_with_attempts([ValueError("reset mismatch")])
    with pytest.raises(ValueError, match="reset mismatch"):
        asyncio.run(loop.run({}, cell_id="dojo/cell"))
    assert len(seen) == 1


def test_whitebox_fixed_budget_never_resamples_audit_abstentions():
    loop,seen=_loop_with_attempts([AuditAbstention('missing effect evidence')]*8)
    loop.whitebox=True
    with pytest.raises(AuditAbstention):
        asyncio.run(loop.run({},cell_id='wb/cell'))
    assert len(seen)==1


def test_verified_official_safe_zero_survives_path_audit_abstention():
    verdict = {
        "reward": 0.0,
        "reward_reason": "path_audit_abstained",
        "official_attack_success": False,
        "final_state_verified": True,
        "audit_complete": False,
        "reset_verified": True,
    }
    validate_reward_evidence(verdict)


def test_audit_abstention_cannot_admit_positive_reward_or_unknown_official_result():
    positive = {
        "reward": 1.0, "reward_reason": "verified_goal", "official_attack_success": False,
        "final_state_verified": True, "audit_complete": False, "reset_verified": True,
    }
    with pytest.raises(AuditAbstention):
        validate_reward_evidence(positive)
    unknown = {**positive, "reward": 0.0, "reward_reason": "path_audit_abstained",
               "official_attack_success": None}
    with pytest.raises(AuditAbstention):
        validate_reward_evidence(unknown)


def test_unverified_environment_reset_remains_fatal():
    verdict = {
        "reward": 0.0, "reward_reason": "path_audit_abstained", "official_attack_success": False,
        "final_state_verified": True, "audit_complete": False, "reset_verified": False,
    }
    with pytest.raises(ValueError, match="reset"):
        validate_reward_evidence(verdict)
