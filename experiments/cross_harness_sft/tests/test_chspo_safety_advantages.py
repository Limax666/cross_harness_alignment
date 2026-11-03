import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "verl"))
sys.path.insert(0, str(ROOT / "scripts"))
from rl_guarded_chspo_trainer import constrained_advantages, next_safety_streaks, require_objective_version


def row(harness="hermes", arm="risk", attack=False, persistence=False, invalid=False,
        benchmark="agentharm", blocked=0, outcome="safe_incomplete"):
    return {"benchmark": benchmark, "harness": harness, "arm": arm,
            "official_attack_success": attack,
            "prohibited_persistence": persistence, "invalid_tool_dispatched": invalid,
            "blocked_tool_calls": blocked, "termination": "completed", "reward_outcome": outcome}


def test_all_unsafe_group_gets_negative_advantage():
    evidence = [row(attack=True) for _ in range(4)] + [row(arm="benign") for _ in range(4)]
    result, audit = constrained_advantages(torch.zeros(8, 3), evidence, {})
    assert torch.all(result[:4] < 0)
    assert torch.all(result[4:] == 0)
    assert audit["train_cost_rates"]["agentharm/hermes"]["attack"] == 1.0


def test_safe_utility_retained_and_unsafe_positive_removed_per_harness():
    evidence = [row("hermes", "risk", attack=True), row("hermes", "benign"),
                row("codex", "risk"), row("codex", "benign")]
    result, audit = constrained_advantages(torch.ones(4, 2), evidence, {})
    assert torch.all(result[0] < 0)
    assert torch.all(result[1:] == 1)
    assert audit["duals"]["agentharm/hermes"]["attack"] > audit["duals"]["agentharm/codex"]["attack"]


def test_unpaired_batch_rejected():
    try:
        constrained_advantages(torch.zeros(1, 2), [row()], {})
    except ValueError as exc:
        assert "unpaired" in str(exc)
    else:
        raise AssertionError("unpaired batch was accepted")


def test_blocked_attempt_stays_above_nonengagement_even_with_saturated_protocol_dual():
    evidence = [row(arm="benign", blocked=1), row(arm="benign", blocked=1),
                row(arm="benign", outcome="benign_nonengagement"), row(arm="benign")]
    evidence += [row()]
    advantages = torch.tensor([[0.19], [-0.06], [-0.31], [0.19], [0.0]])
    duals = {"agentharm/hermes": {"attack": 0.25, "persistence": 0.50, "protocol": 1.0}}
    result, audit = constrained_advantages(advantages, evidence, duals)
    assert result[0, 0] == advantages[0, 0]
    assert result[1, 0] == advantages[1, 0]
    assert result[2, 0] < result[1, 0]
    assert audit["train_cost_rates"]["agentharm/hermes"]["protocol"] == 0.0
    assert audit["train_cost_rates"]["agentharm/hermes"]["blocked_attempt"] == 0.4


def test_codex_duals_do_not_mix_benchmarks():
    evidence = [row("codex", "risk", attack=True, benchmark="agentdojo"),
                row("codex", "benign", benchmark="agentdojo"),
                row("codex", "risk", benchmark="agentharm"),
                row("codex", "benign", benchmark="agentharm")]
    _, audit = constrained_advantages(torch.zeros(4, 2), evidence, {})
    assert audit["duals"]["agentdojo/codex"]["attack"] > audit["duals"]["agentharm/codex"]["attack"]


def test_gate_stops_before_third_high_risk_update():
    rates = {"agentdojo/codex": {"attack": 0.2, "persistence": 0.0}}
    first = next_safety_streaks(rates, {})
    second = next_safety_streaks(rates, first)
    try:
        next_safety_streaks(rates, second)
    except ValueError as exc:
        assert "safety gate" in str(exc)
    else:
        raise AssertionError("third unsafe update was admitted")
    assert next_safety_streaks({"agentdojo/codex": {"attack": 0.0, "persistence": 0.0}}, second)["agentdojo/codex"]["attack"] == 0


def test_gate_uses_train_sft_baseline_for_persistence():
    baseline = {"agentdojo/codex": {"attack": 0.0, "persistence": 0.4375}}
    state = {}
    for _ in range(3):
        state = next_safety_streaks(baseline, state)
    assert state["agentdojo/codex"]["persistence"] == 0
    high = {"agentdojo/codex": {"attack": 0.0, "persistence": 0.625}}
    first = next_safety_streaks(high, state)
    second = next_safety_streaks(high, first)
    try:
        next_safety_streaks(high, second)
    except ValueError as exc:
        assert "persistence" in str(exc)
    else:
        raise AssertionError("persistent safety deterioration was admitted")


def test_rare_attack_keeps_uncentered_penalty_after_zero_attack_batches():
    duals = {}
    safe = [row() for _ in range(4)] + [row(arm="benign") for _ in range(4)]
    for _ in range(100):
        _, audit = constrained_advantages(torch.zeros(8, 1), safe, duals)
        assert audit["duals"]["agentharm/hermes"]["attack"] >= 0.25
        assert audit["duals"]["agentharm/hermes"]["persistence"] >= 0.50
    rare_attack = [row(attack=True)] + safe[1:]
    result, audit = constrained_advantages(torch.zeros(8, 1), rare_attack, duals)
    assert result[0, 0] <= -0.25
    assert torch.all(result[1:] == 0)
    assert audit["train_cost_rates"]["agentharm/hermes"]["attack"] == 0.25


def test_at_sft_baseline_persistence_does_not_drive_dual_to_saturation():
    # One benign and one risk row with persistence produces a 0.5 risk rate
    # only when the risk row is unsafe; use sixteen risk rows for the exact
    # AgentDojo/Codex frozen SFT rate of 7/16.
    evidence = [row("codex", "risk", persistence=i < 7, benchmark="agentdojo")
                for i in range(16)]
    evidence += [row("codex", "benign", benchmark="agentdojo") for _ in range(16)]
    duals = {}
    for _ in range(30):
        _, audit = constrained_advantages(torch.zeros(32, 1), evidence, duals)
    assert audit["train_cost_rates"]["agentdojo/codex"]["persistence"] == 7 / 16
    assert audit["duals"]["agentdojo/codex"]["persistence"] == 0.50


def test_gate_rejects_repeated_single_attack_above_zero_sft_baseline():
    rates = {"agentharm/hermes": {"attack": 1 / 16, "persistence": 0.0}}
    first = next_safety_streaks(rates, {})
    second = next_safety_streaks(rates, first)
    try:
        next_safety_streaks(rates, second)
    except ValueError as exc:
        assert "agentharm/hermes/attack" in str(exc)
    else:
        raise AssertionError("three attack-positive train batches were admitted")


def test_old_objective_checkpoint_cannot_resume_with_new_safety_rule():
    require_objective_version("CHS-PO-harness-safety-dual-v3")
    try:
        require_objective_version("CHS-PO-harness-safety-dual-v2")
    except ValueError as exc:
        assert "objective version" in str(exc)
    else:
        raise AssertionError("v2 checkpoint was admitted into a v3 run")


def test_v4_objective_requires_explicit_version_flag(monkeypatch):
    monkeypatch.setenv("CHSPO_OBJECTIVE_VERSION", "CHS-PO-harness-safety-dual-v4")
    require_objective_version("CHS-PO-harness-safety-dual-v4")
    try:
        require_objective_version("CHS-PO-harness-safety-dual-v3")
    except ValueError as exc:
        assert "objective version" in str(exc)
    else:
        raise AssertionError("v3 objective admitted into a v4 run")
