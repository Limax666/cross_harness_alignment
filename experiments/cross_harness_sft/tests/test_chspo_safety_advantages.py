import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "verl"))
sys.path.insert(0, str(ROOT / "scripts"))
from rl_guarded_chspo_trainer import constrained_advantages, next_safety_streaks


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
