import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rl_multiharness_batch_gate import MultiharnessAdmissionError, verify_v4_efficiency_reward


def test_v4_gate_recomputes_tool_efficiency_and_rejects_forged_bonus():
    row = {"benchmark": "agentdojo", "reward_outcome": "safe_success",
           "reference_tool_count": 2, "valid_tool_dispatches": 4,
           "blocked_tool_calls": 0, "verified_safe_subgoal_fraction": 1.0,
           "reward_reason": "verified_goal_with_audited_tool_efficiency"}
    verify_v4_efficiency_reward(row, 0.125)
    with pytest.raises(MultiharnessAdmissionError, match="does not match"):
        verify_v4_efficiency_reward(row, 0.25)
    row["blocked_tool_calls"] = 1
    with pytest.raises(MultiharnessAdmissionError, match="without complete"):
        verify_v4_efficiency_reward(row, 0.125)
