import json

from flashrsi_hermes_gepa_contextfit_v10 import MAX_EVIDENCE_CHARS, compact_evidence, reflection_prompt
from flashrsi_harnessrisk_algorithm import resolve_algorithm


def test_contextfit_projection_excludes_raw_tool_transcript_and_is_bounded():
    raw_tool = "do-not-replay " * 1000
    evidence = compact_evidence([{
        "task": "task", "output": "answer", "metrics": {"utility": 0.0},
        "feedback": {"status": "finished", "ground_truth": None, "error_message": None},
        "trajectory": [{"type": "tool_execution_end", "result": raw_tool},
                       {"type": "hermes_result", "completed": False, "api_calls": 12,
                        "iteration_budget_exhausted": True, "final_response": "brief"}],
    }])
    assert len(evidence) <= MAX_EVIDENCE_CHARS + 1
    assert raw_tool not in evidence
    row = json.loads(evidence)[0]
    assert row["terminal"]["iteration_budget_exhausted"] is True
    assert len(row["trajectory_sha256"]) == 64
    assert "at most 700 words" in reflection_prompt("SOUL.md", "old", evidence)


def test_contextfit_selector_is_hermes_specific(monkeypatch):
    monkeypatch.setenv("HR_GEPA_REFLECTION_PROTOCOL", "hermes-v10-contextfit")
    assert resolve_algorithm("gepa")[0].__name__ == "HermesContextfitGEPA"
