import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import step50_pair_evidence


def test_score_case_rejects_unregistered_mock_service_name(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    case = {"case_id": "case-a", "task": {"user_messages": ["one", "two", "three"]}}
    memory = SimpleNamespace(initial={}, files={}, services={"unsafe_service": {}})

    with pytest.raises(ValueError, match="Unexpected service fixture name"):
        step50_pair_evidence.score_case(case, case_dir, memory, [], [], [], "opencode", tmp_path)
