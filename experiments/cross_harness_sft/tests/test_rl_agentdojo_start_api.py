"""HTTP episode-start protocol regression for explicit AgentDojo injection fixtures."""
from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cross_harness_sft.benchmark_server import create_app


class AgentDojoDriver:
    version = "test"

    def __init__(self):
        self.calls = []

    def start(self, task_id, seed, *, injection_vector=None, injection_payload=None):
        self.calls.append((task_id, seed, injection_vector, injection_payload))
        if task_id.endswith(":injection") and not injection_vector:
            raise ValueError("explicit fixture required")
        return {"episode": object(), "tools": []}


class OtherDriver:
    version = "test"

    def start(self, task_id, seed):
        return {"episode": object(), "tools": []}


def test_start_forwards_explicit_fixture_only_to_agentdojo():
    driver = AgentDojoDriver()
    client = TestClient(create_app(driver))
    response = client.post("/episodes/start", json={"task_id": "family:injection", "seed": 7,
        "injection_vector": "visible", "injection_payload": "frozen attack"})
    assert response.status_code == 200
    assert driver.calls == [("family:injection", 7, "visible", "frozen attack")]
    assert client.post("/episodes/start", json={"task_id": "family:clean", "seed": 8}).status_code == 200
    assert driver.calls[-1] == ("family:clean", 8, None, None)
    assert client.post("/episodes/start", json={"task_id": "family:injection", "seed": 9}).status_code == 400


def test_other_drivers_reject_fixture_parameters():
    client = TestClient(create_app(OtherDriver()))
    response = client.post("/episodes/start", json={"task_id": "any", "seed": 1,
        "injection_vector": "visible", "injection_payload": "attack"})
    assert response.status_code == 400
    assert client.post("/episodes/start", json={"task_id": "any", "seed": 1}).status_code == 200
