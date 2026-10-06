"""Fail-closed native AgentDojo injection fixture checks."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cross_harness_sft.backends.agentdojo import AgentDojoDriver


class FakeEnvironment:
    def __init__(self, injected):
        self.injected = injected

    def model_copy(self, deep):
        assert deep
        return FakeEnvironment(self.injected.copy())


class FakeSuite:
    tools = []

    def get_injection_vector_defaults(self):
        return {"visible_a": "clean_a", "visible_b": "clean_b"}

    def load_and_inject_default_environment(self, injections):
        return FakeEnvironment(injections)


class FakeTask:
    GOAL = "do the attack"


class FakeDriver(AgentDojoDriver):
    def __init__(self):
        self._case_map = {
            "family:clean": (FakeSuite(), object(), None),
            "family:injection": (FakeSuite(), object(), FakeTask()),
        }


def test_injection_requires_explicit_vector_and_payload():
    driver = FakeDriver()
    for kwargs in ({"injection_vector": "visible_a"},
                   {"injection_payload": "attack"},
                   {"injection_vector": "unknown", "injection_payload": "attack"}):
        try:
            driver.start("family:injection", 1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("injection fixture must fail closed")


def test_auto_injection_matches_official_direct_attack(monkeypatch):
    import agentdojo.attacks.baseline_attacks as baseline

    class FakeDirectAttack:
        def __init__(self, suite, target_pipeline=None):
            pass

        def attack(self, user, injection):
            return {"visible_a": "attack", "visible_b": "attack"}

    monkeypatch.setattr(baseline, "DirectAttack", FakeDirectAttack)
    driver = FakeDriver()
    episode = driver.start("family:injection", 1)["episode"]
    assert episode.environment.injected == {"visible_a": "attack", "visible_b": "attack"}


def test_auto_injection_fails_closed_without_candidates(monkeypatch):
    import agentdojo.attacks.baseline_attacks as baseline

    class EmptyDirectAttack:
        def __init__(self, suite, target_pipeline=None):
            pass

        def attack(self, user, injection):
            return {}

    monkeypatch.setattr(baseline, "DirectAttack", EmptyDirectAttack)
    driver = FakeDriver()
    try:
        driver.start("family:injection", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("auto injection must fail closed with no injectable vectors")


def test_single_vector_is_only_modified_surface():
    driver = FakeDriver()
    episode = driver.start("family:injection", 1, injection_vector="visible_a",
                           injection_payload="attack")["episode"]
    assert episode.environment.injected == {"visible_a": "attack"}
    assert "visible_b" not in episode.environment.injected


def test_pilot_rejects_registered_but_unobserved_vector():
    driver = FakeDriver()
    driver._case_map["workspace:user_task_6:injection_task_1"] = (FakeSuite(), object(), FakeTask())
    try:
        driver.start("workspace:user_task_6:injection_task_1", 1,
                     injection_vector="visible_a", injection_payload="attack")
    except ValueError:
        pass
    else:
        raise AssertionError("pilot accepted injection slot unseen by official task")


def test_clean_arm_rejects_injection_arguments():
    driver = FakeDriver()
    try:
        driver.start("family:clean", 1, injection_vector="visible_a", injection_payload="attack")
    except ValueError:
        pass
    else:
        raise AssertionError("clean fixture accepted attack payload")
