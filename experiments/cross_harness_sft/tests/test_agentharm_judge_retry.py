from types import SimpleNamespace

import httpx
import pytest

from cross_harness_sft.backends import agentharm


def _response(status_code=200):
    def raise_for_status():
        if status_code >= 400:
            raise httpx.HTTPStatusError("failed", request=None, response=None)

    return SimpleNamespace(
        status_code=status_code,
        raise_for_status=raise_for_status,
        json=lambda: {"choices": [{"message": {"content": "Answer: yes"}}]},
    )


def test_openai_compatible_judge_retries_transient_dns_error(monkeypatch):
    monkeypatch.setenv("TEST_JUDGE_KEY", "test-key")
    monkeypatch.setattr(agentharm.time, "sleep", lambda _: None)
    calls = iter([httpx.ConnectError("temporary DNS failure"), _response()])

    def post(*args, **kwargs):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(agentharm.httpx, "post", post)
    judge = agentharm.OpenAICompatibleJudge("test-model", "https://example.test/v1", "TEST_JUDGE_KEY")

    result = judge._generate("grade this")

    assert result.completion == "Answer: yes"


def test_openai_compatible_judge_does_not_retry_permanent_4xx(monkeypatch):
    monkeypatch.setenv("TEST_JUDGE_KEY", "test-key")
    monkeypatch.setattr(agentharm.time, "sleep", lambda _: None)
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        return _response(status_code=401)

    monkeypatch.setattr(agentharm.httpx, "post", post)
    judge = agentharm.OpenAICompatibleJudge("test-model", "https://example.test/v1", "TEST_JUDGE_KEY")

    with pytest.raises(httpx.HTTPStatusError):
        judge._generate("grade this")

    assert len(calls) == 1
