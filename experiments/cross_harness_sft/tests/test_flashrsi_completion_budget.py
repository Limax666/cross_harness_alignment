import pytest

from flashrsi_completion_budget import completion_budget


def test_default_preserves_frozen_512_token_condition():
    assert completion_budget({}) == 512


def test_explicit_1024_token_condition_is_separately_selected():
    assert completion_budget({'FLASHRSI_COMPLETION_BUDGET': '1024'}) == 1024
    assert completion_budget({'FLASHRSI_COMPLETION_BUDGET': '2048'}) == 2048
    assert completion_budget({'FLASHRSI_COMPLETION_BUDGET': '4096'}) == 4096


@pytest.mark.parametrize('value', ['', '0', '513', '-1', '16384', 'abc'])
def test_rejects_unregistered_budgets(value):
    with pytest.raises(ValueError):
        completion_budget({'FLASHRSI_COMPLETION_BUDGET': value})
