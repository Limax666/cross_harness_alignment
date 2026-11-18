from flashrsi_b1_result_status import apply_reset_status


def test_incomplete_case_is_not_reported_as_reset_mismatch():
    record = {"unscorable_reason": "incomplete_or_unattested"}
    apply_reset_status(record)
    assert record["unscorable_reason"] == "incomplete_or_unattested"


def test_actual_reset_failure_is_reported():
    record = {"reset_verified": False}
    apply_reset_status(record)
    assert record["unscorable_reason"] == "reset_mismatch"


def test_successful_reset_does_not_create_failure():
    record = {"reset_verified": True}
    apply_reset_status(record)
    assert "unscorable_reason" not in record
