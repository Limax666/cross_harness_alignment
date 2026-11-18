"""Preserve the distinction between an unchecked reset and a failed reset."""


def apply_reset_status(record):
    if record.get("reset_verified") is False:
        record["unscorable_reason"] = "reset_mismatch"
