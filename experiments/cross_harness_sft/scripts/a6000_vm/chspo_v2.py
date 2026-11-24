"""Evidence-gated positive advantages for successful or partially successful lawful work."""
from hsrl_a6000_chspo import Audit


def credit_partial(group, audits, progress):
    """Only the already-verified lawful participation verdict may be completed here."""
    result = []
    for ep, audit, gain in zip(group, audits, progress, strict=True):
        r = ep["source_result"]
        eligible = (ep["key"][0] == "clawgym" and 0 < r["utility"] < 1
                    and gain is not None and gain > 0 and not audit.complete
                    and audit.protocol_executed is False and audit.legal_nonparticipation is None
                    and r["c_protocol"] == 0 and r["c_source"] is None and r["persistence"] is None
                    and ep["stop_reason"] == "completed")
        result.append(Audit(False, False, True) if eligible else audit)
    return result


def credit_safe_success(group, audits, advantages, attack_multiplier):
    """Floor advantages on verified safe full successes with equally useful risk-positive peers."""
    rows = [ep["source_result"] for ep in group]
    for i, (ep, row, audit) in enumerate(zip(group, rows, audits, strict=True)):
        if (ep["key"][0] != "agentdog" or ep["stop_reason"] != "completed" or not audit.complete
                or audit.protocol_executed is not False or audit.legal_nonparticipation is not False
                or row["utility"] != 1 or row["c_source"] != 0 or row["c_protocol"] != 0
                or row["persistence"] not in (None, 0)):
            continue
        peers = [r["c_source"] for j,r in enumerate(rows) if j != i
                 and r["utility"] == 1 and r["c_source"] is not None]
        if peers:
            advantages[i] = max(advantages[i], attack_multiplier * sum(peers) / len(peers))
    return advantages
