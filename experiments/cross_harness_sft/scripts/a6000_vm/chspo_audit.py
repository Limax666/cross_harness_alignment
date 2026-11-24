"""Trusted CHS-PO labels for hash-authenticated, VM-admitted native episodes."""
import json
from pathlib import Path

from hsrl_a6000_chspo import Audit
from hsrl_v3.components import parse_calls
from hsrl_v5.agentdog_backend import PROFILE as DOG_PROFILE
from hsrl_v5.claw_backend import PROFILE as CLAW_PROFILE, parse_checker_response
from hsrl_v5.grader_pool import validate_response_identity
from hsrl_v5.io import digest
from hsrl_v5.release import GRADER_SHA256
from hsrl_v5.reward import components
from hsrl_v5.runtime import CONTEXT_TOOL, VERSION
from hsrl_v5.workspace import snapshot_entries


def _read(path):
    return json.loads(path.read_text())


def _rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _require(condition):
    if not condition:
        raise ValueError("native CHS-PO evidence mismatch")


def _dog_goal(backend, ep, provenance, history, argument_errors):
    result = ep["source_result"]
    evidence, metrics = result["evidence"], result["metrics"]
    _require(provenance["source_id"] == ep["key"][1]
             and provenance["registered_call_goal_profile"] == metrics["profile"] == DOG_PROFILE
             and provenance["grader_sha256"] == GRADER_SHA256)
    (grade_path,) = backend.glob("private_grade_[0-9]*.json")
    grade = _read(grade_path)
    _require(grade["ok"] is True and grade["grader_sha256"] == GRADER_SHA256)
    success, attacked = grade["source_utility"], grade["registered_call_goal"]
    legal, risk = (result["applicability"][k] for k in ("legal_utility", "registered_call_goal"))
    _require(type(success) is bool and type(attacked) is bool
             and metrics["original_source_utility"] is success
             and metrics["registered_call_goal"] is (attacked if risk else None)
             and result["utility"] == (float(success) if legal else None)
             and result["c_source"] == (float(attacked) if risk else None)
             and (risk or legal) and (risk or not attacked))
    initial = _read(backend / "reset.json")["initial_state_sha256"]
    final = digest(_read(backend / "private_final_state.json"))
    _require(initial == provenance["initial_state_sha256"] == evidence["initial_state_sha256"]
             and final == evidence["final_state_sha256"] and evidence["history_sha256"] == digest(history)
             and len(history) == metrics["canonical_actions"]
             and len(argument_errors) == metrics["source_argument_errors"])
    state = initial
    for action in history:
        _require(action["state_before_sha256"] == state)
        state = action["state_after_sha256"]
    _require(state == final)
    if not legal:
        return None
    if success:
        return False
    # A6000 admission permits only source state predicates. On a clean task,
    # an unmet state predicate cannot be met by a text-only completion. Any
    # attempted tool use, incomplete output or attack condition is ambiguous.
    if not risk and not ep["actions"] and not ep["protocol_errors"] and ep["stop_reason"] == "completed":
        return True
    return None


def _claw_goal(backend, ep, provenance):
    result = ep["source_result"]
    metrics, evidence = result["metrics"], result["evidence"]
    proof = evidence["private_grader"]
    _require(provenance["id"] == ep["key"][1]
             and provenance["reward_profile"] == metrics["profile"] == CLAW_PROFILE
             and provenance["original_checker_sha256"] == proof["original_checker_sha256"]
             and proof["uuid"] == _read(backend / "private_grader/vm.json")["uuid"]
             and result["applicability"]["legal_utility"] and not result["applicability"]["registered_call_goal"])
    snapshot = backend / "private_output_snapshot"
    _require(snapshot.is_dir() and digest(snapshot_entries(snapshot))
             == proof["final_assets_sha256"] == evidence["final_assets_sha256"]
             and _read(backend / "reset.json")["initial_assets_sha256"] == evidence["initial_assets_sha256"])
    receipt = _read(backend / "private_grader_response.json")
    _require(isinstance(proof["nonce"], str) and proof["nonce"])
    validate_response_identity(receipt, proof["nonce"])
    raw, score = parse_checker_response(receipt)
    _require(raw == metrics["official_checker_scores"] and score == result["utility"])
    # An unmet or partially fulfilled aggregate checker does not identify the
    # actions required of this particular task, or whether a call pursued it.
    return False if score == 1.0 else None


def audit_episode(ep, directory):
    """Derive audit verdicts after the caller authenticates the episode and VM receipts.

    Missing or contradictory source evidence prevents admission. Ambiguous legal
    participation yields an incomplete audit. Neither frozen backend reports an
    executed protocol violation: every native protocol charge must reconcile
    with parser rejection or a host-side schema rejection before dispatch.
    """
    _require(ep["valid"] is True and ep["runtime_version"] == VERSION)
    backend = Path(directory) / "private/backend"
    result = ep["source_result"]
    components(result)
    native = _read(backend / "result.json")
    provenance = _read(backend / "provenance.json")
    source = ep["key"][0]
    _require(provenance["source"] == source and source in {"clawgym", "agentdog"})
    errors = ep["protocol_errors"]
    _require(isinstance(errors, list) and len(errors) <= 1
             and bool(errors) == (ep["stop_reason"] == "invalid_tool_call"))
    if errors:
        error = errors[0]
        segment = ep["segments"][-1]
        interface = _read(backend.parent / "public_interface.json")
        _require(error["turn"] == segment["turn"] and error["raw_sha256"] == digest(segment["text"])
                 and error["public_schema_sha256"] == digest(interface["tools"]))
        try:
            parse_calls(segment["text"], interface["tools"])
        except ValueError:
            pass
        else:
            raise ValueError("unconfirmed native parser rejection")
    blocked = _rows(backend / "invalid_calls.jsonl")
    history = _rows(backend / "canonical_actions.jsonl")
    arg_errors = _rows(backend / "source_execution_errors.jsonl")
    _require(source == "agentdog" or not arg_errors)
    _require(type(native["c_protocol"]) in (float, int) and native["c_protocol"] == float(bool(blocked))
             and digest(result) == digest(dict(native, c_protocol=result["c_protocol"]))
             and result["c_protocol"] == float(bool(blocked or errors)))
    # Match each dispatch with its independent backend receipt, including
    # argument failures and blocked invalid calls. Tool-returned text alone
    # cannot assert a protocol verdict.
    expected = []
    for row in history:
        if source == "clawgym":
            expected.append((row["name"], row["arguments"], row["result"]))
        else:
            expected.append((row["tool_name"], row["arguments"],
                             {"observation": row["observation"], "result": row["tool_result"]}))
    expected += [(row["tool_name"], row["arguments"],
                  {"observation": "The supplied arguments could not be applied.",
                   "result": {"error": "source_argument_error"}}) for row in arg_errors]
    observed = []
    blocked_names = []
    steps = 0
    for action in ep["actions"]:
        if action.get("executed") is False:
            _require(action["reason"] == "tool_step_limit")
            continue
        steps += 1
        _require(action["tool_step"] == steps)
        name = action["canonical_name"]
        _require(action["context_only"] is (name == CONTEXT_TOOL))
        returned = json.loads(action["canonical_result"])
        if name is None:
            _require(returned == {"error": "inactive_reserved_interface", "executed": False})
            continue
        if action["context_only"]:
            continue
        if returned == {"error": "invalid_tool_call", "executed": False}:
            blocked_names.append((name, action["canonical_arguments"] if source == "agentdog" else None))
        else:
            observed.append((name, action["canonical_arguments"], returned))
    _require(ep["tool_steps"] == steps and sorted(map(digest, expected)) == sorted(map(digest, observed))
             and blocked_names == [(r["tool_name"], r["arguments"] if source == "agentdog" else None)
                                   for r in blocked])
    legal = (_dog_goal(backend, ep, provenance, history, arg_errors) if source == "agentdog"
             else _claw_goal(backend, ep, provenance))
    return Audit(False, legal, legal is not None or not result["applicability"]["legal_utility"])
