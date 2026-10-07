"""Convert a VeRL rollout batch into strict pilot admission evidence.

Call this immediately before ``RayPPOTrainer._update_actor``. The producer of
``multiharness_evidence`` must be the trusted online agent loop, never model text.
"""
from __future__ import annotations

import math

from rl_multiharness_batch_gate import MultiharnessAdmissionError, require_online_batch
from rl_multiharness_stratified_sampler import EXPECTED_STRATA, GROUPS_PER_STRATUM


def _as_list(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def require_verl_update_batch(batch, *, policy_snapshot: str, cells: dict[str, dict]) -> dict:
    tensors = batch.batch
    extras = batch.non_tensor_batch
    required = ("responses", "response_mask", "attention_mask", "rollout_log_probs", "old_log_probs", "rm_scores")
    if any(key not in tensors for key in required) or "multiharness_evidence" not in extras:
        raise MultiharnessAdmissionError("missing VeRL token fields or trusted pilot evidence")
    evidence = _as_list(extras["multiharness_evidence"])
    expected_groups = len(EXPECTED_STRATA) * GROUPS_PER_STRATUM
    if len(evidence) != 4 * expected_groups:
        raise MultiharnessAdmissionError("expected exactly G=4 VeRL episodes per sampled cell")
    if "uid" not in extras:
        raise MultiharnessAdmissionError("missing VeRL GRPO group identities")
    uids = _as_list(extras["uid"])
    if len(uids) != 4 * expected_groups:
        raise MultiharnessAdmissionError("VeRL GRPO uid count differs from rollout count")
    by_uid = {}
    by_cell = {}
    for uid, item in zip(uids, evidence):
        if not isinstance(uid, str) or not uid or not isinstance(item, dict):
            raise MultiharnessAdmissionError("invalid VeRL uid or pilot evidence")
        cell = item.get("cell_id")
        if uid in by_uid and by_uid[uid] != cell:
            raise MultiharnessAdmissionError("VeRL grouped different native cells under one uid")
        if cell in by_cell and by_cell[cell] != uid:
            raise MultiharnessAdmissionError("native cell split across multiple VeRL GRPO groups")
        by_uid[uid] = cell
        by_cell[cell] = uid
    if len(by_uid) != expected_groups or any(uids.count(uid) != 4 for uid in by_uid):
        raise MultiharnessAdmissionError("VeRL did not retain each sampled cell with four samples")
    rows = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise MultiharnessAdmissionError("pilot evidence must be a structured mapping")
        token_ids = _as_list(tensors["responses"][index])
        masks = _as_list(tensors["response_mask"][index])
        attention = _as_list(tensors["attention_mask"][index])[-len(token_ids):]
        sampled = _as_list(tensors["rollout_log_probs"][index])
        old = _as_list(tensors["old_log_probs"][index])
        scores = _as_list(tensors["rm_scores"][index])
        if not all(len(values) == len(token_ids) for values in (masks, attention, sampled, old, scores)):
            raise MultiharnessAdmissionError("VeRL response fields have different widths")
        if any(value not in (0, 1) for value in attention):
            raise MultiharnessAdmissionError("invalid response attention mask")
        if not any(attention) or 0 in attention[:sum(attention)]:
            raise MultiharnessAdmissionError("response attention mask has holes")
        width = sum(attention)
        if any(score != 0 for score in scores[:width-1] + scores[width:]):
            raise MultiharnessAdmissionError("reward appears outside final response token")
        # rm_scores round-trips through a float32 tensor; the continuous
        # outcome+path reward is therefore compared with float32 tolerance,
        # never exact equality (exact match only held for {-1, 0, 1}).
        stored_reward = float(scores[width-1])
        if not math.isclose(stored_reward, float(item["reward"]),
                            rel_tol=0.0, abs_tol=1e-6):
            raise MultiharnessAdmissionError("VeRL reward differs from verified outcome")
        origins = item.get("token_origins")
        if not isinstance(origins, list) or len(origins) != width:
            raise MultiharnessAdmissionError("missing generation-time token origins")
        row = dict(item)
        row.update(
            response_ids=token_ids[:width], response_mask=masks[:width],
            token_origins=origins,
            rollout_log_probs=sampled[:width], old_log_probs=old[:width],
        )
        rows.append(row)
    return require_online_batch(rows, policy_snapshot=policy_snapshot, cells=cells)
