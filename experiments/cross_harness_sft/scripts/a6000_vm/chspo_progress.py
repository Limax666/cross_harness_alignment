"""Episode-level ClawGym progress relative to a same-task, unchanged workspace."""
import json
from pathlib import Path

from a6000_vm.chspo_audit import audit_episode
from hsrl_v5.io import digest


def group_progress(episodes, episode_dirs):
    """Return per-slot official score gains or None when attribution is ambiguous.

    Caller authenticates reservation, episode and VM cleanup receipts before use.
    Zero means no observed gain over reset. It never establishes nonparticipation.
    """
    if len(episodes) != len(episode_dirs) or not episodes:
        raise ValueError("missing progress inputs")
    first = episodes[0]
    if any((ep["key"], ep["policy_version"]) != (first["key"], first["policy_version"])
           for ep in episodes):
        raise ValueError("progress requires one registered same-policy task group")
    unknown = [None] * len(episodes)
    if first["key"][0] != "clawgym":
        return unknown
    bindings, graders, nonces = set(), set(), set()
    rows = []
    for ep, directory in zip(episodes, episode_dirs):
        audit_episode(ep, directory)
        backend = Path(directory) / "private/backend"
        provenance = json.loads((backend / "provenance.json").read_text())
        result = ep["source_result"]
        evidence = result["evidence"]
        proof = evidence["private_grader"]
        bindings.add((provenance["source_tree_sha256"], proof["case_id"],
                      proof["original_checker_sha256"], evidence["initial_assets_sha256"]))
        if proof["uuid"] in graders or proof["nonce"] in nonces:
            raise ValueError("reused progress grader receipt")
        graders.add(proof["uuid"])
        nonces.add(proof["nonce"])
        scores = result["metrics"]["official_checker_scores"]
        # Keep scalar and component-valued checker schemas distinct.
        components = scores if isinstance(scores, dict) else {None: scores}
        rows.append((evidence["final_assets_sha256"], components, result["utility"]))
    if len(bindings) != 1:
        raise ValueError("different reset or checker identity in progress group")
    reset = next(iter(bindings))[-1]
    # Identical inputs with disagreeing official grades invalidate the comparison.
    observed = {}
    for final, components, utility in rows:
        score_id = digest({"components": list(sorted(components.items())), "utility": utility})
        if observed.setdefault(final, score_id) != score_id:
            return unknown
    baseline = next(((c, u) for final, c, u in rows if final == reset), None)
    if baseline is None:
        return unknown
    base_components, base_utility = baseline
    gains = []
    for ep, (final, components, utility) in zip(episodes, rows):
        if final == reset:
            gains.append(0.0)
            continue
        wrote = any(action.get("executed") is not False and not action["context_only"]
                    and action["canonical_name"] in {"write_file", "run_python", "run_shell"}
                    for action in ep["actions"])
        if (not wrote or ep["source_result"]["c_protocol"] or components.keys() != base_components.keys()
                or any(value < base_components[k] for k, value in components.items())):
            gains.append(None)
        else:
            gains.append(max(0.0, utility - base_utility))
    return gains
