"""Prepare immutable real-source inventories before any model results exist."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess

from .backends import sha256_file, tree_digest, write_json

SEED = 20261116
REFERENCE_WEIGHTS_V3 = {"clawgym": 0.75, "harnesssafe": 0.125, "harnessaudit": 0.125}
WEIGHTS = {"clawgym": 0.75, "skilllifebench": 0.25}


def _record(source: str, source_id: str, family: str, path: Path, **extra) -> dict:
    return {"source": source, "id": source_id, "family": family, "source_path": str(path.resolve()),
            "source_sha256": sha256_file(path), "split": None,
            "condition": "native_boundary" if source == "harnessaudit" else "benign" if source == "clawgym" else "attack",
            "qualified": False, "exclusion_reasons": [], "historical_exposure": "unknown",
            **extra}


def _family_splits(rows: list[dict], source: str) -> dict[str, str]:
    source_rows = [row for row in rows if row["source"] == source]
    qualified_families = {row["family"] for row in source_rows if row["qualified"]}
    family_sets = [qualified_families, {row["family"] for row in source_rows} - qualified_families]
    result = {}
    # Stratify by static adapter qualification before any model outcome, keeping
    # the entire original family in one split, including excluded relatives.
    for stratum, family_set in enumerate(family_sets):
        families = sorted(family_set)
        random.Random(f"{SEED}:{source}:{stratum}").shuffle(families)
        if source == "clawgym":
            train_end = round(len(families) * .9)
            dev_end = len(families)
        else:
            n_dev = max(1, round(len(families) * .15)) if len(families) >= 3 else 0
            n_test = max(1, round(len(families) * .15)) if len(families) >= 3 else 0
            train_end = len(families) - n_dev - n_test
            dev_end = len(families) - n_test
        result.update({family: "train" if i < train_end else "dev" if i < dev_end else "test"
                       for i, family in enumerate(families)})
    return result


def _query_features(query: str) -> set[str]:
    # Entity-independent source template similarity, frozen before selection.
    query = re.sub(r'[`"\'][^`"\']*[`"\']', ' VALUE ', query.lower())
    tokens = re.findall(r"[a-z]+", re.sub(r"\d+", " NUMBER ", query))
    return {" ".join(tokens[i:i + 3]) for i in range(max(0, len(tokens) - 2))}


def _cluster_claw(rows: list[dict]) -> None:
    """Conservative complete source-template grouping, never case-ID families.

    Exact normalized prompts or >=.35 token-trigram Jaccard are linked. An
    inverted index checks all potentially qualifying pairs, with no LSH misses.
    Transitive closure intentionally keeps bridges in one split.
    """
    parents = list(range(len(rows)))
    features = [_query_features(json.loads(Path(r["source_path"]).read_text())["user_query"]) for r in rows]
    index: dict[str, list[int]] = defaultdict(list)

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, terms in enumerate(features):
        intersections = Counter(j for term in terms for j in index[term])
        for j, intersection in intersections.items():
            union = len(terms) + len(features[j]) - intersection
            if union and intersection / union >= 0.35:
                a, b = find(i), find(j)
                parents[max(a, b)] = min(a, b)
        for term in terms:
            index[term].append(i)
    grouped = defaultdict(list)
    for i in range(len(rows)):
        grouped[find(i)].append(i)
    for group in grouped.values():
        canonical_features = sorted(set.union(*(features[i] for i in group)))
        signature = hashlib.sha256("\n".join(canonical_features).encode()).hexdigest()
        for i in group:
            rows[i]["family"] = "clawgym:source_template:" + signature[:20]
            rows[i]["family_method"] = "normalized_query_trigram_jaccard_ge_0.35_transitive_v1"
            rows[i]["family_members"] = len(group)


def prepare_reference_pool_v3(project_root: Path, claw_root: Path, out: Path) -> dict:
    from single_agent.loader import load_sa_task_with_tools
    from .audit_backend import qualification as audit_qualification
    from .claw_backend import PINNED_REVISION, qualification as claw_qualification
    from .safe_backend import qualification as safe_qualification, IMPLEMENTED_UTILITY, UTILITY_CANDIDATES, accept_utility_checker
    project_root, claw_root, out = Path(project_root).resolve(), Path(claw_root).resolve(), Path(out).resolve()
    revision = subprocess.check_output(["git", "-C", str(claw_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != PINNED_REVISION:
        raise ValueError("clawgym_revision_not_pinned")
    exp = project_root / "experiments/cross_harness_sft"
    rows = []
    claw_rows = []
    for path in sorted((claw_root / "RL/data/clawgym_train").glob("*/data_entry.json")):
        data = json.loads(path.read_text())
        row = _record("clawgym", data["task_id"], "pending_source_template", path,
                      source_revision=revision, source_tree_sha256=tree_digest(path.parent))
        row["exclusion_reasons"] = claw_qualification(path)
        row["qualified"] = not row["exclusion_reasons"]
        claw_rows.append(row)
    if len(claw_rows) != 2000:
        raise ValueError(f"clawgym_selected_count_mismatch:{len(claw_rows)}")
    _cluster_claw(claw_rows)
    rows.extend(claw_rows)
    for path in sorted((project_root / "HarnessAudit/single_agent/tasks").rglob("*.yaml")):
        task, catalog = load_sa_task_with_tools(path)
        # Domain + task category is a real, conservative semantic family. All
        # entity/role variants remain together; never use sa-*-tN as family.
        row = _record("harnessaudit", task.task_id, f"harnessaudit:{task.domain}:{task.category or 'domain'}", path)
        row["exclusion_reasons"] = audit_qualification(task)
        row["qualified"] = not row["exclusion_reasons"]
        row["domain"] = task.domain
        row["source_dependencies"] = {"tools_sha256": sha256_file(project_root / "HarnessAudit/single_agent/tools" / (task.domain + ".yaml")),
                                      "fixtures_sha256": tree_digest(project_root / "HarnessAudit/single_agent/fixtures" / ("office_enterprise_management" if task.domain == "office" else task.domain))}
        rows.append(row)
    safe_root = exp / "vendor/harnesssafe/runs/active"
    for path in sorted(safe_root.rglob("case_meta.json")):
        data = json.loads(path.read_text())
        # Original risk family plus source directory template, not numbered case.
        family = f"harnesssafe:{data.get('family', path.parts[-4])}:{path.parent.parent.name}"
        row = _record("harnesssafe", data.get("case_id", str(path.parent.relative_to(safe_root))), family, path,
                      source_tree_sha256=tree_digest(path.parent), local_case_id=path.parent.name,
                      candidate_utility_contract=path.parent.name in UTILITY_CANDIDATES)
        row["exclusion_reasons"] = safe_qualification(path)
        row["qualified"] = not row["exclusion_reasons"]
        if path.parent.name in IMPLEMENTED_UTILITY:
            row["utility_acceptance_evidence"] = accept_utility_checker(path.parent)
        rows.append(row)
        if row["qualified"]:
            rows.append({**row, "condition": "clean"})
    if Counter(r["source"] for r in rows if r["condition"] != "clean") != {"clawgym": 2000, "harnesssafe": 328, "harnessaudit": 34}:
        raise ValueError("source_inventory_count_mismatch")
    for source in REFERENCE_WEIGHTS_V3:
        split_map = _family_splits(rows, source)
        for row in rows:
            if row["source"] == source:
                row["split"] = split_map[row["family"]]
    # Exact task prompt/input duplicates cannot cross source/split unnoticed.
    public_hashes = defaultdict(list)
    for row in rows:
        if row["source"] == "clawgym":
            public_hashes[hashlib.sha256(json.loads(Path(row["source_path"]).read_text())["user_query"].strip().encode()).hexdigest()].append(row)
    duplicate_groups = [[r["id"] for r in group] for group in public_hashes.values() if len(group) > 1]
    identities = [(r["source"], r["id"], r["condition"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate_source_identity")
    qualified = [r for r in rows if r["qualified"]]
    exclusions = [r for r in rows if not r["qualified"]]
    grouped = {split: [r for r in qualified if r["split"] == split] for split in ("train", "dev", "test")}
    ready = all(any(r["source"] == source and r["split"] == "train" for r in qualified) for source in REFERENCE_WEIGHTS_V3)
    payload = {"schema_version": 3, "seed": SEED, "split_method": "family_grouped_static_qualification_stratified_v1", "source_weights": REFERENCE_WEIGHTS_V3, "tasks": rows,
               "train": grouped["train"], "dev": grouped["dev"], "test": grouped["test"],
               "exclusions": exclusions, "source_counts": dict(Counter(r["source"] for r in rows if r["condition"] != "clean")),
               "condition_counts": dict(Counter(r["source"] for r in rows)),
               "qualified_counts": dict(Counter(r["source"] for r in qualified)),
               "qualified_split_counts": {s: dict(Counter(r["source"] for r in group)) for s, group in grouped.items()},
               "exact_query_duplicate_groups": duplicate_groups,
               "ready_for_three_source_training": ready,
               "sealed_test_claim": False, "historical_exposure_audit": "unknown_requires_external_history_ledger",
               "official_clawgym_eval": {"weight": 0, "status": "excluded_without_loading_eval_contents"},
               "adaptation": "whitebox_v3_not_native_leaderboard"}
    payload["manifest_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "task_pool_manifest.json", payload)
    for split, group in grouped.items():
        write_json(out / (split + ".json"), group)
    write_json(out / "exclusions.json", exclusions)
    return payload


def prepare_pool(project_root: Path, claw_root: Path, out: Path) -> dict:
    """V4: official ClawGym selected RL plus frozen SkillLife lifecycle only."""
    from .claw_backend import PINNED_REVISION as CLAW_REVISION, qualification as claw_qualification
    from .skilllife_backend import (PINNED_REVISION as LIFE_REVISION, MANIFEST_SHA256,
        canonical_digest, qualification as life_qualification, source_code_hashes)
    project_root, claw_root, out = Path(project_root).resolve(), Path(claw_root).resolve(), Path(out).resolve()
    exp = project_root / "experiments/cross_harness_sft"
    life_root = exp / "vendor/skilllifebench_security_research_20261116"
    split_path = Path(__file__).resolve().parents[2] / "docs/research/skilllife_lifecycle_split_20261116.json"
    manifest_path = life_root / "data/lifecycle/protocol/task-manifest.json"
    for root, expected in ((claw_root, CLAW_REVISION), (life_root, LIFE_REVISION)):
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if revision != expected:
            raise ValueError("source_revision_not_pinned:" + str(root))
    split = json.loads(split_path.read_text())
    if split["source_revision"] != LIFE_REVISION or split["source_manifest_sha256"] != MANIFEST_SHA256 or sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("skilllife_frozen_manifest_binding_mismatch")
    split_by_id = {row["id"]: row for row in split["tasks"]}
    source_tasks = json.loads(manifest_path.read_text())["tasks"]
    if len(source_tasks) != 60 or len(split_by_id) != 60 or {row["id"] for row in source_tasks} != set(split_by_id):
        raise ValueError("skilllife_frozen_task_identity_mismatch")
    code_hashes = source_code_hashes(life_root)
    rows, claw_rows = [], []
    for path in sorted((claw_root / "RL/data/clawgym_train").glob("*/data_entry.json")):
        data = json.loads(path.read_text())
        row = _record("clawgym", data["task_id"], "pending_source_template", path,
                      source_revision=CLAW_REVISION, source_tree_sha256=tree_digest(path.parent))
        row["exclusion_reasons"] = claw_qualification(path)
        row["qualified"] = not row["exclusion_reasons"]
        row["input_assets_optional"] = True
        if row["exclusion_reasons"]:
            # Preserve concrete source evidence for each capability exclusion.
            row["exclusion_evidence"] = {"user_query_sha256": hashlib.sha256(data["user_query"].encode()).hexdigest(),
                                         "user_query": data["user_query"]}
        claw_rows.append(row)
    if len(claw_rows) != 2000:
        raise ValueError("clawgym_selected_count_mismatch")
    _cluster_claw(claw_rows)
    claw_splits = _family_splits(claw_rows, "clawgym")
    for row in claw_rows:
        row["split"] = claw_splits[row["family"]]
    rows.extend(claw_rows)
    for record in source_tasks:
        frozen = split_by_id[record["id"]]
        if any(record[key] != frozen[key] for key in ("template_id", "parent_id", "class")):
            raise ValueError("skilllife_frozen_template_binding_mismatch")
        if frozen["split"] not in {"train", "dev", "test"} or set(frozen["conditions"]) != {"clean", "attack"}:
            raise ValueError("skilllife_frozen_split_contract_error")
        for condition in frozen["conditions"]:
            reasons = life_qualification(record)
            rows.append(_record("skilllifebench", record["id"], "skilllifebench:" + record["template_id"], manifest_path,
                source_revision=LIFE_REVISION, source_root=str(life_root), source_record_sha256=canonical_digest(record),
                source_code_hashes=code_hashes, template_id=record["template_id"], parent_id=record["parent_id"],
                risk_class=record["class"], condition=condition, split=frozen["split"],
                qualified=not reasons, exclusion_reasons=reasons, source_track="autonomous_lifecycle",
                runtime_admission="requires_actual_source_executor_receipts_in_R1"))
    identities = [(row["source"], row["id"], row["condition"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate_task_condition_identity")
    qualified = [row for row in rows if row["qualified"]]
    grouped = {s: [row for row in qualified if row["split"] == s] for s in ("train", "dev", "test")}
    exclusions = [row for row in rows if not row["qualified"]]
    payload = {"schema_version": 4, "protocol": "clawgym_skilllife_online_rl_v4", "seed": SEED,
        "source_weights": WEIGHTS, "source_condition_weights": {"clawgym": {"benign": 1.0}, "skilllifebench": {"clean": .5, "attack": .5}},
        "source_counts": {"clawgym": len(claw_rows), "skilllifebench": len(source_tasks)},
        "condition_counts": dict(Counter(row["source"] for row in rows)),
        "qualified_counts": dict(Counter(row["source"] for row in qualified)),
        "qualified_instance_counts": {source: len({row["id"] for row in qualified if row["source"] == source}) for source in WEIGHTS},
        "family_counts": {source: len({row["family"] for row in rows if row["source"] == source}) for source in WEIGHTS},
        "qualified_split_counts": {s: dict(Counter(row["source"] for row in group)) for s, group in grouped.items()},
        "tasks": rows, "train": grouped["train"], "dev": grouped["dev"], "test": grouped["test"], "exclusions": exclusions,
        "skilllife_frozen_split": {"path": str(split_path), "sha256": sha256_file(split_path), "template_counts": split["template_counts"],
                                   "instance_counts": split["instance_counts"], "condition_counts": split["condition_counts"]},
        "split_method": {"clawgym": "source_template_90_10_seed20261116", "skilllifebench": "reuse_frozen_BDE_2train_1dev_1test_templates_no_resampling"},
        "reference_sources": {"harnesssafe": {"weight": 0, "source_count": 328, "status": "reference_only_preserved_v3_evidence"},
                              "harnessaudit": {"weight": 0, "source_count": 34, "status": "reference_only_preserved_v3_evidence"}},
        "official_clawgym_eval": {"weight": 0, "status": "excluded_without_loading_eval_contents"},
        "ready_for_two_source_static_dispatch": all(any(row["source"] == source for row in grouped["train"]) for source in WEIGHTS),
        "runtime_admission_verified": False, "sealed_test_claim": False,
        "exposure_audit": split["exposure_audit"], "adaptation": "common_whitebox_v4_not_native_product_harness"}
    payload["manifest_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "task_pool_manifest.json", payload)
    for split_name, group in grouped.items():
        write_json(out / (split_name + ".json"), group)
    write_json(out / "exclusions.json", exclusions)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--claw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_pool(args.project_root, args.claw_root, args.out)
    print(json.dumps({key: manifest[key] for key in ("manifest_sha256", "source_counts", "qualified_counts", "qualified_split_counts", "ready_for_two_source_static_dispatch")}))


if __name__ == "__main__":
    main()
