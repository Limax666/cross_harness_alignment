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

from .backends import sha256_file, tree_digest, write_json, verify_pinned_tree

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



def validate_lifecycle_split(split: dict, source_tasks: list[dict]) -> dict:
    """Recompute the approved template assignment, independent of mutable labels."""
    if split.get("seed") != SEED:
        raise ValueError("lifecycle_split_seed_changed")
    by_id = {row["id"]: row for row in source_tasks}
    frozen_rows = split.get("tasks", [])
    if len(source_tasks) != 60 or len(by_id) != 60 or len(frozen_rows) != 60 or {row["id"] for row in frozen_rows} != set(by_id):
        raise ValueError("lifecycle_split_identity_or_duplicate_error")
    template_classes, template_parents, template_sizes = {}, {}, Counter()
    parents_to_template = {}
    for row in source_tasks:
        template, risk, parent = row["template_id"], row["class"], row["parent_id"]
        if risk not in {"B", "D", "E"} or template_classes.get(template, risk) != risk or template_parents.get(template, parent) != parent:
            raise ValueError("lifecycle_template_class_or_parent_crossing")
        if parents_to_template.get(parent, template) != template:
            raise ValueError("lifecycle_parent_crosses_template")
        template_classes[template], template_parents[template] = risk, parent
        parents_to_template[parent] = template
        template_sizes[template] += 1
    if len(template_sizes) != 12 or set(template_sizes.values()) != {5}:
        raise ValueError("lifecycle_template_instance_counts_changed")
    expected = {}
    for risk in ("B", "D", "E"):
        templates = [template for template, value in template_classes.items() if value == risk]
        if len(templates) != 4:
            raise ValueError("lifecycle_class_template_count_changed")
        templates.sort(key=lambda template: hashlib.sha256(f"{SEED}|{template}".encode()).hexdigest())
        expected.update({template: "train" if index < 2 else "dev" if index == 2 else "test" for index, template in enumerate(templates)})
    instance_counts, condition_counts = Counter(), Counter()
    for row in frozen_rows:
        source = by_id[row["id"]]
        if any(row.get(key) != source[key] for key in ("template_id", "parent_id", "class")):
            raise ValueError("lifecycle_split_source_binding_changed")
        if row.get("split") != expected[source["template_id"]] or sorted(row.get("conditions", [])) != ["attack", "clean"]:
            raise ValueError("lifecycle_deterministic_split_or_pair_changed")
        instance_counts[row["split"]] += 1
        condition_counts[row["split"]] += len(row["conditions"])
    counts = Counter(expected.values())
    if counts != {"train": 6, "dev": 3, "test": 3} or instance_counts != {"train": 30, "dev": 15, "test": 15} or condition_counts != {"train": 60, "dev": 30, "test": 30}:
        raise ValueError("lifecycle_actual_split_counts_changed")
    if split.get("template_counts") != dict(counts) or split.get("instance_counts") != dict(instance_counts) or split.get("condition_counts") != dict(condition_counts):
        raise ValueError("lifecycle_declared_split_counts_inconsistent")
    declared_templates = split.get("templates", [])
    if len(declared_templates) != 12 or {row["template_id"] for row in declared_templates} != set(expected):
        raise ValueError("lifecycle_declared_template_table_inconsistent")
    for row in declared_templates:
        if row.get("split") != expected[row["template_id"]] or row.get("parent_id") != template_parents[row["template_id"]] or row.get("class") != template_classes[row["template_id"]]:
            raise ValueError("lifecycle_declared_template_assignment_inconsistent")
    return {"verified": True, "rule": "sha256(20261116|template)_BDE_2train_1dev_1test",
            "template_counts": dict(counts), "instance_counts": dict(instance_counts),
            "condition_counts": dict(condition_counts), "assignment_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()}



def _tree_digest_from_hashes(files: dict[str, str]) -> str:
    value = hashlib.sha256()
    for name in sorted(files, key=lambda name: Path(name).parts):
        value.update(name.encode())
        value.update(b"\0")
        value.update(bytes.fromhex(files[name]))
    return value.hexdigest()


def verify_hf_task_assets(task_dir: Path, expected: dict[str, str]) -> None:
    if task_dir.is_symlink():
        raise ValueError("HF_consumed_task_symlink")
    actual = {}
    for path in task_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError("HF_consumed_asset_symlink:" + str(path))
        if path.is_file():
            actual[path.relative_to(task_dir).as_posix()] = sha256_file(path)
    if set(actual) != set(expected):
        raise ValueError("HF_consumed_asset_file_set_mismatch:" + task_dir.name)
    for name in expected:
        if actual[name] != expected[name]:
            raise ValueError("HF_consumed_asset_bytes_mismatch:" + task_dir.name + "/" + name)


def verify_hf_selected_source(tasks_root: Path, selection_repo: Path) -> dict:
    """Independently derive all selected asset hashes from fixed raw HF bytes.

    HF_MATERIALIZATION is compared to this derivation as an index, never used as
    a self-signed trust root. No source code or payload is executed.
    """
    from .claw_backend import HF_REVISION, HF_SHA256, PINNED_REVISION
    tasks_root, selection_repo = Path(tasks_root).resolve(), Path(selection_repo).resolve()
    index = json.loads((tasks_root.parent / "HF_MATERIALIZATION.json").read_text())
    if index.get("hf_revision") != HF_REVISION or index.get("raw_sha256") != HF_SHA256 or index.get("selection_revision") != PINNED_REVISION:
        raise ValueError("HF_materialization_revision_binding_mismatch")
    selection_proof = verify_pinned_tree(selection_repo, PINNED_REVISION, ["RL/data/clawgym_train"])
    selected_root = selection_repo / "RL/data/clawgym_train"
    if index.get("repo") != "RUC-AIBOX/ClawGym-Task" or Path(index.get("selection_root", "")).resolve() != selected_root:
        raise ValueError("HF_materialization_index_selection_origin_mismatch")
    selected = {}
    for path in sorted(selected_root.glob("*/data_entry.json")):
        entry = json.loads(path.read_text())
        prompt_sha = hashlib.sha256(entry["user_query"].strip().encode()).hexdigest()
        if prompt_sha in selected:
            raise ValueError("ambiguous_pinned_selection_prompt")
        selected[prompt_sha] = (path.parent.name, entry, path)
    if len(selected) != 2000:
        raise ValueError("pinned_selection_count_mismatch")
    raw = Path(index["raw_path"]).resolve(strict=True)
    records, restored, seen = {}, [], set()
    raw_digest, raw_count, existing_asset_count = hashlib.sha256(), 0, 0
    with raw.open("rb") as stream:
        for raw_count, line in enumerate(stream, 1):
            raw_digest.update(line)
            row = json.loads(line)
            prompt_sha = hashlib.sha256(row["prompt"].strip().encode()).hexdigest()
            if prompt_sha not in selected:
                continue
            task_id, entry, entry_path = selected[prompt_sha]
            if task_id in seen:
                raise ValueError("duplicate_selected_HF_record")
            seen.add(task_id)
            source_assets, expected = set(), {}
            hf_asset_hashes = {}
            for asset in row["input_files"]:
                name = Path(asset["file_path"])
                if name.is_absolute() or ".." in name.parts or not name.parts or "\\" in asset["file_path"]:
                    raise ValueError("unsafe_HF_source_asset_path")
                if name.as_posix() in source_assets:
                    raise ValueError("duplicate_HF_source_asset_path")
                source_assets.add(name.as_posix())
                if name.parts[0] == "reward":
                    if len(name.parts) < 2:
                        raise ValueError("invalid_HF_reward_path")
                    output = Path("reward/check.py") if name.parts[1:] == ("test.py",) else name
                else:
                    output = Path("input_files") / name
                key = output.as_posix()
                if key in expected:
                    raise ValueError("colliding_HF_adapted_asset_path")
                data = asset["content"].encode()
                expected[key] = hashlib.sha256(data).hexdigest()
                hf_asset_hashes[key] = expected[key]
                old = selected_root / task_id / output
                if old.exists():
                    if old.read_bytes() != data:
                        raise ValueError("pinned_Git_asset_differs_from_selected_HF_record:" + task_id + "/" + key)
                    existing_asset_count += 1
                else:
                    restored.append({"task": task_id, "path": key, "sha256": expected[key]})
            entry_bytes = (json.dumps(entry, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
            expected["data_entry.json"] = hashlib.sha256(entry_bytes).hexdigest()
            wrapper = selected_root / task_id / "reward/reward.sh"
            expected["reward/reward.sh"] = sha256_file(wrapper)
            verify_hf_task_assets(tasks_root / task_id, expected)
            record = {"id": task_id, "hf_line": raw_count, "prompt_sha256": prompt_sha,
                "hf_record_sha256": hashlib.sha256(line.rstrip(b"\r\n")).hexdigest(),
                "hf_record_canonical_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                "file_hashes": expected, "files": hf_asset_hashes, "tree_sha256": _tree_digest_from_hashes(expected),
                "selection_entry_sha256": sha256_file(entry_path)}
            records[task_id] = record
    if raw_digest.hexdigest() != HF_SHA256:
        raise ValueError("HF_raw_bytes_differ_from_frozen_revision")
    if raw_count != 15402 or len(records) != 2000 or {p.name for p in tasks_root.iterdir()} != set(records):
        raise ValueError("HF_fixed_selected_2000_identity_or_raw_count_mismatch")
    declared_records = index.get("records", [])
    declared = {record["id"]: record for record in declared_records}
    if len(declared_records) != 2000 or set(declared) != set(records) or index.get("selection_count") != 2000:
        raise ValueError("HF_materialization_index_identity_mismatch")
    for task_id, record in records.items():
        for key in ("hf_line", "prompt_sha256", "hf_record_sha256", "files", "tree_sha256"):
            if declared[task_id].get(key) != record[key]:
                raise ValueError("HF_materialization_index_differs_from_fixed_raw_derivation:" + task_id + ":" + key)
    signature = lambda rows: {(row["task"], row["path"], row["sha256"]) for row in rows}
    if signature(index.get("restored_HF_input_files", [])) != signature(restored):
        raise ValueError("HF_restored_asset_index_mismatch")
    return {"verified": True, "format": "hf_selected2k", "repo": "RUC-AIBOX/ClawGym-Task",
        "hf_revision": HF_REVISION, "hf_sha256": HF_SHA256, "raw_path": str(raw), "raw_record_count": raw_count,
        "selection_revision": PINNED_REVISION, "selected_count": len(records),
        "existing_Git_asset_count": existing_asset_count, "restored_input_count": len(restored),
        "restored_HF_input_files": restored, "selection_pinned_bytes": selection_proof,
        "materialization_index_sha256": sha256_file(tasks_root.parent / "HF_MATERIALIZATION.json"),
        "adaptations": ["reward/test.py renamed reward/check.py without changing bytes", "data_entry serialized as canonical sorted JSON from pinned selection", "official reward.sh copied from pinned Git selection"],
        "records": records}


def prepare_pool(project_root: Path, claw_root: Path, out: Path) -> dict:
    """V4: official ClawGym selected RL plus frozen SkillLife lifecycle only."""
    from .claw_backend import PINNED_REVISION as CLAW_REVISION, qualification as claw_qualification, source_contract
    from .skilllife_backend import (PINNED_REVISION as LIFE_REVISION, MANIFEST_SHA256,
        canonical_digest, qualification as life_qualification, source_code_hashes)
    project_root, claw_root, out = Path(project_root).resolve(), Path(claw_root).resolve(), Path(out).resolve()
    exp = project_root / "experiments/cross_harness_sft"
    life_root = exp / "vendor/skilllifebench_security_research_20261116"
    split_path = Path(__file__).resolve().parents[2] / "docs/research/skilllife_lifecycle_split_20261116.json"
    manifest_path = life_root / "data/lifecycle/protocol/task-manifest.json"
    hf_proof = None
    if (claw_root.parent / "HF_MATERIALIZATION.json").is_file():
        selection_repo = exp / "vendor/clawgym_agents_hsrl_v3_0e8526d"
        hf_proof = verify_hf_selected_source(claw_root, selection_repo)
        claw_tasks_root = claw_root
        claw_pinned = {key: value for key, value in hf_proof.items() if key != "records"}
    else:
        claw_tasks_root = claw_root / "RL/data/clawgym_train"
        claw_pinned = verify_pinned_tree(claw_root, CLAW_REVISION, ["RL/data/clawgym_train"])
    pinned_sources = {"clawgym": claw_pinned,
                      "skilllifebench": verify_pinned_tree(life_root, LIFE_REVISION, ["skilllifebench", "data/lifecycle/protocol/task-manifest.json"])}
    split = json.loads(split_path.read_text())
    if split["source_revision"] != LIFE_REVISION or split["source_manifest_sha256"] != MANIFEST_SHA256 or sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("skilllife_frozen_manifest_binding_mismatch")
    split_by_id = {row["id"]: row for row in split["tasks"]}
    source_tasks = json.loads(manifest_path.read_text())["tasks"]
    if len(source_tasks) != 60 or len(split_by_id) != 60 or {row["id"] for row in source_tasks} != set(split_by_id):
        raise ValueError("skilllife_frozen_task_identity_mismatch")
    split_admission = validate_lifecycle_split(split, source_tasks)
    code_hashes = source_code_hashes(life_root)
    rows, claw_rows = [], []
    for path in sorted(claw_tasks_root.glob("*/data_entry.json")):
        data = json.loads(path.read_text())
        row = _record("clawgym", data["task_id"], "pending_source_template", path,
                      source_revision=CLAW_REVISION, source_tree_sha256=tree_digest(path.parent))
        if hf_proof is not None:
            record = hf_proof["records"][row["id"]]
            row["source_revision"] = hf_proof["hf_revision"]
            row["source_sha256"] = record["file_hashes"]["data_entry.json"]
            row["source_tree_sha256"] = record["tree_sha256"]
            row["source_identity"] = {"format": "hf_selected2k", "hf_revision": hf_proof["hf_revision"],
                "hf_sha256": hf_proof["hf_sha256"], "selection_revision": CLAW_REVISION,
                "official_selection_id": row["id"], "hf_line": record["hf_line"],
                "hf_record_sha256": record["hf_record_sha256"],
                "hf_record_canonical_sha256": record["hf_record_canonical_sha256"],
                "selection_entry_sha256": record["selection_entry_sha256"]}
        row["input_contract"] = source_contract(path)
        row["exclusion_reasons"] = claw_qualification(path, contract=row["input_contract"], source_identity=row.get("source_identity"))
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
        "pinned_consumed_sources": pinned_sources, "split_admission": split_admission,
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
    if hf_proof is not None:
        write_json(out / "hf_consumed_source_proof.json", hf_proof)
    write_json(out / "claw_source_contract_audit.json", {"source_task_count": len(claw_rows),
        "method": "user_goal_clauses_plus_actual_fixtures_and_checker_AST_not_URL_presence",
        "tasks": [{"id": row["id"], "qualified": row["qualified"], "reasons": row["exclusion_reasons"],
                   "contract": row["input_contract"]} for row in claw_rows]})
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
