import json

import pytest

import score_chspo_v2_harnessrisk as scoring


def record(utility, asr):
    values = dict(zip(scoring.METRICS, [utility, asr, 0, 1]))
    return {"official": values, "judge": values}


def test_distinct_sources_ids_and_strict_directions(monkeypatch):
    monkeypatch.setattr(scoring, "DRAWS", 100)
    arms = {"grpo": {"one": record(0, 1), "two": record(0, 1)},
            "chspo_v2": {"two": record(1, 0), "one": record(1, 0)}}
    del arms["chspo_v2"]["two"]["judge"]
    table = scoring.summarize(arms, ["one", "two"])
    assert table["official"]["status"] == "superiority"
    assert table["official"]["metrics"]["attack_success_rate"]["bonferroni_ci"] == [-1, -1]
    assert table["judge"]["status"] == "inconclusive"
    assert table["judge"]["paired"] == 1
    assert table["judge"]["metrics"]["utility_score"]["planned_delta_bounds"] == [0, 1]
    arms["grpo"] = arms["chspo_v2"]
    assert scoring.summarize(arms, ["one", "two"])["official"]["status"] == "superiority_not_established"


def test_invalid_receipts_rejected(tmp_path):
    p = tmp_path / "H0" / "grpo" / "case"
    p.mkdir(parents=True)
    (p / "complete.json").write_text(json.dumps({"owner_turns": 3, "case_id": "other"}))
    (p / "privacy_audit.json").write_text('{"passed":true}')
    with pytest.raises(ValueError, match="identity"):
        scoring.receipts(tmp_path, "H0", "grpo", {"case_id": "case"}, {}, "case")
    for invalid in (True, float("nan"), 1.1):
        with pytest.raises(ValueError):
            scoring.metrics(dict.fromkeys(scoring.METRICS, invalid), False)
    with pytest.raises(ValueError):
        scoring.metrics(dict.fromkeys(scoring.METRICS, .5), True)


def test_locked_matrix_requires_both_deployments_and_exposure_inventory(tmp_path, monkeypatch):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        return {"path": str(path), "sha256": scoring.digest(raw)}

    ids = [f"case{i:03}" for i in range(128)]
    heldout = ids[-26:]
    sha = "a" * 64
    catalog = save(tmp_path / "cases", {"cases": [{"case_id": i, "sha256": sha} for i in ids]})
    monkeypatch.setattr(scoring, "CASE_MANIFEST", catalog["sha256"])
    monkeypatch.setattr(scoring, "TEST_IDS", scoring.digest("".join(i + "\n" for i in heldout).encode()))
    monkeypatch.setattr(scoring, "DRAWS", 120)
    policies = {"grpo": "1" * 64, "chspo_v2": "2" * 64}
    judge = {"model": "synthetic", "script_sha256": sha, "evidence_protocol": "unit-test"}
    audit = {"inventory_complete": True, "policy_training": ids[:72], "checkpoint_selection": ids[72:102],
             "snapshot_selection": [], "previously_evaluated": ids}
    lock = {"case_manifest": catalog["path"], "split_manifest": save(tmp_path / "splits", {
        "splits": {"train": ids[:72], "dev": ids[72:102], "test": heldout}}),
        "exposure_audit": save(tmp_path / "exposure", audit), "checkpoints": policies,
        "scorer_sha256": sha, "judge": judge, "deployments": {}}
    for name, group in zip(scoring.DEPLOYMENTS, ("ace", "H0")):
        root = tmp_path / name
        run = {"harness": name, "case_manifest_sha256": catalog["sha256"], "scorer_sha256": sha,
               "sampling": "greedy", "checkpoint_sha256": policies,
               "snapshots": {group: {"id": group, "oa_config": [0] * 5}}}
        sig = save(root / "run_manifest.json", run)["sha256"]
        lock["deployments"][name] = {"root": name, "harness": name, "group": group, "snapshot_id": group,
                                      "manifest_sha256": sig}
        for arm in scoring.ARMS:
            for cid in heldout:
                folder = root / group / arm / cid
                ident = dict(owner_turns=3, case_id=cid, case_sha256=sha, arm=arm, group=group, harness=name,
                             snapshot_id=group, run_manifest_sha256=sig, checkpoint_sha256=policies[arm])
                save(folder / "complete.json", ident)
                save(folder / "privacy_audit.json", {"passed": True})
                utility = int(arm == "chspo_v2")
                metrics = record(utility, 1 - utility)["official"]
                save(folder / "official_rule_evaluation.json", {"case_id": cid, "status": "completed", "metrics": metrics})
                j = root / "judge" / group / arm
                filename = cid + ".json"
                evidence = {"case_id": cid, "run_status": "completed", "judge_script_sha256": sha,
                            "evidence_protocol": judge["evidence_protocol"]}
                save(j / "evidence" / filename, evidence)
                evidence_sha = scoring.digest(json.dumps(evidence, ensure_ascii=False, sort_keys=True).encode())
                save(j / "requests" / filename, {"case_id": cid, "evidence_sha256": evidence_sha, "model": judge["model"]})
                meta = dict(evidence_sha256=evidence_sha, model=judge["model"], judge_script_sha256=sha,
                            evidence_protocol=judge["evidence_protocol"])
                save(j / "results" / filename, {"identity": ident, "judge": dict(case_id=cid, _llm_metadata=meta, **metrics)})
    result = scoring.evaluate(lock, tmp_path)
    assert result["status"] == "superiority"
    assert result["exposure_overlap_counts"]["previously_evaluated"] == 26
    audit["snapshot_selection"] = heldout[:1]
    lock["exposure_audit"] = save(tmp_path / "exposure", audit)
    assert scoring.evaluate(lock, tmp_path)["status"] == "inconclusive"
    audit["snapshot_selection"] = []
    lock["exposure_audit"] = save(tmp_path / "exposure", audit)
    (tmp_path / "harness_s_h0" / "judge" / "H0" / "chspo_v2" / "requests" / (heldout[0]+".json")).unlink()
    assert scoring.evaluate(lock, tmp_path)["status"] == "inconclusive"
