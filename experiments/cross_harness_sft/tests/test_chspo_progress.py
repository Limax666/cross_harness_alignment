import json

import pytest

from a6000_vm import chspo_progress as progress


# Native authentication has separate receipt tests. These fixtures isolate the
# cross-receipt comparison performed after each individual native audit.
def compare(tmp_path, monkeypatch, rows):
    eps = []
    paths = []
    audited = []
    monkeypatch.setattr(progress, "audit_episode", lambda ep, path: audited.append(path))
    for index, (final, scores, options) in enumerate(rows):
        path = tmp_path / str(index)
        backend = path / "private/backend"
        backend.mkdir(parents=True)
        (backend / "provenance.json").write_text(json.dumps({"source_tree_sha256": "source"}))
        values = scores.values() if isinstance(scores, dict) else [scores]
        result = dict(utility=sum(values) / len(values), c_protocol=options.get("cost", 0),
                      evidence=dict(final_assets_sha256=final, initial_assets_sha256="reset",
                                    private_grader=dict(uuid=str(index), nonce=options.get("nonce", str(index)),
                                                        case_id=options.get("case", "case"),
                                                        original_checker_sha256="checker")),
                      metrics=dict(official_checker_scores=scores))
        eps.append(dict(key=["clawgym", "fixture"], policy_version=0, source_result=result,
                        actions=[dict(context_only=False, canonical_name=options.get("tool", "write_file"))]))
        paths.append(path)
    gains = progress.group_progress(eps, paths)
    assert audited == paths
    return gains


def test_changed_workspace_gets_only_official_gain_above_reset(tmp_path, monkeypatch):
    assert compare(tmp_path, monkeypatch, [
        ("reset", {"a": 1, "b": 0}, {}), ("edit", {"a": 1, "b": .5}, {}),
        ("same_score", {"a": 1, "b": 0}, {}), ("tradeoff", {"a": .5, "b": 1}, {}),
        ("changed_schema", {"b": 1}, {}), ("read", {"a": 1, "b": 1}, {"tool": "read_file"}),
        ("cost", {"a": 1, "b": 1}, {"cost": 1})]) == [0, .25, 0, None, None, None, None]


@pytest.mark.parametrize("rows", [
    [("edited", .75, {}), ("edited2", 1, {})],
    [("reset", .5, {}), ("reset", .75, {}), ("edit", 1, {})],
    [("reset", .5, {}), ("edit", 1, {}), ("edit", .75, {})],
])
def test_abstains_without_consistent_comparator(tmp_path, monkeypatch, rows):
    assert compare(tmp_path, monkeypatch, rows) == [None] * len(rows)


@pytest.mark.parametrize("override", [{"case": "other"}, {"nonce": "0"}])
def test_rejects_mismatched_or_reused_authority(tmp_path, monkeypatch, override):
    with pytest.raises(ValueError, match="identity|reused"):
        compare(tmp_path, monkeypatch, [("reset", .25, {}), ("edited", .75, override)])


def test_other_source_abstains_without_opening_files(tmp_path):
    assert progress.group_progress([dict(key=["agentdog", "fixture"], policy_version=2)], [tmp_path]) == [None]
