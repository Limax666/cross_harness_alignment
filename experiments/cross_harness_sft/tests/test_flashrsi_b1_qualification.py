import hashlib
import json

import pytest

import flashrsi_run_gepa_security_agentdojo as runner


def test_formal_run_rejects_calibration_only_qualification(tmp_path, monkeypatch):
    split = tmp_path / "split.jsonl"
    split.write_text("frozen split\n")
    digest = hashlib.sha256(split.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"family_split_manifest_sha256": digest,
                                    "model_revision": runner.MODEL_REVISION,
                                    "flashrsi_revision": runner.FLASHRSI_REVISION}))
    row = {"manifest_sha256": digest, "same_user_task": True, "same_tool_schemas": True,
           "required_facts_visible_both": True, "reset_verified_both": True,
           "attack_oracle_negative": False, "attack_oracle_positive": True,
           "utility_oracle_positive_both": True,
           "status": "calibration_candidate_only_not_formal_admission"}
    qualification = tmp_path / "qualification.json"
    qualification.write_text(json.dumps({"families": [row, row]}))
    monkeypatch.setattr(runner, "SPLIT_MANIFEST", split)
    monkeypatch.setattr(runner, "MANIFEST", manifest)
    monkeypatch.setattr(runner, "QUALIFICATION", qualification)
    with pytest.raises(RuntimeError, match="formal"):
        runner._preflight()
