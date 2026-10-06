"""Regression checks for family isolation and the SFT safety gate."""

import importlib.util
import json
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_teacher_sft_release.py"
SPEC = importlib.util.spec_from_file_location("teacher_sft_release", SCRIPT)
release = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(release)


class TeacherReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lookup, _ = release.source_lookups()

    def test_agentharm_arms_and_variants_share_one_family(self):
        rows = [
            {"benchmark": "agentharm", "task_id": f"agentharm:{variant}", "task_type": arm}
            for arm in ("clean", "query") for variant in ("1-1", "1-2", "1-3", "1-4")
        ]
        specs = [release.source_spec(row, self.lookup) for row in rows]
        self.assertEqual(len({spec["source_family_id"] for spec in specs}), 1)
        self.assertEqual({spec["primary_behavior_type"] for spec in specs},
                         {"benign_tool_completion", "harmful_request_refusal"})

    def test_actbench_paired_arms_share_one_family(self):
        task_id = next(iter(self.lookup["actbench"]))
        rows = [{"benchmark": "actbench", "task_id": task_id, "task_type": arm,
                 "raw_payload": {"task_id": task_id}} for arm in ("benign", "attack")]
        specs = [release.source_spec(row, self.lookup) for row in rows]
        self.assertEqual(len({spec["source_family_id"] for spec in specs}), 1)
        self.assertTrue(all(spec["mapping_status"] == "mapped" for spec in specs))

    def test_judge_cannot_accept_side_effect_or_mismatched_category(self):
        candidate = {"primary_behavior_type": "sensitive_data_boundary"}
        verdict = {"decision": "accept", "category": "verified_safe_attack_defense",
                   **{score: 4 for score in release.SCORES}, "issue_tags": ["side_effect"],
                   "rationale": "The helper command inadvertently triggered an email.", "call_status": "ok"}
        self.assertEqual(release.enforce_verdict_policy(candidate, verdict)["decision"], "reject")
        verdict["issue_tags"] = []
        verdict["rationale"] = "No harmful action observed."
        verdict["category"] = "verified_benign_tool_completion"
        self.assertEqual(release.enforce_verdict_policy(candidate, verdict)["decision"], "review")
        verdict["category"] = "verified_safe_attack_defense"
        verdict["rationale"] = "No unauthorized actions or unsafe side effects occurred."
        self.assertEqual(release.enforce_verdict_policy(candidate, verdict)["decision"], "accept")


if __name__ == "__main__":
    unittest.main()
