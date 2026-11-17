"""Read-only actual retained-run admission checks; never call a judge/model."""
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / x) for x in ('scripts', 'src', 'vendor/FlashRSI')]


def test_actual_full_opro_is_eligible_but_incomplete_ace_and_wrong_case_set_are_not():
    from flashrsi_ace_judge_followup_v17 import eligible
    from flashrsi_matrix_gpu1_queue_v17 import REFERENCE, read
    expected = {r['case_id'] for r in read(REFERENCE.parent / 'case_manifest.json')['cases']}
    full = REFERENCE.parent / 'opencode_opro_full'
    assert eligible(full, expected)
    incomplete = full.parents[1] / 'flashrsi_harnessrisk_4b_private_opencode_v17_gpu1_20261117/opencode_ace_full'
    # No assertion about this actively running job: readiness must agree with
    # genuine final128 acceptance rather than its process or attempts count.
    if not (incomplete / 'result.json').exists():
        assert not eligible(incomplete, expected)
    assert not eligible(full, expected - {'action_041'})
