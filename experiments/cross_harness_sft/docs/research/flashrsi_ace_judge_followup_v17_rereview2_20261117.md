# ACE judge follower scoped rereview 2 — 2026-11-17

**Verdict: PASS for the reviewed follower change at `1072498`. No unresolved Critical or Important findings in this review scope.** The remaining Important finding from rereview 1 is closed. This is source/controller review clearance for the already-authorized finite follower, not evidence that its real wait or judge run has completed.

Reviewed `5731dd8..1072498`, the supplied fix2 diff, current follower, implementation report and retained green log. Scope was nonzero-child preservation and new breakage in that correction. No tests were rerun and no source execution, API/model/GPU/service/cron or other live actions were performed. Only this report was written.

`scripts/flashrsi_ace_judge_followup_v17.py:39–42` now gives any retained nonzero judge exit the explicit status `needs_judge_process_repair`, even if strict artifact admission is complete. Finalization at lines 85–89 uses that status for the saved task, terminal event and process exit. It preserves the raw return code and independent artifact acceptance metadata; a nonzero child can no longer become follower exit 0 merely because its 128-case manifest was written before failure.

The absent-return-code cached path remains eligible for completion only through the unchanged final strict acceptance. A zero child exit still requires `accepted.status=complete`; incomplete admission remains nonzero. The fix adds no retry, overwrite or judge dispatch, so complete cached case successes remain reusable. The previous immutable PID/start-tick wait, nonempty pending-list deferral condition and final source-binding recheck remain present.

The added focused assertions cover cached completion, successful child completion, nonzero child with complete artifacts, and zero child with incomplete admission. The retained `flashrsi_ace_judge_followup_v17_fix2_green.log` records **4 passed in 0.15s**; this reviewer read that log and the test diff rather than rerunning them. No new control-flow issue was found in this narrow fix.

Real waiting, API execution, billing behavior and actual final 128-case judge acceptance remain runtime work for the finite follower. The active strict ACE model stage and all existing source, privacy, cache, shared-worker and billing gates remain unchanged by this correction.
