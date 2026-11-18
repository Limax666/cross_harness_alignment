# OpenClaw context v19 scoped rereview 1 — 2026-11-17

**Verdict: PASS at `9aa5b26` for the fix through `e65c524`/`9aa5b26`. Original Important finding closed; no new Critical or Important findings found in this scope.** This clears the reviewed context/projection wiring for the already-authorized fresh profile. It does not establish that real H0, tool transitions, token fit, six RSI stages or judge have completed.

Scope: the original first-wire/compaction conflict, new projection observer/auditor APIs and resulting integration changes. Read the supplied fix diff, implementation update, current observer/auditor/entry/supervisor, relevant selected-runtime APIs and retained green log. Only this report was written. No tests were rerun; no source task/program, model/API/GPU/Docker/service/cron or other live actions were performed.

## Original finding resolution

The new entry installs `ProjectionObserver` around the actual selected native `run`. The registered runtime re-exports `loop.run` directly, so `native.run.__globals__['compact']` is the function the loop actually calls. The observer deep-copies inputs/arguments/output for observation, delegates to the original function with unchanged arguments and returns its original result, then restores the original global on exit. It changes no Snapshot or native file bytes and does not enable disabled compaction.

The versioned v4 auditor retains the original immutable SQLite prefix, complete ordered owner ledger, actual session path and nonempty model-request gates. Its first-wire expectation now derives from authenticated native projection, so a genuine compacted request is no longer forced to equal the full uncompacted SQLite transcript. Empty projection receipts retain the original full-history first-wire comparison for disabled compaction.

For observed compaction, `authenticate_projection` binds the first raw input to SQLite-before plus current owner; requires one projection per actual forwarded request; checks the exact argument shape and a positive input limit no larger than 24576; recomputes native compaction; and compares the output and registered provider wire conversion against every actual request. Subsequent inputs must preserve the prior projected prefix and append the matching raw SQLite delta. The limited fresh-envelope conversion matches the inspected native loop: SQLite adds assistant `stop_reason` and tool `receipt`, while the fresh provider messages omit those fields and retain copied receipt attachments (`openclaw_native/loop.py:259–264,346–352,386–411`). This preserves original raw evidence rather than rewriting it to fit the wire.

## Source identity and confinement

The independent supervisor-side check reads the actual exported Snapshot's three critical files and requires their hashes to match code-owned `compaction.py`, `models.py` and `providers.py` pins, plus the entry's source receipt. Read-only SHA checks against the current code-owned files matched all three declared pins. The supervisor recomputes only the pinned original pure compaction/model modules through a separately named package; it does not import or execute candidate code from the exported Snapshot. The original registered provider conversion remains the wire comparison.

This explicitly restricts compatibility: mutations of those three critical native modules, unsupported observer APIs or incompatible projection/transition behavior remain invalid under strict admission. They are not scored zero, silently repaired or accepted by a summary marker. The implementation update documents that restriction for Evo-Harness and other candidate code evolution; results must retain it as part of this new profile. Other selected resource/native changes are not rewritten by this fix and must still satisfy the observation contract.

No sandbox/import-root/network expansion, new output/turn/time allowance, business-state reset, owner rewrite or source-score change is introduced. The v19 fresh H0 and six same-profile RSI contract, inherited source freeze/admission/cleanup/lease/billing/cache guards, and separate historical results remain unchanged by this fix.

## Evidence and remaining limits

The new deterministic regression uses actual pinned native compaction on a long history: projection differs from raw input, the old audit rejects it, the new audit accepts the exact projection/wire, and altered projected tail or raw SQLite prefix is rejected. The retained `docs/research/flashrsi_openclaw_context_v19_fix1_green_final.log` records **24 passed in 1.24s** across the affected four test files. This reviewer read that evidence; the implementer ran it.

The added regression covers the previously broken boundary. Actual observer operation inside the real isolated entry, multiple genuine tool transitions, request/context usage and native source compatibility of evolved candidates remain runtime evidence to collect. Approximate native token estimation still does not guarantee Qwen token fit. Any unsupported source, wire, context, tool or infrastructure failure remains invalid and must retain its concrete evidence. The already-running strict ACE/follower need no change for this fix.
