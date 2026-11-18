# Safe RSI GitHub handoff publication plan

> Implemented inline in isolated publication worktree. Use existing project verification and an independent read-only review before publishing; source training files remain frozen.

**Goal:** Publish all relevant Harness Scaling safe RSI scripts/docs/configs and a truthful new-server Codex handoff to GitHub main.
**Architecture:** Merge existing main and verified HSRL worktree commits, collect relevant uncommitted work, retain external asset pins and third-party patches, then verify packaged CLI/READY/imports and push fast-forward. No model/data/credential upload or second training actor.
**Spec:** User GitHub+AGENT handoff request; docs/new_server_handoff_20261118.md.

## Acceptance and execution

- [x] Isolate publication; merge source branch and main without editing live trainer.
- [x] Collect HSRL/FlashRSI docs/scripts/tests/configs plus missing hsrl_v3 imports, verification workflow and third-party patch/pins; exclude ignored runtime assets and unrelated changes.
- [x] Write current 14K/50:50/R1 specification, observed environment and source-server state; distinguish GPU1-bound v9 from future GPU0/foreign-host adaptation; preserve no-inflight/cumulative-budget/disabled-cron constraints.
- [x] Check 46 frozen runtime bytes, CLI help,14 packaged regressions,662 Python ASTs and local import closure, actual pinned HarnessRisk patch reconstruction; independent review reports no Important/Critical.
- [x] Complete real READY Dataset/public API and K2 sampling contract verification from publication code, retain logs/JUnit; scan staged contents/history for secrets and private runtime assets.
- [x] Commit and push main without force; confirm exact remote SHA. Keep source actor/controller identities and code hashes unchanged; update local handoff pointers only.

## Review focus

Credentials in docs/history; missed local imports; claiming untested host portability; treating live files as atomic resume; losing source/profile/budget lineage; restoring user-deleted timers; confusing observed counters with independently audited updates. No broad source-QA replay or simulation replaces actual affected API evidence.

Initial publication independently confirmed on GitHub main as `5a1cab60ec38933f24711b4fea605a18bd367de1`; this receipt/plan closeout is a documentation-only follow-up. No source training files or timers changed.
