# GPU1 queue/scanner v21 independent scoped review — 2026-11-17

**Verdict: one Important blocker in `5855389`; no Critical finding.** Queue registration and the actual Claude runtime interfaces fit the inherited native admission contract, but the new scanner has no current-cohort completion guard alongside its historical completion summary. Root should resolve I1 before relying on this scanner for deployment/completion visibility. Claude v9 native-profile correctness is outside this review and still requires its separate independent clearance.

Scope: read the owned review diff, implementation report, actual v21 queue/scanner/tests, inherited v20 controller and v19/v18/v17 scanner transformations, inherited process-registration chain, and Claude v9 initial/reference/profile interface definitions. Compared the owned queue/scanner/tests/report to `5855389`: no working-tree drift at review time. Earlier v20 findings remain closed except as stated below; this is not a fresh review of the Claude native profile's internal proof rules.

## Critical findings

None.

## Important findings

### I1 — New current-queue visibility has no separate status/completion guard

**Location:** `scripts/flashrsi_matrix_repair_scan_v21.py:70–76`, composed with `scripts/flashrsi_matrix_repair_scan_v17.py:51–60`.

v21 inserts `queued = queued_job_entries(...)`, but changes only the fallback of the inherited status expression. The generated behavior is:

```python
status = next((value for value in ('complete', 'running', 'suspected_stall', 'needs_judge')
               if value in states),
              'queued' if queued else 'needs_repair' if runs else 'pending')
```

Therefore, with a historical accepted run for `pi/opro` and a fresh v21 `pi/opro` task whose retained status is `queued` under an exactly matched live launcher identity, `states == {'complete'}` and `queued` is nonempty, yet the job's reported status is **`complete`**. The same historical result wins when the fresh native run is running or failed. Historical completion should remain visible with its protocol/path; the missing piece is an explicit current-queue/cohort state. Historical non-native Pi/Claude runs deliberately retain legacy inspection, while the fresh task requires new matched native H0/profile evidence. A historical legacy completion therefore cannot establish completion of that fresh task.

The new `queued_launches`, `queues[].tasks` and `runs` still preserve the detailed evidence; the defect is in the top-level job status/counts and their `all_complete` reduction. If all other logical jobs have accepted historical results, that reduction can report `all_complete=true` while the new finite queue still owns pending fresh native work. This contradicts the requested live queued visibility and the prohibition on declaring the new objective complete from old measurements. It does not launch duplicate work or weaken the queue's own source admission.

**Required correction:** distinguish historical accepted coverage from the state of identity-matched current finite-queue work. Provide explicit `current_queue_jobs`/status/counts or per-job `current_status`, preserving fresh owned queued/running/blocked/failure states and retained evidence. Historical status can keep its existing precedence if clearly labelled with its aggregate mode and protocol/path evidence. Conservatively require `all_complete=false` while any identity-confirmed current-queue task is not complete, including H0 and failed/blocked tasks. Add a regression combining an old `complete` run with a live identity-matched queued fresh task, plus a fresh failure case; the existing tests cover queued visibility only when there are no prior run states. Root confirmed this current-cohort versus best-historical distinction during review. Changes to old scanners or historical acceptance are outside this finding's scope.

## Integration checks without additional blockers

- v21 uses exact-count transformations of the reviewed v20 source in its own globals. Its generated protocol, launcher and private lock are v21; the shared GPU1 lane lock, exact predecessor wait before source/resource freeze, port18764, empty child CUDA visibility, post-endpoint-wait binding checks, sticky cleanup stop, finite judge/billing behavior and cached judge check under the worker lease remain inherited. Existing modules and active ACE code are not changed by this commit.
- The native registry retains Pi and adds the actual Claude v9 host/initial/reference, helper contract/profile, source-pins callback, reference/case validators and Snapshot-source mapper. The explicit Claude resource path/hash/type/ID agree with the retained resource evidence. Shared freezing retains and rechecks both original resources/copies and both profiles' source pins around stages. Only resource bytes are taken from old runs.
- Harness order is OpenCode, OpenClaw, Codex, Pi, Claude Code. OpenCode keeps its reviewed existing H0 and two pending algorithms; the other four receive fresh H0 before all six original algorithm names. No task is dropped. Both original-resource commands use the actual `--base`/`--snapshot` interface and each native algorithm gets its fresh `HR_INITIAL_REFERENCE`. The queue's coverage flags continue to say full-five-harness completion is false.
- Claude's actual initial manifest uses the shared expected16-turn/600-second-native/900-second-sandbox/4096-reply/temperature0/port18764 contract and original model/runtime/manifest/Snapshot bindings. Its ledger uses `returncode`/`measurement_valid`; its completed receipt supplies the exact fields required by v20's native admission. Shared admission invokes the actual Claude `validate_reference`, which traverses all128 source/owner/private/native/host receipts and rechecks source pins under scoped/restored `HR_OUTPUT_ROOT`. Reference and case callback signatures fit the queue's calls. These are interface checks, not acceptance of those callbacks' internal native proof logic.
- The actual Claude reference transform emits `target_execution_protocol`, `target_contract`, `initial_reference_dir` and initial ID. v21 requires that exact `target_contract` in addition to the inherited selected identity/lineage/rules, strict worker/attempt/scoring and actual per-case native-source validation. The known judge artifact directory exception remains unchanged for both natives.
- Scanner roots cover v17 through v21 and expand exactly one harness-directory level. Both actual native reference names use the identity-derived `run_scope`; that helper requires a recognized harness/algorithm, port18764 and output root. New native host/profile markers and native runs under v20/v21 roots select strict inspection; historical others use legacy inspection. Launcher ownership compares the entire current identity with retained PID/start-ticks/command and checks launcher name/output scope, so a reused PID does not claim pending tasks. H0 and all retained failure/task data remain in `queues[].tasks`.
- The scanner invokes no writer, repair/tick loop or subprocess. Its inherited process-read helpers execute only inside explicit scan calls. Imported v21 controller/main guards and trusted profile/initial/reference definitions dispatch no queue/native work at import.

## Prior minor closed

The v20 reporting-only observation is addressed narrowly in v21's composed `failure_feedback`: native rows are successful only when `returncode == 0`, `measurement_valid is True` and the case result is `completed`; legacy `exit_code == 0` behavior remains. Invalid rows/evidence remain, and acceptance/retry rules are untouched. The regression includes bounded valid native, invalid native and legacy valid rows.

## Evidence and limits

Read the retained **78 passed in3.89s** log (28 v21 plus50 v20/v17 checks), implementation-reported compile/unchanged-v20 checks, and both-resource byte/ID roundtrip log. Those are implementer evidence; this review did not rerun tests or execute/import controller/native/source code. The test suite's strict reference walks stub per-case native evidence, and scanner process identities are fixtures. I1 is established directly by the generated status expression; no live process or workload observation is needed to demonstrate it.

This review performed static file/Git reads and wrote only this report. No process inspection/signaling, queue/native/source/model launch, API/network call, GPU or Docker operation was performed. Genuine queue supersession/identity/lease behavior, both native H0/full128 and six RSI/selected128 paths, native continuation/compaction/mock/provider/namespace closure and actual judge outcomes remain unverified. Separate Claude profile review and I1 resolution remain deployment gates.
