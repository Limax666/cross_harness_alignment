# FlashRSI v17 observation review — 2026-11-17

**Spec: PASS. Quality: PASS for the bounded observation/entry change.** No Critical or Important findings block the authorized genuine full OpenCode ACE run. This review does not establish that the real host now succeeds or explain its historical startup failures.

## Scope and reviewed identity

Read-only source review of the two final files from MAIN commit `e1cec998`, compared with `flashrsi_harnessrisk_benchmark.py`, `flashrsi_harnessrisk_algorithm_reference_v7.py`, and `flashrsi_harnessrisk_algorithm.py`:

| File | SHA-256 |
|---|---|
| `scripts/flashrsi_harnessrisk_benchmark_v17.py` | `bf407b424cd4124ea50bd9143cffa9aedfcf0d67638eda710bdee8880169ef8b` |
| `scripts/flashrsi_harnessrisk_algorithm_reference_v17.py` | `81b840801339f21c7e2894fd4d032c9bc461f908d19d726e24c32fbcffb3158e` |

Paths are relative to `experiments/cross_harness_sft`. Only this review report was written. No tests, source-program execution, model/API calls, GPU operations, Docker operations, or other file changes were performed by this reviewer.

## Spec assessment

- The benchmark subclasses the original and overrides only `_run_once`. Original loading, case hashes, phase splits, 128-case requirement, retry policy, source grading, and strict evolution rejection remain inherited.
- Host executable/argv, `ROOT` cwd, inherited subprocess environment, configured host script, 2460-second timeout, snapshot identity, official case identity, and completed-episode checks match the original. Successful output/messages and `_records` assignment retain the original behavior. A host failure still produces an error trajectory; the inherited strict grader still rejects it before candidate ranking.
- Stdout/stderr now stream into attempt-local files. Normal failures and timeouts retain their bytes; `finally` writes the execution record before the returned trajectory can reach strict grading. Return code, command, cwd, IDs, PID and timing bind the retained observation to its attempt.
- The reference entry requires exactly one source-read anchor and injects an exactly-once benchmark-import guard. Its only algorithm-body replacement selects the new benchmark. v7's real initial-reference transform and manifest transform remain in place. `ROOT` is imported from the original benchmark and re-exported; execution shares entry globals, preserving `__name__ == '__main__'` and sibling source resolution through the v17 `__file__`.
- The algorithm/configuration table, target and worker settings, budgets, selected full-128 path, initial Snapshot use, and host/sandbox implementation are unchanged. There is no new error-to-zero acceptance path or diagnostic case cap.
- Execution metadata records only `HR_HARNESS` and `HARNESSRISK_MODEL_PORT` from the environment. It does not serialize the token or complete environment. Each attempt directory is created with mode `0700`, and the two logs and execution record receive mode `0600`. Raw host logs remain private evidence and must retain that handling.

## Findings by severity

| Severity | Findings |
|---|---|
| Critical | None in the reviewed change. |
| Important | None in the reviewed change. |
| Minor | No action required for this launch. |

## Verification boundary and next evidence

The implementer's diagnosis report records **3 transport-boundary tests passed in 1.62 seconds**, including strict-grade failure retention, successful identity/metrics, metadata whitelisting, and timeout evidence. These are reported results; this reviewer did not rerun them or execute either entry.

Real host behavior and the cause of the prior 30 unscorable dev episodes remain unverified. Proceed with the already-authorized full real ACE algorithm and inspect its first genuine execution evidence promptly. A retained failure is diagnostic evidence, not a passing episode; repair its actual cause without weakening original source grading, isolation, or budgets. As with ordinary `finally` cleanup, abrupt process termination or storage failure can prevent completion of `host_execution.json`; the implementation does not provide crash-atomic or fsync-backed evidence guarantees.
