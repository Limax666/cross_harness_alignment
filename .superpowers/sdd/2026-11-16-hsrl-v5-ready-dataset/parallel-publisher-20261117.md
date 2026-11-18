# Standalone parallel publisher, 2026-11-17

Implemented only `experiments/cross_harness_sft/scripts/publish_hsrl_v6_parallel.py`
and its filesystem tests. Existing release, Dataset, runtime, auditor, trainer,
candidate, manifests, frozen QA, and FlashRSI baseline were not edited. The
publisher is operational code outside the release runtime and is not added to
the release's manifest or runtime identity. Mechanism and protected runtime
SHA-256 values are retained in `parallel-publisher-evidence-20261117/identity.json`.

## Preserved acceptance gates

The source must be private and immutable, with no actor able to write it. The
publisher calls the original `require_admission` and original
`Dataset.for_verification`; this preserves core identity, frozen runtime module
coverage/bytes, source pins, loaded selection, 14,000 unique rows, exact role
quotas, and admission checks. It calls the original `Dataset.verify_file` for
every source-declared file and every staged-declared file. No cache, hardlink,
symlink, skipped file, or simulated readiness is used.

The staged audit preserves the original 875-block K2 checks and complete unique
index coverage. It calls the original `Dataset.public` for every key, and checks
that every packet has exactly `messages` and `tools`. Only these read-only file
and public-render passes run concurrently; sampler traversal stays sequential.
Context/runtime certificate copying, manifest/READY assembly, original
`validate_certificates`, the independent normal frozen `Dataset` subprocess,
and original atomic `install_no_replace` retain the original publication order.

Mandatory closure additionally requires the candidate's actual regular-file
inventory to equal its declared files plus `manifest.json`; final staging must
equal declared files plus `manifest.json` and `READY.json`. Traversal rejects
links and special files before the original loader can open an asset. Existing
outputs, including empty directories and dangling symlinks, are refused. The
atomic installation also refuses a destination created during publication.

## Concurrency and isolation

One bounded executor, at most eight workers, consumes work through a shared
iterator. The copy partitions source directories at depth three and recursively
copies each subtree. Every source open is anchored to directory descriptors with
`O_NOFOLLOW`; regular files are copied into independent new files. Copy metadata
uses `shutil.copystat` on descriptors, including mode, timestamps and supported
extended attributes, matching copy2 semantics. Before/after descriptor identity,
size, mode, mtime and ctime detect changes during traversal/copy. Source atime is
excluded because reads may update it, as with ordinary copying.

All copied bytes and intermediate ready metadata stay beneath an unguessable
0700 parent directory until the unchanged final frozen-loader check passes.
Exceptions wait for executor shutdown before propagating. Failures retain the
private stage and leave the source intact. No source environment, model, teacher,
network, API, GPU, or container is invoked by this helper or its verification.

## Retained verification

The existing `.agents/skills/verify-hsrl-dataset/SKILL.md` was used, scoped to
filesystem fixtures and pure original gate tests under the task's constraints.
The exact final command, environment, exit code, stdout and stderr are in
`parallel-publisher-evidence-20261117/filesystem-and-original-gates.json`.
Result: **25 passed**, exit **0**. This includes 19 new filesystem/CLI tests and
six existing pure original gate tests. The initial missing-implementation test
was observed failing before implementation (`standalone publisher missing`).

The new tests cover regular bytes and independent inodes, permissions/mtime,
extended attributes, empty directories, every-input accounting at exactly eight
concurrent workers, all-file original Dataset hashing with each fixture file
individually corrupted, extra/missing inventory, deep and root links, FIFOs,
unsafe manifest types, real source mutation during copying, worker failure and
join, failed private-stage retention, API/real CLI refusal of existing output,
and original atomic no-replace installation. No READY release is fabricated.

CLI `--help` succeeds (exit 0); `--workers 9` is rejected by argparse (expected
exit 2). Both complete outputs are retained alongside the final test evidence.
The protected-code `git diff --exit-code` returns 0, with output retained in
`protected-diff.json`. Evidence remains outside fixture cleanup paths.

## Invocation after independent review and genuine final audit

```bash
/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -B \
  scripts/publish_hsrl_v6_parallel.py \
  --candidate "$CANDIDATE" --out "$NEW_OUT" \
  --context-audit "$GENUINE_CONTEXT_AUDIT" --runtime-audit "$GENUINE_RUNTIME_AUDIT" \
  --workers 8
```

Run from `experiments/cross_harness_sft`. Only the root operator is authorized to
perform real publication after independent review and the genuine R2 audit pass.

## Limits

No actual 14K candidate was read, copied, or published during implementation.
The end-to-end successful publication and its final frozen subprocess have not
been exercised here; genuine certificate acceptance, real full public rendering,
real quota/index/sampler traversal, full-artifact side effects, and wall-clock
speed remain unverified until the authorized root publication. No speedup factor
or successful release is claimed. The full repository suite was not run because
this task permits filesystem fixtures only; broader suites include real source
evidence and runtime workflows. Tests are empirical scoped evidence, not a
formal guarantee.
