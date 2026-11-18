# Verify a private release

A candidate is inspectable but cannot provide training objective adapters. Final
publication requires all source admission and independently observed runtime and
context evidence, and leaves the candidate nontraining.

## Sub-features

- Failed staging records a shortfall and never creates READY.
- Candidate records/assets/public Parquet and all 875 sampler blocks agree.
- Certificates bind files, source, selection and runtime-defining semantics.
- Atomic publication preserves a nonready candidate on failure or interruption.

## How to get to it (user POV)

Use `hsrl_v5.release --stage`, `--audit-candidate`, then `--publish` as documented
in `docs/hsrl_v6_dataset_usage.md`.

## Driving it with the release CLI

Preconditions: both complete real QC runs, sufficient qualified train roles,
reviewed frozen code and new private output paths. Follow the exact four-path
CLI command in admission.md and the stage/publish commands in the usage document.
Do not stage repeatedly to probe incomplete admission. Do not hand-create audits.

After actual staging, run the frozen-code candidate audit, capture its output,
and independently compare JSONL and Parquet identities/role masks. Check a normal
`Dataset(candidate)` refuses. After real runtime/context audits, publish to the
documented final output and open it in a fresh process using its own `code/`.
Record the final marker/manifest SHA and verify the original candidate still
has `status=candidate` with no READY. Preserve failed staging for diagnosis.

## Gotchas

Private staging may contain ready metadata only after validation; it is not the
known candidate path. Do not expose it as a training path. A certificate's fields
are claims requiring retained genuine evidence. No successful publication is
verified until the real final-path consumer passes.
