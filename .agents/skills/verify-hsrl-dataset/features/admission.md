# Inspect actual source admission

The release CLI reads pinned source inventories and retained real per-case proofs,
then reports whether the balanced release is admissible.

## Sub-features

- Source pins and actual frozen execution-byte validation.
- Terminal completion and unprocessed/invalid distinctions.
- 7,000 general and 5,500/1,000/500 safety quotas with paired clean families.

## How to get to it (user POV)

Run `python -m hsrl_v5.release` with all four source/quality paths and `--status`.

## Driving it with the release CLI

Preconditions: use the exact launch variables from the parent skill; existing
source directories must be readable. Do not start another QC job.

```bash
"$HSRL_PY" -m hsrl_v5.release \
  --claw-source "$HSRL_EXP/outputs/rl/hsrl_v5_claw_prepare_shell_contract_fix1_20261116" \
  --claw-quality "$HSRL_MAIN/outputs/rl/hsrl_v6_claw_real_quality_20261116_r2" \
  --agentdog-source "$HSRL_MAIN/outputs/rl/hsrl_v6_agentdog_components_prepare_20261116" \
  --agentdog-quality "$HSRL_MAIN/outputs/rl/hsrl_v6_agentdog_real_quality_20261116_r1" \
  --status > "$HSRL_EVIDENCE/status.json" 2> "$HSRL_EVIDENCE/status.stderr"
printf '%s\n' "$?" > "$HSRL_EVIDENCE/status.exit"
```

Inspect report source completion, identity errors, admitted roles and shortfall;
compare them with the actual QC summaries. Verify no READY or candidate was
created. Preserve all three evidence files after cleanup.

## Gotchas

The CLI may successfully report `not_ready`; exit zero alone is insufficient.
An infrastructure failure is pending, never model utility zero. Read-only status
does not establish a ready 14K package or all-configuration runtime integration.
