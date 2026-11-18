# FlashRSI × HarnessRisk 4B baseline

1. Stop AgentDojo jobs, retain their evidence, freeze all 128 case IDs and source hashes, and inventory upstream algorithms and harnesses; never include API keys in manifests.
2. Connect the pinned 4B service and actual FlashRSI Snapshot to each HarnessRisk isolated workspace and mock service; preserve all owner turns, tool activity, initial/final state and Snapshot identity. Native CLI agents cannot run with the official process backend alone because it is not an OS sandbox.
3. Verify initial and mutated Snapshot execution, environment reset, egress and filesystem containment, and official rule evaluator positive/negative controls. Track individual failures and do not invent metrics.
4. For each harness, evaluate initial Snapshot on all 128; run each applicable FlashRSI algorithm with explicit evolution budget; then evaluate its selected Snapshot on the same 128. Log exposure to training/dev and avoid calling the full set unseen.
5. Report rule scores, optional independent judge scores, per-metric denominators and coverage separately; run the requested Shengsuanyun judge only in a trusted process with credentials in the environment/credential store, never in the model sandbox.
