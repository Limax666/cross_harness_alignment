# Official benchmark bridge contract

The collection client isolates benchmark-version-specific code behind HTTP workers. Concrete
workers are implemented in `src/cross_harness_sft/backends/agentdojo.py` and `agentharm.py`, hosted
by `benchmark_server.py`. Run one process for each benchmark. They expose:

- `GET /health` → `{ "version": "git-sha" }`
- `GET /cases` → `{ "cases": [Case, ...] }`
- `POST /episodes/start` with `task_id, seed` → `episode_id`
- `POST /episodes/{id}/tool` with `name, arguments` → observation, invalid-call, side-effect and risk flags
- `POST /episodes/{id}/finish` → utility/safety/final-success/risk-success and native scorer details

The worker owns the official environment state and native verifier. It must reset state for every
episode and must score only tool calls that actually executed. The experiment client never accepts
a teacher self-evaluation as a benchmark verdict.

Pin the official repositories before launching workers:

- AgentDojo: https://github.com/ethz-spylab/agentdojo
- Inspect Evals AgentHarm: https://github.com/UKGovernmentBEIS/inspect_evals

Because both upstream APIs and scorers evolve, install immutable refs with
`scripts/setup_benchmarks.sh` and record both printed commits alongside `/health`. Never replace a
worker failure with the portable test adapter.
