# 10 — llm-mesh: LLM mesh simulator

A numpy-only (CPU) simulator of the LLM Mesh pattern: instead of
`User -> App -> One LLM`, a mesh connects many specialized models through
a model registry, a semantic router, an orchestrator (sequential and
parallel dispatch), a self-improving bandit router, dynamic model
selection, observability traces, and automatic failover — a service mesh
for AI models.

## Problem

One frontier model for every request is the most expensive way to serve
mixed workloads: you pay frontier prices for chit-chat, wait on a
reasoning model's latency for a simple summary, and you inherit a single
vendor's outage as your own. In practice, different models are best at
different tasks, but there was no small reproducible harness that
quantifies what a mesh buys you: how much cost routing saves, what
parallel fan-out is worth, whether a router can learn the mapping on its
own, and what happens when a specialist goes down mid-shift.

## Current gap

Mesh discussions are usually architectural diagrams: boxes labeled
"router" and "orchestrator" with no numbers attached. The questions that
decide whether a mesh is worth building — quality vs cost vs latency
tradeoffs per routing policy, the learning curve of an adaptive router,
the blast radius of a model outage with and without failover — were
argued qualitatively. llm-mesh simulates five specialized models with
analytical quality/latency/cost profiles and measures the mesh machinery
against a single-model baseline on identical seeded workloads.

## Solution

`src/llm_inference_engine/llm_mesh/` implements the mesh components:

- **Model registry** (`models.py`) — five simulated specialists
  (`frontier-1`, `reasoner-r1`, `coder-cl`, `vision-qw`, `general-8b`),
  each with per-task quality means, per-token latency, and per-token
  price. Calls are sampled with seeded noise; no real LLM is invoked.
- **Semantic router** (`router.py`) — TF-IDF + class-centroid (Rocchio)
  classifier mapping raw queries to `reasoning` / `coding` / `vision` /
  `chat`, plus four routing policies: quality-first, cost-first,
  latency-first, balanced.
- **Orchestrator** (`orchestrator.py`) — decomposes a request into typed
  subtasks, dispatches sequentially or in parallel (wall-clock modeled as
  max latency + aggregation overhead), aggregates results, and fails over
  to the next-best model on `ModelDown`. Tool agents (e.g. a fraud-DB
  check) are simulated as fixed-latency stages.
- **Self-improving router** (`bandit.py`) — one UCB1 bandit per task type
  learning the best model from scored outcomes, plus a `DynamicSelector`
  that scores models from rolling observed quality/cost/latency means.
- **Observability** (`observability.py`) — append-only `Trace` of every
  invocation: routing decision, model, latency, tokens, cost, quality,
  failover provenance.
- **Workflows** (`workflows.py`) — the two canonical pipelines from the
  concept: fraud-risk insurance claims (vision extract -> reasoning ->
  tool DB check -> summary) and chart+report explanation (parallel
  vision + chat fan-out, reasoning combine).

## Results

All numbers below are simulator output (seed 7), not invented.

**Experiment 1 — mesh vs single frontier model, 400 mixed requests.**
Quality-first mesh routing reaches mean quality **0.9430** versus
**0.9150** for `frontier-1`-only (specialists beat the generalist inside
their niche) at **$1.92 vs $5.00 total cost — 0.38x, a 62% saving**.
Honest cost of that quality win: mesh p95 latency is 6395 ms vs 4956 ms
for the single model, because quality-first routes reasoning work to
`reasoner-r1`, the slowest model per token (18 ms). Routing policy
chooses which axis you optimize; there is no free lunch.

**Experiment 2 — semantic router accuracy.** TF-IDF Rocchio router
trained on 72 labeled queries scores **1.0000 accuracy on 24 held-out
queries** across the four task types.

**Experiment 3 — parallel vs sequential orchestration.** 200 two-subtask
requests (vision chart + chat report): sequential mean wall latency
6078 ms, parallel 3707 ms — **1.64x speedup** from fan-out
(p95 6947 ms vs 4486 ms; aggregation overhead is a fixed 50 ms).

**Experiment 4 — self-improving router (UCB1, 3000 requests).** Mean
regret against the per-task oracle falls from **0.1566** in the first
100-request window to **0.0296** in the last. Final oracle-model
selection shares: reasoning 53%, coding 53%, vision 62%, chat 47% —
converging but still exploring, which is expected UCB1 behavior under
reward noise (quality std 0.03) and small oracle gaps (e.g. 0.96 vs 0.93
on reasoning).

**Experiment 5 — failover.** With `vision-qw` taken down mid-run, a
static per-task model mapping (no health checks) succeeds on only
**75.0%** of 400 requests — every vision request fails. The mesh, which
re-routes around the outage, holds **100.0%** success at mean quality
0.94 by falling back to the next-best available model.

**Experiment 6 — cost-quality Pareto across routing policies**
(400 requests): quality-first (0.9439, $1.92), balanced (0.8473, $0.43),
latency-first (0.7825, $0.33), cost-first (0.7815, $0.33). Cost-first and
latency-first collapse to nearly the same point — the cheapest model per
token is also the fastest here — while balanced buys 6.6 quality points
for an extra $0.10 over cost-first.

## Insights

- **Specialization beats scale on cost, and slightly on quality.**
  Routing each task to its best model cut cost 62% while *raising* mean
  quality 2.8 points versus the frontier-only baseline. The mesh tax is
  paid in tail latency when the best-quality model is also the slowest.
- **Parallelism is the cheapest latency win in the mesh.** 1.64x on a
  two-way fan-out with a 50 ms aggregation overhead — independent
  subtasks should never run sequentially.
- **Adaptive routing converges without prior knowledge.** The UCB1
  router, starting from zero knowledge of the quality profiles, cut
  regret 5x over 3000 requests. Small oracle gaps plus reward noise mean
  it keeps exploring — tune the exploration constant if you want faster
  lock-in.
- **Failover is the mesh's reliability story.** A single-model outage is
  a 25% outage for static routing and a non-event for the mesh. The
  registry's health check is load-bearing infrastructure, not a nicety.
- **Policy is a business decision, not a technical one.** The Pareto
  front is real: quality-first costs 5.8x cost-first. Balanced sits in a
  genuinely useful middle ($0.43, 0.847).

## Honest note on the model

Model profiles are analytical, not measured: quality means, per-token
latencies, and prices are stipulated to resemble the model families they
are named after, and per-call noise is Gaussian. Absolute dollar and
millisecond values are illustrative; the *relative* comparisons between
architectures on identical workloads are the finding. The semantic router
is a 72-document TF-IDF centroid classifier — a stand-in for the
embedding-based semantic routers used in production — and its perfect
held-out score reflects a small, curated corpus. Parallel dispatch models
wall-clock as max latency plus overhead rather than real threads.

## Reproduce

```bash
python3 10-llm-mesh/experiments/run_experiments.py  # ~5 s, seed 7
python3 10-llm-mesh/tests/test_smoke.py
```

Outputs: `10-llm-mesh/results/charts.png`, `10-llm-mesh/results/results.md`.
Run twice; the outputs are byte-identical.
