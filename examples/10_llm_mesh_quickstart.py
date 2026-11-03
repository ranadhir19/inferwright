"""Quickstart: LLM mesh routing, orchestration, and failover.

Routes a few raw queries through the semantic router, runs the
fraud-claim mesh pipeline, and shows cost/quality vs a single
frontier model.
"""
import numpy as np

from llm_inference_engine.llm_mesh import (
    ModelRegistry,
    Orchestrator,
    Trace,
    fraud_claim_pipeline,
    route_query,
    train_router,
)

rng = np.random.default_rng(7)
registry = ModelRegistry()
router = train_router()

queries = [
    "prove that the square root of two is irrational",
    "write a python decorator that retries on timeout",
    "what does this mri scan show",
    "help me write a thank-you note for a gift",
]
print("semantic routing (quality-first):")
for q in queries:
    model, task, conf = route_query(registry, router, q, 400, 200)
    print(f"  [{task:9s} conf={conf:.2f}] -> {model:11s} :: {q[:52]}")

orch = Orchestrator(registry, policy="quality-first")
trace = Trace()
res = fraud_claim_pipeline(orch, rng, "claim-001", trace=trace)
print("\nfraud-claim pipeline (vision -> reasoning -> tool -> chat):")
for r in res.records:
    fo = f" (failover from {r.failover_from})" if r.failover_from else ""
    print(f"  {r.subtask:15s} {r.model:11s} {r.latency_ms:7.0f} ms  "
          f"${r.cost_usd:.4f}  q={r.quality:.2f}{fo}")
print(f"  wall={res.wall_latency_ms:.0f} ms  total=${res.total_cost_usd:.4f}  "
      f"mean quality={res.mean_quality:.3f}")

# outage: take down the vision specialist, watch the mesh re-route
registry.mark_down("vision-qw")
res2 = fraud_claim_pipeline(orch, rng, "claim-002")
print("\nvision-qw DOWN -> mesh re-routes:")
for r in res2.records:
    fo = f" (failover from {r.failover_from})" if r.failover_from else ""
    print(f"  {r.subtask:15s} {r.model:11s}{fo}")
print(f"  success: {not res2.failed}, failovers={res2.n_failovers}, "
      f"mean quality={res2.mean_quality:.3f}")

print("\nMesh: specialized routing + tool agents + failover, all traced.")
