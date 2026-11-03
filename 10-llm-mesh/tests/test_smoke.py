"""Smoke tests for llm_mesh: registry, router, orchestration, bandit."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from llm_inference_engine.llm_mesh import (  # noqa: E402
    TASK_TYPES,
    BanditRouter,
    DynamicSelector,
    ModelDown,
    ModelRegistry,
    Orchestrator,
    Subtask,
    Trace,
    chart_report_pipeline,
    default_models,
    fraud_claim_pipeline,
    route,
    test_texts_labels,
    train_router,
)


def test_registry_covers_all_tasks():
    reg = ModelRegistry()
    assert len(reg.models) == 5
    for t in TASK_TYPES:
        best = max(reg.models, key=lambda n: reg.quality_of(n, t))
        assert reg.quality_of(best, t) >= 0.90, (t, best)


def test_call_determinism():
    reg = ModelRegistry()
    r1 = reg.call(np.random.default_rng(7), "coder-cl", "coding", 400, 200)
    r2 = reg.call(np.random.default_rng(7), "coder-cl", "coding", 400, 200)
    assert r1.latency_ms == r2.latency_ms
    assert r1.quality == r2.quality
    assert r1.cost_usd == r2.cost_usd > 0


def test_router_fits_and_classifies():
    router = train_router()
    tx, lb = test_texts_labels()
    assert router.accuracy(tx, lb) >= 0.9
    for text, label in [("solve this integral step by step", "reasoning"),
                        ("write a sql migration", "coding"),
                        ("describe this satellite photo", "vision"),
                        ("tell me a bedtime story", "chat")]:
        pred, conf = router.predict(text)
        assert pred == label, (text, pred)
        assert 0.0 <= conf <= 1.0


def test_routing_policies_sane():
    reg = ModelRegistry()
    assert route(reg, "reasoning", 400, 200, "quality-first") == "reasoner-r1"
    assert route(reg, "coding", 400, 200, "quality-first") == "coder-cl"
    assert route(reg, "vision", 400, 200, "quality-first") == "vision-qw"
    # cost-first must pick the cheapest qualifying model
    cheapest = route(reg, "chat", 400, 200, "cost-first")
    costs = {n: reg.expected_cost_usd(n, 400, 200)
             for n in reg.available()
             if reg.quality_of(n, "chat") >= 0.60}
    assert cheapest == min(costs, key=costs.get)


def test_orchestrator_accounting():
    reg = ModelRegistry()
    orch = Orchestrator(reg)
    rng = np.random.default_rng(7)
    subs = [Subtask("a", "vision", 800, 250), Subtask("b", "chat", 1500, 300)]
    seq = orch.run(rng, "r1", subs, mode="sequential")
    par = orch.run(rng, "r2", subs, mode="parallel")
    # sequential wall = sum; parallel wall = max + overhead (per-run records)
    lats = [r.latency_ms for r in seq.records]
    assert seq.wall_latency_ms == sum(lats)
    plats = [r.latency_ms for r in par.records]
    assert par.wall_latency_ms == max(plats) + 50.0
    assert par.wall_latency_ms < seq.wall_latency_ms
    # cost and quality aggregate honestly
    assert seq.total_cost_usd == sum(r.cost_usd for r in seq.records)
    assert abs(seq.mean_quality -
               sum(r.quality for r in seq.records) / 2) < 1e-12


def test_failover_reroutes():
    reg = ModelRegistry()
    reg.mark_down("vision-qw")
    orch = Orchestrator(reg)
    rng = np.random.default_rng(7)
    res = orch.run(rng, "r1", [Subtask("v", "vision", 800, 250)],
                   failover=True)
    assert not res.failed
    assert res.records[0].model != "vision-qw"
    # without failover machinery the raw call raises
    try:
        reg.call(rng, "vision-qw", "vision", 800, 250)
        raise AssertionError("expected ModelDown")
    except ModelDown:
        pass


def test_tool_stage():
    reg = ModelRegistry()
    orch = Orchestrator(reg)
    res = orch.run(np.random.default_rng(7), "r1",
                   [Subtask("db", "tool", 100, 50, tool="fraud-db")])
    rec = res.records[0]
    assert rec.model == "tool:fraud-db"
    assert rec.latency_ms == 30.0
    assert rec.cost_usd == 0.0001


def test_workflows_end_to_end():
    reg = ModelRegistry()
    orch = Orchestrator(reg)
    rng = np.random.default_rng(7)
    trace = Trace()
    f = fraud_claim_pipeline(orch, rng, "c1", trace=trace)
    assert len(f.records) == 4  # vision, reasoning, tool, chat
    assert sum(1 for r in f.records if r.task_type == "tool") == 1
    c = chart_report_pipeline(orch, rng, "c2", trace=trace)
    assert len(c.records) == 3
    assert len(trace) == 7
    s = trace.summary()
    assert s["n_failed"] == 0
    assert s["n_ok"] == 7
    assert s["total_cost_usd"] > 0
    assert 0.0 < s["mean_quality"] <= 1.0


def test_bandit_learns_oracle():
    reg = ModelRegistry()
    bandit = BanditRouter(reg, c=1.0)
    rng = np.random.default_rng(7)
    for i in range(600):
        task = TASK_TYPES[i % 4]
        model = bandit.select(task)
        res = reg.call(rng, model, task, 400, 200)
        bandit.update(task, model, res.quality)
    # after learning, the oracle is selected more often than chance (1/5)
    for task in TASK_TYPES:
        share = bandit.selection_share(task).get(bandit.oracle_model(task), 0)
        assert share > 0.25, (task, share)


def test_dynamic_selector_runs():
    reg = ModelRegistry()
    sel = DynamicSelector(reg)
    rng = np.random.default_rng(7)
    for i in range(40):
        task = TASK_TYPES[i % 4]
        model = sel.select(task, 400, 200)
        res = reg.call(rng, model, task, 400, 200)
        sel.observe(task, model, res.quality, res.cost_usd, res.latency_ms)
    # with no failures it always returns an available model
    assert sel.select("reasoning", 400, 200) in reg.available()


def test_trace_summary_finite():
    reg = ModelRegistry()
    orch = Orchestrator(reg)
    trace = Trace()
    orch.run(np.random.default_rng(7), "r1",
             [Subtask("a", "coding", 400, 200)], trace=trace)
    s = trace.summary()
    assert np.isfinite(s["total_cost_usd"])
    assert np.isfinite(s["mean_quality"])
    assert s["per_model"]


if __name__ == "__main__":
    test_registry_covers_all_tasks()
    print("registry ok")
    test_call_determinism()
    print("determinism ok")
    test_router_fits_and_classifies()
    print("router ok")
    test_routing_policies_sane()
    print("policies ok")
    test_orchestrator_accounting()
    print("orchestrator ok")
    test_failover_reroutes()
    print("failover ok")
    test_tool_stage()
    print("tool ok")
    test_workflows_end_to_end()
    print("workflows ok")
    test_bandit_learns_oracle()
    print("bandit ok")
    test_dynamic_selector_runs()
    print("selector ok")
    test_trace_summary_finite()
    print("trace ok")
    print("ALL SMOKE TESTS PASSED")
