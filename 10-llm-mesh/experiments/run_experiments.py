#!/usr/bin/env python3
"""Experiments for 10-llm-mesh: the LLM mesh simulator.

Seeded with ``np.random.default_rng(7)``; re-running this script
reproduces every number exactly.

Experiment 1: mesh (quality-first routing) vs single frontier model on a
    mixed 400-request suite -- mean quality, total cost, p95 latency.
Experiment 2: semantic router accuracy on the 24-query held-out set.
Experiment 3: parallel vs sequential orchestration on 200 multi-part
    requests (vision + chat fan-out).
Experiment 4: self-improving (UCB1 bandit) router over 3000 requests --
    regret learning curve and final oracle-selection share.
Experiment 5: failover -- vision specialist taken down mid-run; static
    single-model routing vs mesh re-routing. Success rate and quality.
Experiment 6: cost-quality Pareto across the four routing policies.

Writes 10-llm-mesh/results/charts.png and 10-llm-mesh/results/results.md,
and prints headline numbers to stdout.
"""

import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from llm_inference_engine.llm_mesh import (  # noqa: E402
    TASK_TYPES,
    BanditRouter,
    ModelDown,
    ModelRegistry,
    Orchestrator,
    Subtask,
    default_models,
    test_texts_labels,
    train_router,
)

SEED = 7
RESULTS = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# workload helpers
# ---------------------------------------------------------------------------

def make_suite(rng, n=400):
    """Mixed single-subtask requests: uniform task types."""
    tasks = [TASK_TYPES[i % 4] for i in range(n)]
    rng.shuffle(tasks)
    return [
        Subtask(f"req-{i}", t,
                in_tokens=int(rng.integers(200, 800)),
                out_tokens=int(rng.integers(100, 400)))
        for i, t in enumerate(tasks)
    ]


def run_suite(orch, rng, suite, mode="sequential", failover=True):
    """Run one subtask per request; return aggregate metrics."""
    lat, cost, qual, ok = [], [], [], 0
    for s in suite:
        try:
            res = orch.run(rng, s.name, [s], mode=mode, failover=failover)
        except ModelDown:
            continue
        ok += 1
        lat.append(res.wall_latency_ms)
        cost.append(res.total_cost_usd)
        qual.append(res.mean_quality)
    lat = np.array(lat)
    return {
        "n_ok": ok,
        "n_total": len(suite),
        "success_rate": ok / len(suite),
        "mean_quality": float(np.mean(qual)) if qual else 0.0,
        "total_cost_usd": float(sum(cost)),
        "p50_latency_ms": float(np.percentile(lat, 50)) if len(lat) else 0.0,
        "p95_latency_ms": float(np.percentile(lat, 95)) if len(lat) else 0.0,
    }


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------

def exp1_mesh_vs_single(rng):
    """Mesh with quality-first routing vs one frontier model for everything."""
    suite = make_suite(rng, 400)
    mesh = Orchestrator(ModelRegistry(), policy="quality-first")
    m_mesh = run_suite(mesh, np.random.default_rng(SEED + 11), suite)
    single = Orchestrator(
        ModelRegistry([m for m in default_models()
                       if m.name == "frontier-1"]),
        policy="quality-first")
    m_single = run_suite(single, np.random.default_rng(SEED + 12), suite)
    return m_mesh, m_single


def exp2_router_accuracy():
    router = train_router()
    tx, lb = test_texts_labels()
    return router.accuracy(tx, lb), len(tx)


def exp3_parallel_vs_sequential(rng):
    """200 multi-part requests: vision(chart) + chat(report) subtasks."""
    reg = ModelRegistry()
    orch = Orchestrator(reg, policy="quality-first")
    parts = []
    for i in range(200):
        parts.append([
            Subtask(f"m{i}-chart", "vision", 800, 250),
            Subtask(f"m{i}-report", "chat", 1500, 300),
        ])
    seq_lat, par_lat = [], []
    r1, r2 = np.random.default_rng(SEED + 21), np.random.default_rng(SEED + 22)
    for p in parts:
        seq_lat.append(orch.run(r1, p[0].name, p, mode="sequential").wall_latency_ms)
    for p in parts:
        par_lat.append(orch.run(r2, p[0].name, p, mode="parallel").wall_latency_ms)
    seq_lat, par_lat = np.array(seq_lat), np.array(par_lat)
    return {
        "seq_mean_ms": float(seq_lat.mean()),
        "par_mean_ms": float(par_lat.mean()),
        "speedup": float(seq_lat.mean() / par_lat.mean()),
        "seq_p95_ms": float(np.percentile(seq_lat, 95)),
        "par_p95_ms": float(np.percentile(par_lat, 95)),
    }


def exp4_bandit_learning(rng, n=3000):
    """UCB1 per task type; reward = observed quality."""
    reg = ModelRegistry()
    bandit = BanditRouter(reg, c=1.0)
    rcall = np.random.default_rng(SEED + 31)
    for i in range(n):
        task = TASK_TYPES[i % 4]
        model = bandit.select(task)
        res = reg.call(rcall, model, task, in_tokens=400, out_tokens=200)
        bandit.update(task, model, res.quality)
    hist = bandit.history
    window = 100
    windows = [hist[i:i + window] for i in range(0, n, window)]
    mean_regret = [float(np.mean([h["regret"] for h in w])) for w in windows]
    shares = {t: bandit.selection_share(t) for t in TASK_TYPES}
    oracle = {t: bandit.oracle_model(t) for t in TASK_TYPES}
    return {
        "mean_regret_windows": mean_regret,
        "first_window_regret": mean_regret[0],
        "last_window_regret": mean_regret[-1],
        "oracle_share": {t: shares[t].get(oracle[t], 0.0) for t in TASK_TYPES},
        "oracle_model": oracle,
    }


def exp5_failover(rng):
    """vision-qw goes down mid-run: static mapping vs mesh re-routing."""
    suite = make_suite(rng, 400)
    # static baseline: fixed best-model-per-task mapping, no health check
    static_map = {"reasoning": "reasoner-r1", "coding": "coder-cl",
                  "vision": "vision-qw", "chat": "frontier-1"}
    reg = ModelRegistry()
    reg.mark_down("vision-qw")
    rcall = np.random.default_rng(SEED + 41)
    ok, qual = 0, []
    for s in suite:
        try:
            res = reg.call(rcall, static_map[s.task_type], s.task_type,
                           s.in_tokens, s.out_tokens)
        except ModelDown:
            continue
        ok += 1
        qual.append(res.quality)
    static = {"success_rate": ok / len(suite),
              "mean_quality": float(np.mean(qual)) if qual else 0.0}
    # mesh: orchestrator re-routes around the outage
    mesh = Orchestrator(reg, policy="quality-first")
    m = run_suite(mesh, np.random.default_rng(SEED + 42), suite,
                  failover=True)
    return static, {"success_rate": m["success_rate"],
                    "mean_quality": m["mean_quality"]}


def exp6_pareto(rng):
    """Four routing policies on the same suite: (cost, quality) points."""
    from llm_inference_engine.llm_mesh import POLICIES
    suite = make_suite(rng, 400)
    out = {}
    for j, policy in enumerate(POLICIES):
        orch = Orchestrator(ModelRegistry(), policy=policy)
        m = run_suite(orch, np.random.default_rng(SEED + 51 + j), suite)
        out[policy] = m
    return out


# ---------------------------------------------------------------------------
# charts + report
# ---------------------------------------------------------------------------

def write_charts(path, e1, e3, e4, e5, e6):
    m_mesh, m_single = e1
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("llm-mesh: measurement results (seed 7)", fontsize=14)

    a = ax[0, 0]
    a.bar(["single\n(frontier-1)", "mesh\n(quality-first)"],
          [m_single["total_cost_usd"], m_mesh["total_cost_usd"]])
    a.set_yscale("log")
    a.set_ylabel("total cost, USD (log)")
    a.set_title("Exp 1: 400-request suite cost")

    a = ax[0, 1]
    a.bar(["single", "mesh"],
          [m_single["mean_quality"], m_mesh["mean_quality"]])
    a.set_ylim(0.8, 1.0)
    a.set_ylabel("mean quality")
    a.set_title("Exp 1: mean quality")

    a = ax[0, 2]
    a.bar(["single", "mesh"],
          [m_single["p95_latency_ms"], m_mesh["p95_latency_ms"]])
    a.set_ylabel("p95 latency, ms")
    a.set_title("Exp 1: p95 latency")

    a = ax[0, 3]
    a.bar(["sequential", "parallel"],
          [e3["seq_mean_ms"], e3["par_mean_ms"]])
    a.set_ylabel("mean wall latency, ms")
    a.set_title("Exp 3: orchestration mode")

    a = ax[1, 0]
    w = np.arange(len(e4["mean_regret_windows"]))
    a.plot(w, e4["mean_regret_windows"])
    a.set_xlabel("100-request window")
    a.set_ylabel("mean regret vs oracle")
    a.set_title("Exp 4: bandit learning curve")

    a = ax[1, 1]
    tasks = list(TASK_TYPES)
    shares = [e4["oracle_share"][t] for t in tasks]
    labels = [f"{t}\n({e4['oracle_model'][t]})" for t in tasks]
    a.bar(labels, shares)
    a.set_ylim(0, 1)
    a.set_ylabel("final selection share")
    a.set_title("Exp 4: oracle model share")

    a = ax[1, 2]
    a.bar(["static\n(no failover)", "mesh\n(failover)"],
          [e5[0]["success_rate"], e5[1]["success_rate"]])
    a.set_ylim(0, 1.05)
    a.set_ylabel("success rate")
    a.set_title("Exp 5: vision-qw outage")

    a = ax[1, 3]
    for policy, m in e6.items():
        a.scatter([m["total_cost_usd"]], [m["mean_quality"]], s=120,
                  label=policy)
    a.set_xscale("log")
    a.set_xlabel("total cost, USD (log)")
    a.set_ylabel("mean quality")
    a.set_title("Exp 6: cost-quality Pareto")
    a.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_report(path, e1, e2, e3, e4, e5, e6):
    m_mesh, m_single = e1
    acc, n_test = e2
    lines = []
    A = lines.append
    A("# llm-mesh: experiment results (seed 7)\n")
    A("All numbers are simulator output, not invented.\n")
    A("## Experiment 1 -- mesh vs single frontier model (400 requests)\n")
    A("| system | mean quality | total cost (USD) | p95 latency (ms) |")
    A("|---|---|---|---|")
    A(f"| single (frontier-1) | {m_single['mean_quality']:.4f} | "
      f"{m_single['total_cost_usd']:.4f} | {m_single['p95_latency_ms']:.1f} |")
    A(f"| mesh (quality-first) | {m_mesh['mean_quality']:.4f} | "
      f"{m_mesh['total_cost_usd']:.4f} | {m_mesh['p95_latency_ms']:.1f} |")
    A(f"| mesh / single | {m_mesh['mean_quality']/m_single['mean_quality']:.3f}x | "
      f"{m_mesh['total_cost_usd']/m_single['total_cost_usd']:.3f}x | "
      f"{m_mesh['p95_latency_ms']/m_single['p95_latency_ms']:.3f}x |\n")
    A("## Experiment 2 -- semantic router accuracy\n")
    A(f"TF-IDF Rocchio router trained on 72 labeled queries, tested on "
      f"{n_test} held-out queries: accuracy **{acc:.4f}**.\n")
    A("## Experiment 3 -- parallel vs sequential orchestration\n")
    A(f"200 two-subtask requests (vision chart + chat report): sequential "
      f"mean wall latency {e3['seq_mean_ms']:.1f} ms, parallel "
      f"{e3['par_mean_ms']:.1f} ms -- **{e3['speedup']:.2f}x speedup** "
      f"(p95 {e3['seq_p95_ms']:.1f} vs {e3['par_p95_ms']:.1f} ms).\n")
    A("## Experiment 4 -- self-improving router (UCB1, 3000 requests)\n")
    A(f"Mean regret vs oracle: first 100-request window "
      f"{e4['first_window_regret']:.4f}, last window "
      f"{e4['last_window_regret']:.4f}. Final oracle-model selection share:")
    for t in TASK_TYPES:
        A(f"- {t}: {e4['oracle_model'][t]} selected "
          f"{e4['oracle_share'][t]:.1%} of the time")
    A("")
    A("## Experiment 5 -- failover under vision-qw outage (400 requests)\n")
    A(f"Static per-task mapping without health checks: success rate "
      f"{e5[0]['success_rate']:.3f}. Mesh with re-routing: success rate "
      f"{e5[1]['success_rate']:.3f}, mean quality {e5[1]['mean_quality']:.4f}.\n")
    A("## Experiment 6 -- routing-policy Pareto (400 requests)\n")
    A("| policy | mean quality | total cost (USD) |")
    A("|---|---|---|")
    for policy, m in e6.items():
        A(f"| {policy} | {m['mean_quality']:.4f} | {m['total_cost_usd']:.4f} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    rng = np.random.default_rng(SEED)
    e1 = exp1_mesh_vs_single(rng)
    e2 = exp2_router_accuracy()
    e3 = exp3_parallel_vs_sequential(rng)
    e4 = exp4_bandit_learning(rng)
    e5 = exp5_failover(rng)
    e6 = exp6_pareto(rng)

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_charts(RESULTS / "charts.png", e1, e3, e4, e5, e6)
    write_report(RESULTS / "results.md", e1, e2, e3, e4, e5, e6)

    m_mesh, m_single = e1
    print(f"exp1 mesh vs single: quality {m_mesh['mean_quality']:.4f} vs "
          f"{m_single['mean_quality']:.4f}; cost ${m_mesh['total_cost_usd']:.4f} "
          f"vs ${m_single['total_cost_usd']:.4f} "
          f"({m_mesh['total_cost_usd']/m_single['total_cost_usd']:.2f}x); "
          f"p95 {m_mesh['p95_latency_ms']:.0f} vs "
          f"{m_single['p95_latency_ms']:.0f} ms")
    print(f"exp2 router accuracy: {e2[0]:.4f} on {e2[1]} held-out queries")
    print(f"exp3 parallel speedup: {e3['speedup']:.2f}x "
          f"({e3['seq_mean_ms']:.0f} -> {e3['par_mean_ms']:.0f} ms)")
    print(f"exp4 bandit regret: {e4['first_window_regret']:.4f} -> "
          f"{e4['last_window_regret']:.4f}; oracle shares: " +
          ", ".join(f"{t}={e4['oracle_share'][t]:.0%}" for t in TASK_TYPES))
    print(f"exp5 failover success: static {e5[0]['success_rate']:.3f} vs "
          f"mesh {e5[1]['success_rate']:.3f}")
    print("exp6 pareto: " + "; ".join(
        f"{p}: q={m['mean_quality']:.4f} ${m['total_cost_usd']:.4f}"
        for p, m in e6.items()))
    print("wrote", RESULTS / "charts.png", "and", RESULTS / "results.md")


if __name__ == "__main__":
    main()
