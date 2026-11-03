#!/usr/bin/env python3
"""Experiments for 08-sched-lab: scheduling policy comparison.

Seeded with ``np.random.default_rng(61)``; a single RNG stream is drawn
sequentially, so re-running this script reproduces every number exactly.

GPU model: t = T0 + K * batch_tokens per iteration
    (T0 = 2.0 ms, K = 0.012 ms/token), KV-cache token cap 16384,
    per-request reservation = prompt_len + gen_len.

Experiment 1: 4 policies x {1, 3, 6} req/s, 1500 requests per cell.
Experiment 2: burst of 10 x 4096-token prompts amid chat traffic.
Experiment 3: preemption cost of priority-preempt vs continuous.

Writes 08-sched-lab/results/charts.png and 08-sched-lab/results/results.md,
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

from llm_inference_engine.sched_lab import (  # noqa: E402
    POLICY_NAMES,
    SimConfig,
    add_burst,
    make_workload,
    run_policy,
    summarize,
)

SEED = 61
RESULTS = Path(__file__).resolve().parent.parent / "results"

CFG = SimConfig()  # T0=2.0ms, K=0.012ms/tok, KV cap 16384, chunk 512


# ---------------------------------------------------------------------------
# workload helpers
# ---------------------------------------------------------------------------

def workload_with_burst(rng, n, rate, long_frac, t_burst, burst_prompt=4096,
                        burst_gen=256, burst_count=10):
    """Base mixed workload plus a burst; also returns an is_burst mask."""
    specs = make_workload(rng, n, rate, long_frac=long_frac)
    merged = add_burst(specs, t_burst, prompt_len=burst_prompt,
                       gen_len=burst_gen, count=burst_count)
    # Burst requests are exactly the appended block; base arrivals are
    # continuous so exact equality with t_burst is a non-event.
    is_burst = (
        (merged["arrival"] == t_burst)
        & (merged["prompt_len"] == burst_prompt)
        & (merged["gen_len"] == burst_gen)
    )
    assert int(is_burst.sum()) == burst_count
    return merged, is_burst


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------

def exp1_policy_comparison(rng):
    """4 policies x low/med/high arrival rates, 1500 requests per cell."""
    rates = [1.0, 3.0, 6.0]
    out = {}
    for rate in rates:
        specs = make_workload(rng, 1500, rate, long_frac=0.15)
        for policy in POLICY_NAMES:
            res = run_policy(policy, specs, CFG)
            out[(rate, policy)] = (res, summarize(res))
    return rates, out


def exp2_burst(rng):
    """Burst of 10 x 4096-token prompts at t=60s amid pure chat traffic.

    Uses long_frac=0 so the burst is the only source of long prefills, and
    the standard KV cap (16384) shared with the other experiments.
    """
    specs, is_burst = workload_with_burst(
        rng, n=400, rate=3.0, long_frac=0.0, t_burst=60.0)
    out = {}
    for policy in POLICY_NAMES:
        res = run_policy(policy, specs, CFG)
        chat = ~is_burst
        m_chat = summarize(res, chat)
        m_chat["max_iter_dt_ms"] = float(res.iter_dt_ms.max())
        out[policy] = {
            "res": res,
            "all": summarize(res),
            "chat": m_chat,
            "burst": summarize(res, is_burst),
        }
    return out


def exp3_preemption_cost(rng):
    """priority-preempt vs continuous under KV contention."""
    specs, _ = workload_with_burst(
        rng, n=600, rate=6.0, long_frac=0.30, t_burst=30.0)
    out = {}
    for policy in ["continuous", "priority-preempt"]:
        res = run_policy(policy, specs, CFG)
        short = res.is_short
        m = summarize(res)
        total_compute = res.total_prompt_tokens + res.total_gen_tokens
        out[policy] = {
            "res": res,
            "all": m,
            "short": summarize(res, short),
            "extra_compute_pct": 100.0 * m["recompute_tokens"] / total_compute,
        }
    return out


# ---------------------------------------------------------------------------
# charts + markdown
# ---------------------------------------------------------------------------

def make_charts(rates, exp1):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax_mean, ax_p99, ax_thr, ax_util = axes.flat
    xs = np.array(rates)
    for policy in POLICY_NAMES:
        mean = [exp1[(r, policy)][1]["ttft_mean_ms"] for r in rates]
        p99 = [exp1[(r, policy)][1]["ttft_p99_ms"] for r in rates]
        thr = [exp1[(r, policy)][1]["throughput_tok_s"] for r in rates]
        util = [exp1[(r, policy)][1]["gpu_util"] for r in rates]
        ax_mean.plot(xs, mean, "o-", label=policy)
        ax_p99.plot(xs, p99, "s-", label=policy)
        ax_thr.plot(xs, thr, "^-", label=policy)
        ax_util.plot(xs, util, "d-", label=policy)
    ax_mean.set_yscale("log")
    ax_p99.set_yscale("log")
    ax_mean.set_title("Mean TTFT vs arrival rate")
    ax_p99.set_title("p99 TTFT vs arrival rate")
    ax_thr.set_title("Throughput vs arrival rate")
    ax_util.set_title("GPU utilization vs arrival rate")
    for ax in axes.flat:
        ax.set_xlabel("arrival rate (req/s)")
        ax.grid(True, alpha=0.3)
    ax_mean.set_ylabel("ms (log)")
    ax_p99.set_ylabel("ms (log)")
    ax_thr.set_ylabel("generated tokens/s")
    ax_util.set_ylabel("time-weighted batch_tokens / KV cap")
    ax_thr.legend(fontsize=8)
    fig.suptitle(
        "LLM serving scheduler comparison (seed 61; analytical GPU model "
        "t = 2.0 + 0.012*batch_tokens ms; KV cap 16384)")
    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "charts.png", dpi=110)
    plt.close(fig)


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


def write_results_md(rates, exp1, exp2, exp3, headlines):
    L = []
    A = L.append
    A("# sched-lab results\n")
    A("Seed: 61 (`np.random.default_rng(61)`, single sequential stream).\n")
    A("GPU model (analytical): `t = 2.0 + 0.012 * batch_tokens` ms per iteration; "
      "KV-cache token cap 16384; per-request reservation `prompt_len + gen_len`.\n")
    A("Workload: Poisson arrivals; 85% chat (prompt 32-512, gen 32-512) + "
      "15% documents (prompt 1024-4096, gen 128-1024).\n")

    A("## Experiment 1 -- policy x load (1500 requests per cell)\n")
    A("| rate (req/s) | policy | TTFT mean (ms) | TTFT p99 (ms) | "
      "TPOT mean (ms) | TPOT p99 (ms) | throughput (tok/s) | GPU util | preemptions |")
    A("|:---:|:---|---:|---:|---:|---:|---:|---:|---:|")
    for rate in rates:
        for policy in POLICY_NAMES:
            m = exp1[(rate, policy)][1]
            A(f"| {rate} | {policy} | {fmt(m['ttft_mean_ms'])} | "
              f"{fmt(m['ttft_p99_ms'])} | {fmt(m['tpot_mean_ms'], 2)} | "
              f"{fmt(m['tpot_p99_ms'], 2)} | {fmt(m['throughput_tok_s'])} | "
              f"{fmt(m['gpu_util'], 3)} | {m['preemptions']} |")
    A("")

    A("## Experiment 2 -- burst of 10 x 4096-token prompts at t=60 s "
      "amid pure chat traffic (3 req/s, no background documents)\n")
    A("| policy | chat TTFT mean (ms) | chat TTFT p99 (ms) | "
      "burst TTFT mean (ms) | max iteration latency (ms) | throughput (tok/s) |")
    A("|:---|---:|---:|---:|---:|---:|")
    for policy in POLICY_NAMES:
        e = exp2[policy]
        A(f"| {policy} | {fmt(e['chat']['ttft_mean_ms'])} | "
          f"{fmt(e['chat']['ttft_p99_ms'])} | "
          f"{fmt(e['burst']['ttft_mean_ms'])} | "
          f"{fmt(e['chat']['max_iter_dt_ms'])} | "
          f"{fmt(e['all']['throughput_tok_s'])} |")
    A("")

    A("## Experiment 3 -- preemption cost: priority-preempt vs continuous "
      "(6 req/s, 30% long, burst at t=30 s)\n")
    A("| policy | short TTFT mean (ms) | short TTFT p99 (ms) | "
      "all TTFT p99 (ms) | throughput (tok/s) | preemptions | "
      "recompute tokens | extra compute (%) |")
    A("|:---|---:|---:|---:|---:|---:|---:|---:|")
    for policy in ["continuous", "priority-preempt"]:
        e = exp3[policy]
        A(f"| {policy} | {fmt(e['short']['ttft_mean_ms'])} | "
          f"{fmt(e['short']['ttft_p99_ms'])} | {fmt(e['all']['ttft_p99_ms'])} | "
          f"{fmt(e['all']['throughput_tok_s'])} | {e['all']['preemptions']} | "
          f"{e['all']['recompute_tokens']} | {fmt(e['extra_compute_pct'], 2)} |")
    A("")

    A("## Headlines\n")
    for h in headlines:
        A(f"- {h}")
    A("")
    A("Note: the iteration-cost model is analytical, calibrated to the shape "
      "of real serving curves (fixed launch overhead + linear token scaling), "
      "not measured on a GPU. Absolute millisecond values are illustrative; "
      "relative policy differences are the finding.\n")
    (RESULTS / "results.md").write_text("\n".join(L))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(SEED)

    rates, exp1 = exp1_policy_comparison(rng)
    exp2 = exp2_burst(rng)
    exp3 = exp3_preemption_cost(rng)

    # Headlines (must be identical across runs).
    m_high = {p: exp1[(6.0, p)][1] for p in POLICY_NAMES}
    h1 = (f"high load (6 req/s): continuous TTFT mean "
          f"{fmt(m_high['continuous']['ttft_mean_ms'])} ms / p99 "
          f"{fmt(m_high['continuous']['ttft_p99_ms'])} ms vs static-fcfs "
          f"{fmt(m_high['static-fcfs']['ttft_mean_ms'])} ms / p99 "
          f"{fmt(m_high['static-fcfs']['ttft_p99_ms'])} ms")

    c_p99 = exp2["continuous"]["chat"]["ttft_p99_ms"]
    ch_p99 = exp2["chunked-prefill"]["chat"]["ttft_p99_ms"]
    c_mit = exp2["continuous"]["chat"]["max_iter_dt_ms"]
    ch_mit = exp2["chunked-prefill"]["chat"]["max_iter_dt_ms"]
    s_p99 = exp2["static-fcfs"]["chat"]["ttft_p99_ms"]
    mit_cut = 100.0 * (c_mit - ch_mit) / c_mit
    static_cut = 100.0 * (s_p99 - ch_p99) / s_p99
    h2 = (f"burst of 10x4096-token prompts amid chat traffic: chunked-prefill "
          f"holds chat TTFT p99 at {fmt(ch_p99)} ms (same as continuous "
          f"{fmt(c_p99)} ms) while bounding worst-case iteration latency to "
          f"{fmt(ch_mit)} ms vs {fmt(c_mit)} ms ({fmt(mit_cut)}% lower); "
          f"static-fcfs collapses to p99 {fmt(s_p99)} ms "
          f"({fmt(static_cut)}% worse than chunked-prefill)")

    d = exp3["priority-preempt"]
    b = exp3["continuous"]
    short_gain = 100.0 * (b["short"]["ttft_p99_ms"] - d["short"]["ttft_p99_ms"]) \
        / b["short"]["ttft_p99_ms"]
    h3 = (f"priority-preempt: {d['all']['preemptions']} preemptions cost "
          f"{d['all']['recompute_tokens']} recompute tokens "
          f"({fmt(d['extra_compute_pct'], 2)}% extra compute) for "
          f"{fmt(short_gain)}% lower short-request TTFT p99 "
          f"({fmt(b['short']['ttft_p99_ms'])} -> {fmt(d['short']['ttft_p99_ms'])} ms)")

    headlines = [h1, h2, h3]

    make_charts(rates, exp1)
    write_results_md(rates, exp1, exp2, exp3, headlines)

    print("HEADLINE exp1/high-load: " + h1)
    print("HEADLINE exp2/burst: " + h2)
    print("HEADLINE exp3/preempt-cost: " + h3)
    print("wrote " + str(RESULTS / "charts.png"))
    print("wrote " + str(RESULTS / "results.md"))


if __name__ == "__main__":
    main()
