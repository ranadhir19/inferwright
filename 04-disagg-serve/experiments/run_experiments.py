#!/usr/bin/env python3
"""Experiments for 04-disagg-serve: colocated vs disaggregated LLM serving.

Seeded with ``np.random.default_rng(31)``; a single RNG stream is consumed in
a fixed order, so repeated runs produce byte-identical headline numbers.

Experiment 1: workload (short-heavy / mixed / long-heavy) x arrival rate
    (low / med / high) x architecture.  Reports TTFT, TBT, throughput,
    utilization.
Experiment 2a: TTFT vs prompt length at fixed rate -> prompt-length crossover.
Experiment 2b: TTFT vs interconnect bandwidth (long-heavy) -> bandwidth
    crossover where the KV-transfer overhead erases the disaggregation win.

Writes 04-disagg-serve/results/charts.png and results/results.md.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llm_inference_engine.disagg_serve import (
    ColocatedConfig,
    DisaggConfig,
    Request,
    make_workload,
    run_colocated,
    run_disaggregated,
)

SEED = 31
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

RATES = [("low", 4.0), ("med", 10.0), ("high", 16.0)]
WORKLOADS = [("short-heavy", 0.8), ("mixed", 0.5), ("long-heavy", 0.2)]
N_MAIN = 1200
N_SWEEP = 400


def fresh(requests):
    return [Request(r.prompt_len, r.gen_len, r.arrival) for r in requests]


def summarize(res):
    return {
        "ttft_mean": float(np.mean(res.ttft_ms)),
        "ttft_p50": float(np.median(res.ttft_ms)),
        "ttft_p99": float(np.percentile(res.ttft_ms, 99)),
        "tbt_mean": float(np.mean(res.tbt_gaps_ms)),
        "tbt_p99": float(np.percentile(res.tbt_gaps_ms, 99)),
        "thr": float(res.throughput_tok_s),
        "util": float(res.utilization),
    }


def run_both(workload):
    rc = run_colocated(fresh(workload))
    rd = run_disaggregated(fresh(workload))
    return summarize(rc), summarize(rd)


# --------------------------------------------------------------------------
def experiment_1(rng):
    """Workload x rate x architecture grid."""
    rows = []
    for wname, sfrac in WORKLOADS:
        for rname, rate in RATES:
            w = make_workload(rng, rate, N_MAIN, short_frac=sfrac)
            sc, sd = run_both(w)
            rows.append((wname, rname, rate, "colocated", sc))
            rows.append((wname, rname, rate, "disaggregated", sd))
            print(f"exp1 {wname:11s} {rname:4s} rate={rate:4.1f} "
                  f"coloc TTFT {sc['ttft_mean']:7.1f}/{sc['ttft_p99']:7.1f} "
                  f"disagg TTFT {sd['ttft_mean']:7.1f}/{sd['ttft_p99']:7.1f} "
                  f"(mean/p99 ms)", flush=True)
    return rows


def experiment_2a(rng):
    """TTFT vs prompt length at fixed arrival rate; find the crossover."""
    rate = 6.0
    lengths = [64, 128, 256, 512, 1024, 2048, 4096]
    out = []
    for L in lengths:
        inter = rng.exponential(1.0 / rate, size=N_SWEEP)
        arrivals = np.cumsum(inter)
        gens = np.clip(np.rint(rng.normal(256.0, 64.0, N_SWEEP)).astype(int), 1, None)
        w = [Request(L, int(g), float(a)) for g, a in zip(gens, arrivals)]
        sc, sd = run_both(w)
        out.append((L, sc, sd))
        print(f"exp2a L={L:5d}: coloc TTFT {sc['ttft_mean']:7.1f}/{sc['ttft_p99']:7.1f} "
              f"disagg TTFT {sd['ttft_mean']:7.1f}/{sd['ttft_p99']:7.1f} "
              f"(mean/p99 ms)", flush=True)
    # Crossover: first L where disagg mean TTFT < colocated mean TTFT,
    # linearly interpolated between the bracketing points.
    cross = None
    for (l0, sc0, sd0), (l1, sc1, sd1) in zip(out[:-1], out[1:]):
        d0 = sd0["ttft_mean"] - sc0["ttft_mean"]
        d1 = sd1["ttft_mean"] - sc1["ttft_mean"]
        if d0 > 0 and d1 < 0:
            cross = l0 + (0.0 - d0) / (d1 - d0) * (l1 - l0)
            bracket = (l0, l1)
            break
    return out, cross, bracket if cross else (None, None)


def experiment_2b(rng):
    """TTFT vs interconnect bandwidth (long-heavy workload)."""
    rate = 10.0
    bws = [5, 10, 20, 50, 100, 200, 400]
    out = []
    coloc_ref = None
    for bw in bws:
        w = make_workload(rng, rate, N_SWEEP, short_frac=0.2)
        sc = summarize(run_colocated(fresh(w)))
        sd = summarize(run_disaggregated(fresh(w), DisaggConfig(interconnect_gbps=float(bw))))
        if coloc_ref is None:
            coloc_ref = sc
        out.append((bw, sc, sd))
        print(f"exp2b bw={bw:4d} GB/s: coloc TTFT {sc['ttft_mean']:7.1f}/{sc['ttft_p99']:7.1f} "
              f"disagg TTFT {sd['ttft_mean']:7.1f}/{sd['ttft_p99']:7.1f} "
              f"(mean/p99 ms)", flush=True)
    # Bandwidth below which the transfer overhead erases the mean-TTFT win:
    # interpolate between the bracketing bandwidths.
    cross = None
    for (b0, sc0, sd0), (b1, sc1, sd1) in zip(out[:-1], out[1:]):
        d0 = sd0["ttft_mean"] - sc0["ttft_mean"]
        d1 = sd1["ttft_mean"] - sc1["ttft_mean"]
        if d0 > 0 and d1 < 0:
            cross = b0 + (0.0 - d0) / (d1 - d0) * (b1 - b0)
            bracket = (b0, b1)
            break
    return out, cross, bracket if cross else (None, None)


# --------------------------------------------------------------------------
def write_charts(rows, sweep_a):
    os.makedirs(RESULTS, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) TTFT vs prompt length
    ax = axes[0]
    Ls = [L for L, _, _ in sweep_a]
    ax.plot(Ls, [s["ttft_mean"] for _, s, _ in sweep_a], "o-", label="colocated")
    ax.plot(Ls, [s["ttft_mean"] for _, _, s in sweep_a], "s-", label="disaggregated")
    ax.set_xscale("log")
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("mean TTFT (ms)")
    ax.set_title("(a) TTFT vs prompt length (rate 6/s)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # (b) throughput vs arrival rate (mixed workload)
    ax = axes[1]
    mixed = [(rate, a, s) for (w, _, rate, a, s) in rows if w == "mixed"]
    rates = sorted(set(r for r, _, _ in mixed))
    ax.plot(rates, [s["thr"] for r, a, s in mixed if a == "colocated"],
            "o-", label="colocated")
    ax.plot(rates, [s["thr"] for r, a, s in mixed if a == "disaggregated"],
            "s-", label="disaggregated")
    ax.set_xlabel("arrival rate (req/s)")
    ax.set_ylabel("throughput (gen tokens/s)")
    ax.set_title("(b) Throughput vs arrival rate (mixed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(RESULTS, "charts.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_results_md(rows, sweep_a, cross_a, bracket_a, sweep_b, cross_b, bracket_b):
    cc, dc = ColocatedConfig(), DisaggConfig()
    lines = []
    lines.append("# 04-disagg-serve: results")
    lines.append("")
    lines.append(f"Seed: {SEED} (np.random.default_rng). All runs deterministic.")
    lines.append("")
    lines.append("## Model parameters")
    lines.append("")
    lines.append(f"- Colocated: {cc.workers} workers, per-iteration "
                 f"t = {cc.t0_ms} ms + {cc.k_ms_per_token} ms/token, "
                 f"KV cap {cc.kv_cap_tokens} tokens/worker.")
    lines.append(f"- Disaggregated: {dc.prefill_workers} prefill worker(s) at "
                 f"{dc.prefill_tok_per_s:,.0f} tok/s + {dc.decode_workers} decode workers "
                 f"(t = {dc.decode_t0_ms} ms + {dc.decode_k_ms_per_token} ms/token, "
                 f"KV cap {dc.decode_kv_cap_tokens} tokens/worker).")
    lines.append(f"- KV cache: 2 * {dc.n_layers} layers * {dc.d_model} d_model * "
                 f"prompt_len * {dc.bytes_per_elem} bytes (fp16). "
                 f"Interconnect {dc.interconnect_gbps:.0f} GB/s + "
                 f"{dc.transfer_latency_ms:.1f} ms fixed transfer latency.")
    lines.append(f"- Workload: Poisson arrivals; prompt mix short=128 / long=2048 "
                 f"tokens; gen_len ~ Normal(256, 64) clipped at >= 1.")
    lines.append("")
    lines.append("## Experiment 1: workload x arrival rate")
    lines.append("")
    lines.append("| workload | rate | arch | TTFT mean (ms) | TTFT p99 (ms) | "
                 "TBT mean (ms) | TBT p99 (ms) | throughput (tok/s) | util |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for wname, rname, rate, arch, s in rows:
        lines.append(f"| {wname} | {rname} ({rate:.0f}/s) | {arch} | "
                     f"{s['ttft_mean']:.1f} | {s['ttft_p99']:.1f} | "
                     f"{s['tbt_mean']:.2f} | {s['tbt_p99']:.2f} | "
                     f"{s['thr']:.1f} | {s['util']:.2f} |")
    lines.append("")
    lines.append("## Experiment 2a: TTFT vs prompt length (rate 6/s)")
    lines.append("")
    lines.append("| prompt_len | colocated TTFT mean/p99 (ms) | "
                 "disaggregated TTFT mean/p99 (ms) |")
    lines.append("|---|---|---|")
    for L, sc, sd in sweep_a:
        lines.append(f"| {L} | {sc['ttft_mean']:.1f} / {sc['ttft_p99']:.1f} | "
                     f"{sd['ttft_mean']:.1f} / {sd['ttft_p99']:.1f} |")
    lines.append("")
    lines.append(f"Crossover (mean TTFT): disaggregation starts winning at "
                 f"prompt_len ~{cross_a:.0f} tokens "
                 f"(bracketed between {bracket_a[0]} and {bracket_a[1]}).")
    lines.append("")
    lines.append("## Experiment 2b: TTFT vs interconnect bandwidth (long-heavy, 10/s)")
    lines.append("")
    lines.append("| bandwidth (GB/s) | colocated TTFT mean/p99 (ms) | "
                 "disaggregated TTFT mean/p99 (ms) |")
    lines.append("|---|---|---|")
    for bw, sc, sd in sweep_b:
        lines.append(f"| {bw} | {sc['ttft_mean']:.1f} / {sc['ttft_p99']:.1f} | "
                     f"{sd['ttft_mean']:.1f} / {sd['ttft_p99']:.1f} |")
    lines.append("")
    lines.append(f"The disaggregation win on mean TTFT is erased below "
                 f"~{cross_b:.0f} GB/s (bracketed between {bracket_b[0]} and "
                 f"{bracket_b[1]} GB/s); at 5 GB/s the KV transfer alone adds "
                 f"~215 ms per long prompt.")
    lines.append("")
    lines.append("## Honesty notes")
    lines.append("")
    lines.append("- This is a scheduling/interference model, not real GPU kernels: "
                 "per-iteration time is the linear model t = T0 + K * batch_tokens.")
    lines.append("- Prefill is not chunked; a full prompt is processed in one iteration.")
    lines.append("- Both architectures get the same total worker count (4 colocated "
                 "vs 1 prefill + 3 decode); the split is a modeling choice.")
    lines.append("- Decode-pool routing is join-shortest-queue by assigned KV tokens.")
    path = os.path.join(RESULTS, "results.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def headlines(rows, cross_a, bracket_a, cross_b, bracket_b):
    def get(w, r, a):
        for (w2, r2, _, a2, s) in rows:
            if w2 == w and r2 == r and a2 == a:
                return s
        raise KeyError
    lh_c = get("long-heavy", "high", "colocated")
    lh_d = get("long-heavy", "high", "disaggregated")
    mx_c = get("mixed", "high", "colocated")
    mx_d = get("mixed", "high", "disaggregated")
    sh_c = get("short-heavy", "low", "colocated")
    sh_d = get("short-heavy", "low", "disaggregated")
    print("HEADLINES")
    print(f"long-heavy/high: disagg TTFT {lh_d['ttft_mean']:.1f} ms vs "
          f"colocated {lh_c['ttft_mean']:.1f} ms "
          f"(p99 {lh_d['ttft_p99']:.1f} vs {lh_c['ttft_p99']:.1f})")
    print(f"mixed/high: disagg TBT p99 {mx_d['tbt_p99']:.2f} ms vs "
          f"colocated {mx_c['tbt_p99']:.2f} ms; "
          f"throughput {mx_d['thr']:.1f} vs {mx_c['thr']:.1f} tok/s")
    print(f"short-heavy/low: disagg TTFT {sh_d['ttft_mean']:.1f} ms vs "
          f"colocated {sh_c['ttft_mean']:.1f} ms")
    print(f"crossover: disaggregation starts winning on TTFT at prompt_len "
          f"~{cross_a:.0f} (bracket {bracket_a[0]}-{bracket_a[1]})")
    print(f"transfer overhead erases the TTFT win below ~{cross_b:.0f} GB/s "
          f"(bracket {bracket_b[0]}-{bracket_b[1]})")


def main():
    rng = np.random.default_rng(SEED)
    rows = experiment_1(rng)
    sweep_a, cross_a, bracket_a = experiment_2a(rng)
    sweep_b, cross_b, bracket_b = experiment_2b(rng)
    chart_path = write_charts(rows, sweep_a)
    md_path = write_results_md(rows, sweep_a, cross_a, bracket_a,
                               sweep_b, cross_b, bracket_b)
    print("wrote", chart_path)
    print("wrote", md_path)
    headlines(rows, cross_a, bracket_a, cross_b, bracket_b)


if __name__ == "__main__":
    main()
