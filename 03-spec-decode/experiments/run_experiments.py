"""Experiments for 03-spec-decode: validate the speculative decoding formula,
sweep draft lengths / acceptance probabilities / cost ratios, run an
end-to-end wall-clock scenario, and produce charts + a results report.

Deterministic: all Monte Carlo uses np.random.default_rng(21).
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
PROJ_DIR = os.path.join(REPO_ROOT, "03-spec-decode")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from llm_inference_engine.spec_decode import (  # noqa: E402
    expected_speedup,
    optimal_k,
    simulate,
    simulate_throughput,
)

SEED = 21
TRIALS = 100_000
ALPHAS = [0.3, 0.5, 0.7, 0.85, 0.95]
CS = [0.05, 0.1, 0.2]
KS = list(range(1, 9))
TARGET_MS = 40.0


# ---------------------------------------------------------------- experiment 1
def experiment1_validate_formula():
    """Monte Carlo (100k trials/cell) vs analytical formula across the grid."""
    rng = np.random.default_rng(SEED)
    max_dev = 0.0
    max_speedup_dev = 0.0
    worst = None
    n_cells = 0
    # NOTE: simulate() takes a seed, not a shared rng; to keep the global
    # stream deterministic across the grid we derive per-cell seeds from SEED.
    cell_rng = np.random.default_rng(SEED)
    cell_seeds = cell_rng.integers(0, 2**31 - 1, size=(len(ALPHAS), len(CS), len(KS)))
    for ai, alpha in enumerate(ALPHAS):
        for ci, c in enumerate(CS):
            for ki, k in enumerate(KS):
                res = simulate(k, alpha, c, trials=TRIALS,
                               seed=int(cell_seeds[ai, ci, ki]))
                dev = abs(res["mean_accepted_tokens"] - res["analytical_accepted"])
                if dev > max_dev:
                    max_dev, worst = dev, (k, alpha, c, res)
                sdev = abs(res["empirical_speedup"] - res["analytical_speedup"])
                max_speedup_dev = max(max_speedup_dev, sdev)
                n_cells += 1
    return {"max_abs_dev": max_dev, "max_speedup_dev": max_speedup_dev,
            "worst": worst, "n_cells": n_cells, "trials": TRIALS}


# ---------------------------------------------------------------- experiment 2
def experiment2_sweep():
    """Speedup table over k x alpha x c; optimal k per (alpha, c)."""
    speedup = np.zeros((len(CS), len(ALPHAS), len(KS)))
    for ci, c in enumerate(CS):
        for ai, alpha in enumerate(ALPHAS):
            for ki, k in enumerate(KS):
                speedup[ci, ai, ki] = expected_speedup(k, alpha, c)
    opt = {}
    for ci, c in enumerate(CS):
        for ai, alpha in enumerate(ALPHAS):
            best_k, best_s = optimal_k(alpha, c, k_max=8)
            opt[(alpha, c)] = (best_k, best_s)
    bi, bj, bk = np.unravel_index(np.argmax(speedup), speedup.shape)
    return {
        "speedup": speedup, "opt": opt,
        "best": (KS[bk], ALPHAS[bj], CS[bi], float(speedup[bi, bj, bk])),
    }


# ---------------------------------------------------------------- experiment 3
def experiment3_end_to_end():
    """Wall-clock scenario: target forward 40 ms, c = 0.1.

    Main case: alpha = 0.7 at its optimal draft length. Failure case:
    alpha = 0.3, k = 8 (large draft, poor acceptance -> slowdown).
    """
    k_opt, _ = optimal_k(0.7, 0.1, k_max=8)
    main = simulate_throughput(k_opt, 0.7, 0.1, TARGET_MS, trials=TRIALS, seed=SEED)
    fail = simulate_throughput(8, 0.3, 0.1, TARGET_MS, trials=TRIALS, seed=SEED + 1)
    return {"k_opt": k_opt, "main": main, "fail": fail}


# ---------------------------------------------------------------- experiment 4
def experiment4_charts(sweep):
    """(a) speedup heatmap vs (k, alpha) at c=0.1; (b) optimal-k vs alpha."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) heatmap at c = 0.1 (rows alpha, cols k)
    ci = CS.index(0.1)
    data = sweep["speedup"][ci]  # (alpha, k)
    ax = axes[0]
    im = ax.imshow(data, aspect="auto", origin="lower", cmap="RdYlGn",
                   vmin=0.6, vmax=2.2)
    ax.set_xticks(range(len(KS)), KS)
    ax.set_yticks(range(len(ALPHAS)), [str(a) for a in ALPHAS])
    ax.set_xlabel("draft length k")
    ax.set_ylabel("acceptance prob alpha")
    ax.set_title("Expected speedup vs vanilla AR (c = 0.1)")
    for ai in range(len(ALPHAS)):
        for ki in range(len(KS)):
            v = data[ai, ki]
            ax.text(ki, ai, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if v > 1.0 else "white")
    fig.colorbar(im, ax=ax, label="speedup")

    # (b) optimal k vs alpha for each c (fine alpha grid, k_max = 12)
    ax = axes[1]
    fine_alphas = np.linspace(0.05, 0.99, 60)
    for c in CS:
        ks = [optimal_k(a, c, k_max=12)[0] for a in fine_alphas]
        ax.plot(fine_alphas, ks, marker="o", ms=3, label=f"c = {c}")
    ax.set_xlabel("acceptance prob alpha")
    ax.set_ylabel("optimal draft length k")
    ax.set_title("Optimal draft length vs acceptance probability")
    ax.set_yticks(range(0, 13))
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Speculative decoding: analytical model (seeded, reproducible)")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "charts.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def write_results_md(e1, e2, e3):
    lines = []
    A = lines.append
    A("# 03-spec-decode — results\n")
    A("Seeded, reproducible (numpy, CPU only). Monte Carlo: "
      f"{e1['trials']:,} trials/cell over {e1['n_cells']} cells "
      f"(k=1..8 x alpha x c), seed 21.\n")

    A("## 1. Formula vs Monte Carlo\n")
    wk, wa, wc, wres = e1["worst"]
    A(f"Max absolute deviation between the analytical expected accepted "
      f"tokens and the Monte Carlo mean: **{e1['max_abs_dev']:.5f}** tokens "
      f"(worst cell: k={wk}, alpha={wa}, c={wc}, "
      f"MC mean={wres['mean_accepted_tokens']:.5f}, "
      f"formula={wres['analytical_accepted']:.5f}). "
      f"Speedup max deviation: {e1['max_speedup_dev']:.5f}. "
      f"The geometric acceptance model is validated.\n")

    A("## 2. Speedup sweep (expected speedup vs vanilla AR)\n")
    A("Rows: draft length k. Columns: acceptance probability alpha.\n")
    for ci, c in enumerate(CS):
        A(f"\n### c = {c}\n")
        A("| k | " + " | ".join(f"alpha={a}" for a in ALPHAS) + " |")
        A("|---" + "|---" * len(ALPHAS) + "|")
        for ki, k in enumerate(KS):
            row = " | ".join(f"{e2['speedup'][ci, ai, ki]:.3f}" for ai in range(len(ALPHAS)))
            A(f"| {k} | {row} |")
    A("\n### Optimal k per (alpha, c)\n")
    A("| alpha | " + " | ".join(f"c={c}" for c in CS) + " |")
    A("|---" + "|---" * len(CS) + "|")
    for a in ALPHAS:
        cells = []
        for c in CS:
            bk, bs = e2["opt"][(a, c)]
            cells.append(f"k={bk} ({bs:.2f}x)")
        A(f"| {a} | " + " | ".join(cells) + " |")
    bk, ba, bc, bs = e2["best"]
    A(f"\nBest cell in sweep: **{bs:.3f}x speedup at k={bk}, alpha={ba}, c={bc}**.\n")

    A("## 3. End-to-end wall-clock scenario (target forward 40 ms, c = 0.1)\n")
    m, f = e3["main"], e3["fail"]
    A(f"- alpha=0.7 at optimal k={e3['k_opt']}: speculative "
      f"**{m['speculative_tps']:.1f} tok/s** vs vanilla "
      f"**{m['vanilla_tps']:.1f} tok/s** (speedup {m['speedup']:.3f}x, "
      f"analytical {m['analytical_speedup']:.3f}x).\n")
    A(f"- Failure case alpha=0.3, k=8: speculative "
      f"**{f['speculative_tps']:.1f} tok/s** vs vanilla "
      f"**{f['vanilla_tps']:.1f} tok/s** (speedup {f['speedup']:.3f}x, "
      f"analytical {f['analytical_speedup']:.3f}x) — a slowdown: "
      f"draft compute is wasted because most drafts are rejected.\n")

    A("## 4. Charts\n")
    A("`results/charts.png`: (a) speedup heatmap vs (k, alpha) at c=0.1; "
      "(b) optimal-k vs alpha curves for the three c values.\n")

    path = os.path.join(RESULTS_DIR, "results.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def main():
    e1 = experiment1_validate_formula()
    e2 = experiment2_sweep()
    e3 = experiment3_end_to_end()
    chart_path = experiment4_charts(e2)
    md_path = write_results_md(e1, e2, e3)

    bk, ba, bc, bs = e2["best"]
    m, f = e3["main"], e3["fail"]
    headlines = [
        "=== HEADLINES ===",
        f"formula-vs-MC max abs dev: {e1['max_abs_dev']:.5f} tokens "
        f"({e1['n_cells']} cells x {e1['trials']:,} trials, seed {SEED})",
        f"best sweep speedup: {bs:.3f}x at k={bk}, alpha={ba}, c={bc}",
        f"end-to-end (T=40ms, c=0.1, alpha=0.7, k={e3['k_opt']}): "
        f"speculative {m['speculative_tps']:.1f} tok/s vs vanilla "
        f"{m['vanilla_tps']:.1f} tok/s (speedup {m['speedup']:.3f}x)",
        f"failure case (alpha=0.3, k=8, c=0.1): speedup {f['speedup']:.3f}x "
        f"({f['speculative_tps']:.1f} tok/s vs {f['vanilla_tps']:.1f} tok/s) — slowdown",
        f"charts: {chart_path}",
        f"report: {md_path}",
    ]
    print("\n".join(headlines))


if __name__ == "__main__":
    main()
