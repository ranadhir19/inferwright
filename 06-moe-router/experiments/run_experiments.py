#!/usr/bin/env python3
"""06-moe-router experiments: capacity factor, dropping, load balancing, EP.

Reproducibility: every experiment builds its own ``np.random.default_rng``
from a fixed seed, so two runs produce bit-identical headlines, charts and
``results.md``. Run from anywhere::

    python 06-moe-router/experiments/run_experiments.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from llm_inference_engine.moe_router import (
    alltoall_bytes,
    balanced_gate,
    device_load,
    route,
    topk_gate,
)

SEED = 51
BASE = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- config
N_TOKENS = 4096
N_EXPERTS = 8
TOP_KS = (1, 2)
CF_SWEEP = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
# Some experts are naturally popular: token->expert affinity is skewed.
POPULARITY = np.array([0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05])
SKEW_GAIN = 2.0  # bias = SKEW_GAIN * log(popularity), plus N(0,1) token noise
BALANCE_STRENGTH = 0.99  # idealized aux load-balancing loss: shrink skew 99%
BALANCE_NOISE = 0.1    # tie-break noise, dominates the residual skew -> near-uniform


def make_logits(rng: np.random.Generator) -> np.ndarray:
    """Skewed token->expert logits: (n_tokens, n_experts)."""
    bias = SKEW_GAIN * np.log(POPULARITY)
    return bias[None, :] + rng.normal(0.0, 1.0, size=(N_TOKENS, N_EXPERTS))


def gate_skewed(logits: np.ndarray, top_k: int):
    return topk_gate(logits, top_k)


def gate_balanced(
    logits: np.ndarray, top_k: int, rng: np.random.Generator
):
    return balanced_gate(
        logits, top_k, balance_strength=BALANCE_STRENGTH, noise_scale=BALANCE_NOISE, rng=rng
    )


# ------------------------------------------------------- experiment 1
def exp1_routing_quality() -> dict:
    """Sweep capacity factor; report drop rate, imbalance, processed tokens."""
    out: dict = {}
    for top_k in TOP_KS:
        rng = np.random.default_rng(SEED + 100 + top_k)
        logits = make_logits(rng)
        rows = []
        for cf in CF_SWEEP:
            expert_ids, weights = gate_skewed(logits, top_k)
            stats = route(expert_ids, weights, N_EXPERTS, cf)
            rows.append(stats)
        out[top_k] = rows
    return out


# ------------------------------------------------------- experiment 2
def exp2_balance_effect() -> dict:
    """Skewed vs balanced gating: drop rate at CF=1.0."""
    out: dict = {}
    for top_k in TOP_KS:
        rng = np.random.default_rng(SEED + 200 + top_k)
        logits = make_logits(rng)
        expert_ids, weights = gate_skewed(logits, top_k)
        skewed = route(expert_ids, weights, N_EXPERTS, 1.0)
        expert_ids_b, weights_b = gate_balanced(logits, top_k, rng)
        balanced = route(expert_ids_b, weights_b, N_EXPERTS, 1.0)
        out[top_k] = {"skewed": skewed, "balanced": balanced}
    return out


# ------------------------------------------------------- experiment 3
def exp3_expert_parallel() -> dict:
    """E=16 experts over D devices: all-to-all bytes and per-device load."""
    E, D_MODEL, TOP_K = 16, 4096, 2  # fp16 => 2 bytes/elem
    rows = []
    for n_devices in (1, 2, 4, 8):
        total = alltoall_bytes(N_TOKENS, TOP_K, D_MODEL, bytes_per_elem=2)
        load = device_load(N_TOKENS, TOP_K, E, n_devices, D_MODEL, 2)
        rows.append(
            {
                "n_devices": n_devices,
                "experts_per_device": load["experts_per_device"],
                "alltoall_bytes_total": total,
                "comm_bytes_per_device": load["comm_bytes_per_device"],
                "token_slots_per_device": load["token_slots_per_device"],
            }
        )
    return {"rows": rows, "n_tokens": N_TOKENS, "top_k": TOP_K,
            "d_model": D_MODEL, "n_experts": E}


# ------------------------------------------------------- experiment 4
def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def exp4_charts_and_report(exp1: dict, exp2: dict, exp3: dict) -> str:
    """Charts + results.md. Returns the headline string."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("06-moe-router: capacity factor, dropping, load balance")

    # (a) drop rate vs capacity factor, top-1/top-2 x skewed/balanced
    ax = axes[0, 0]
    rng = np.random.default_rng(SEED + 300)
    logits = make_logits(rng)
    for top_k, style in ((1, "-"), (2, "--")):
        for name, gate in (("skewed", gate_skewed), ("balanced", gate_balanced)):
            drops = []
            for cf in CF_SWEEP:
                if name == "skewed":
                    eids, w = gate(logits, top_k)
                else:
                    eids, w = gate(logits, top_k, rng)
                drops.append(100.0 * route(eids, w, N_EXPERTS, cf)["drop_rate"])
            ax.plot(CF_SWEEP, drops, style,
                    label=f"top-{top_k} {name}", marker="o", ms=4)
    ax.set_xlabel("capacity factor CF")
    ax.set_ylabel("drop rate (%)")
    ax.set_title("(a) drop rate vs capacity factor")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) per-expert load histogram at CF=1.0, skewed vs balanced (top-1)
    for j, name in enumerate(("skewed", "balanced")):
        ax = axes[0, 1] if j == 0 else axes[1, 0]
        rng_b = np.random.default_rng(SEED + 400)
        logits_b = make_logits(rng_b)
        if name == "skewed":
            eids, w = gate_skewed(logits_b, 1)
        else:
            eids, w = gate_balanced(logits_b, 1, rng_b)
        stats = route(eids, w, N_EXPERTS, 1.0)
        counts = stats["per_expert_counts"]
        ax.bar(range(N_EXPERTS), counts, color="tab:blue" if name == "skewed" else "tab:green")
        ax.axhline(stats["expert_capacity"], color="red", ls="--", lw=1,
                   label=f"capacity={stats['expert_capacity']}")
        ax.set_xlabel("expert id")
        ax.set_ylabel("routed tokens")
        ax.set_title(f"(b{j+1}) per-expert load at CF=1.0, top-1, {name}\n"
                     f"drop={100*stats['drop_rate']:.1f}%, "
                     f"imbalance={stats['imbalance_ratio']:.2f}")
        ax.legend(fontsize=8)

    # (c) expert-parallel comm tradeoff: total comm flat, per-device compute shrinks
    ax = axes[1, 1]
    ds = [r["n_devices"] for r in exp3["rows"]]
    total = [r["alltoall_bytes_total"] / 2**20 for r in exp3["rows"]]
    slots = [r["token_slots_per_device"] for r in exp3["rows"]]
    ax2 = ax.twinx()
    l1, = ax.plot(ds, total, "o--", color="tab:orange",
                  label="total all-to-all (flat)")
    l2, = ax2.plot(ds, slots, "s-", color="tab:purple",
                   label="token slots/device")
    ax.set_xlabel("devices D (E=16 experts)")
    ax.set_ylabel("total all-to-all MiB per step", color="tab:orange")
    ax2.set_ylabel("compute token slots per device", color="tab:purple")
    ax.set_title("(c) expert parallelism: total comm flat, per-device compute shrinks")
    ax.set_xticks(ds)
    ax.grid(alpha=0.3)
    ax.legend([l1, l2], [l1.get_label(), l2.get_label()], fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "charts.png", dpi=110)
    plt.close(fig)

    return _write_report(exp1, exp2, exp3)


def _write_report(exp1: dict, exp2: dict, exp3: dict) -> str:
    """Write results.md and return the headline string."""
    L: list[str] = []
    L.append("# 06-moe-router — results\n")
    L.append("Seeded `np.random.default_rng(51)`; numpy CPU only. "
             "Two consecutive runs produce identical headlines.\n")

    L.append("## Experiment 1 — routing quality vs capacity factor\n")
    L.append(f"n_tokens={N_TOKENS}, n_experts={N_EXPERTS}, skewed expert "
             f"popularity {list(POPULARITY)}.\n")
    for top_k in TOP_KS:
        L.append(f"### top-{top_k}, skewed gating\n")
        L.append("| CF | expert capacity | drop rate % | imbalance max/mean | "
                 "processed tokens |")
        L.append("|----|---------------|-------------|--------------------|------------------|")
        for s in exp1[top_k]:
            L.append(f"| {s['capacity_factor']:.2f} | {s['expert_capacity']} | "
                     f"{100*s['drop_rate']:.2f} | {s['imbalance_ratio']:.2f} | "
                     f"{s['processed_tokens']} |")
        L.append("")

    L.append("## Experiment 2 — load-balancing loss effect at CF=1.0\n")
    L.append("Balanced gating = skew shrunk 99% (idealized aux load-balancing "
             "loss) + tiny tie-break noise.\n")
    L.append("| top-k | gating | drop rate % | imbalance max/mean | processed tokens |")
    L.append("|-------|--------|-------------|--------------------|------------------|")
    for top_k in TOP_KS:
        for name in ("skewed", "balanced"):
            s = exp2[top_k][name]
            L.append(f"| {top_k} | {name} | {100*s['drop_rate']:.2f} | "
                     f"{s['imbalance_ratio']:.2f} | {s['processed_tokens']} |")
    L.append("")

    L.append("## Experiment 3 — expert parallelism\n")
    e3 = exp3
    L.append(f"E={e3['n_experts']} experts, n_tokens={e3['n_tokens']}, "
             f"top_k={e3['top_k']}, d_model={e3['d_model']}, fp16.\n")
    L.append("| devices D | experts/device | all-to-all total | comm/device | "
             "token slots/device |")
    L.append("|-----------|----------------|------------------|-------------|--------------------|")
    for r in e3["rows"]:
        L.append(f"| {r['n_devices']} | {r['experts_per_device']} | "
                 f"{_fmt_bytes(r['alltoall_bytes_total'])} | "
                 f"{_fmt_bytes(r['comm_bytes_per_device'])} | "
                 f"{r['token_slots_per_device']:.0f} |")
    L.append("")
    L.append("Total all-to-all volume is independent of D (dispatch + combine "
             "of every token->expert vector); per-device compute shrinks as "
             "1/D, so the comm/compute ratio worsens with more devices.\n")

    s1 = exp1[1][2]  # top-1, CF=1.0, skewed (index 2 in CF_SWEEP)
    b1 = exp2[1]["balanced"]
    d8 = exp3["rows"][-1]
    headline = (
        f"top-1 skewed CF=1.0: drop {100*s1['drop_rate']:.1f}%, "
        f"imbalance {s1['imbalance_ratio']:.2f}; "
        f"balanced: drop {100*b1['drop_rate']:.1f}%; "
        f"all-to-all at D=8: {_fmt_bytes(d8['comm_bytes_per_device'])}/device "
        f"({_fmt_bytes(d8['alltoall_bytes_total'])} total)"
    )
    L.append("## Headline\n")
    L.append(headline + "\n")

    (RESULTS_DIR / "results.md").write_text("\n".join(L))
    return headline


def main() -> None:
    exp1 = exp1_routing_quality()
    exp2 = exp2_balance_effect()
    exp3 = exp3_expert_parallel()
    headline = exp4_charts_and_report(exp1, exp2, exp3)
    print("[moe-router] " + headline)
    print(f"[moe-router] wrote {RESULTS_DIR / 'charts.png'}")
    print(f"[moe-router] wrote {RESULTS_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
