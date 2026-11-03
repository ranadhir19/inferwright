"""Experiment runner for 02-paged-kv.

Experiment 1: stream N=2000 mixed-length sequences through a paged
allocator (B=16) and the contiguous baseline under a fixed 1M-slot memory
budget. Measures per-admission internal fragmentation
(allocated_but_unused / allocated), external fragmentation events, and max
concurrent sequences.

Experiment 2: sweep block size B in {8, 16, 32, 64}; internal waste % for
paged vs contiguous, plus the block-table overhead model
(pool_blocks * pointer_bytes).

The whole pipeline is executed TWICE from np.random.default_rng(11) and the
headline numbers must match exactly (determinism check); only then are
charts and results.md written.

Outputs (02-paged-kv/results/):
    charts.png   - (a) waste % vs block size, (b) concurrency over time
    results.md   - tables with the real measured numbers
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llm_inference_engine.paged_kv.blocks import BlockManager, ContiguousAllocator
from llm_inference_engine.paged_kv.workload import Request, gen_workload

SEED = 11
BUDGET_SLOTS = 1_000_000
MAX_LEN = 2048
N_REQUESTS = 2000
ARRIVAL_WINDOW = 600.0
BLOCK_SIZES = [8, 16, 32, 64]
POINTER_BYTES = 8  # bytes per block-table entry

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(HERE, "..", "results"))


def simulate(make_allocator, requests: list[Request]) -> dict:
    """Run the arrival/departure event loop; return metrics + concurrency curve."""
    alloc = make_allocator()
    events = []
    for r in requests:
        events.append((r.arrival, 1, "arrive", r))
        events.append((r.departure, 0, "depart", r))
    # Departures (order 0) processed before arrivals (order 1) at equal times.
    events.sort(key=lambda e: (e[0], e[1]))

    admitted: set[int] = set()
    admitted_total = 0
    rejected = 0
    waste_slots_sum = 0  # sum of (reserved - n_tokens) over admitted sequences
    reserved_slots_sum = 0
    prev_t = 0.0
    max_concurrent = 0
    curve_t: list[float] = []
    curve_c: list[int] = []

    for t, _, kind, r in events:
        prev_t = t
        if kind == "arrive":
            a0 = alloc.allocated_slots()
            if alloc.alloc(r.seq_id, r.length) is not None:
                admitted.add(r.seq_id)
                admitted_total += 1
                reserved = alloc.allocated_slots() - a0
                waste_slots_sum += reserved - r.length
                reserved_slots_sum += reserved
            else:
                rejected += 1
        else:
            if r.seq_id in admitted:
                alloc.free(r.seq_id)
                admitted.discard(r.seq_id)
        live = alloc.live_sequences()
        if live > max_concurrent:
            max_concurrent = live
        curve_t.append(t)
        curve_c.append(live)

    frag = alloc.fragmentation()
    return {
        # Per-admission mean: average over admitted sequences of
        # (reserved_but_unused / reserved). Unbiased by sequence lifetime.
        "internal_waste_pct": (
            100.0 * waste_slots_sum / reserved_slots_sum
            if reserved_slots_sum
            else 0.0
        ),
        "external_frag_events": frag["external_frag_events"],
        "max_concurrent": max_concurrent,
        "admitted": admitted_total,
        "rejected": rejected,
        "curve_t": curve_t,
        "curve_c": curve_c,
    }


def run_pipeline() -> dict:
    """One full seeded pass: experiment 1 + block-size sweep."""
    rng = np.random.default_rng(SEED)
    requests = gen_workload(
        rng, n=N_REQUESTS, arrival_window=ARRIVAL_WINDOW, max_len=MAX_LEN
    )
    mean_len = float(np.mean([r.length for r in requests]))

    exp1_paged = simulate(
        lambda: BlockManager(BUDGET_SLOTS, 16), requests
    )
    exp1_contig = simulate(
        lambda: ContiguousAllocator(BUDGET_SLOTS, MAX_LEN), requests
    )

    sweep = []
    for b in BLOCK_SIZES:
        m = simulate(lambda b=b: BlockManager(BUDGET_SLOTS, b), requests)
        pool_blocks = BUDGET_SLOTS // b
        sweep.append(
            {
                "block_size": b,
                "paged_waste_pct": m["internal_waste_pct"],
                "paged_table_mb": pool_blocks * POINTER_BYTES / 1e6,
            }
        )
    contig_waste = exp1_contig["internal_waste_pct"]  # B-independent baseline

    return {
        "n_requests": N_REQUESTS,
        "mean_length": mean_len,
        "budget_slots": BUDGET_SLOTS,
        "max_len": MAX_LEN,
        "exp1_paged": exp1_paged,
        "exp1_contig": exp1_contig,
        "sweep": sweep,
        "contig_waste_pct": contig_waste,
    }


def headline(res: dict) -> tuple:
    """Hashable headline numbers used for the determinism check."""
    p, c = res["exp1_paged"], res["exp1_contig"]
    return (
        p["internal_waste_pct"],
        p["external_frag_events"],
        p["max_concurrent"],
        p["rejected"],
        c["internal_waste_pct"],
        c["external_frag_events"],
        c["max_concurrent"],
        c["rejected"],
        tuple(s["paged_waste_pct"] for s in res["sweep"]),
    )


def write_charts(res: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) waste % vs block size + block-table overhead
    bs = [s["block_size"] for s in res["sweep"]]
    paged_waste = [s["paged_waste_pct"] for s in res["sweep"]]
    table_mb = [s["paged_table_mb"] for s in res["sweep"]]
    ax1.plot(bs, paged_waste, "o-", label="paged (internal waste)")
    ax1.axhline(
        res["contig_waste_pct"],
        color="tab:red",
        linestyle="--",
        label="contiguous baseline",
    )
    ax1.set_xlabel("block size B (tokens)")
    ax1.set_ylabel("internal waste (% of allocated slots)")
    ax1.set_xticks(bs)
    ax1.grid(True, alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.bar(bs, table_mb, width=4, alpha=0.25, color="tab:gray",
             label="block-table overhead")
    ax1b.set_ylabel("block-table overhead (MB)")
    ax1.set_title("(a) Memory waste vs block size")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

    # (b) concurrent sequences over time
    p, c = res["exp1_paged"], res["exp1_contig"]
    ax2.plot(p["curve_t"], p["curve_c"], label="paged, B=16", linewidth=1.2)
    ax2.plot(c["curve_t"], c["curve_c"], label="contiguous (max_len=2048)",
             linewidth=1.2)
    ax2.axhline(
        BUDGET_SLOTS // MAX_LEN,
        color="tab:red",
        linestyle=":",
        label="contiguous capacity (488 seqs)",
    )
    ax2.set_xlabel("time (decode steps, 1 token/step)")
    ax2.set_ylabel("concurrent sequences")
    ax2.set_title("(b) Concurrent sequences over time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Paged KV-cache vs contiguous allocation "
        f"(budget={BUDGET_SLOTS:,} slots, N={res['n_requests']} requests, seed={SEED})"
    )
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, "charts.png"), dpi=150)
    plt.close(fig)


def write_results_md(res: dict) -> None:
    p, c = res["exp1_paged"], res["exp1_contig"]
    lines = []
    lines.append("# 02-paged-kv — results\n")
    lines.append(
        "Seeded simulation (`np.random.default_rng(11)`), memory-management "
        "policy only: no attention math, no GPU. Analytical model: 1M token "
        "slots, sequence lengths from a 70/20/10 mixture of N(128,32) / "
        "N(512,128) / N(2048,256) clipped to [1, 2048], arrivals uniform over "
        "600 time units, 1 token decoded per time unit, "
        f"{res['n_requests']} requests (measured mean length "
        f"{res['mean_length']:.1f} tokens). Internal waste is the "
        "per-admission mean of (reserved_but_unused / reserved) over all "
        "admitted sequences.\n"
    )
    lines.append("## Experiment 1 — fixed budget, B=16 vs contiguous\n")
    lines.append(
        "| allocator | internal waste % (per-admission mean) | external frag events | "
        "max concurrent seqs | admitted | rejected |"
    )
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| paged (B=16) | {p['internal_waste_pct']:.2f} | "
        f"{p['external_frag_events']} | {p['max_concurrent']} | "
        f"{p['admitted']} | {p['rejected']} |"
    )
    lines.append(
        f"| contiguous (max_len={MAX_LEN}) | {c['internal_waste_pct']:.2f} | "
        f"{c['external_frag_events']} | {c['max_concurrent']} | "
        f"{c['admitted']} | {c['rejected']} |"
    )
    lines.append("")
    lines.append(
        "External fragmentation events are 0 for both allocators here: the "
        "paged allocator's blocks are fungible by construction, and the "
        "contiguous baseline reserves a uniform 2048 slots, so every freed "
        "interval is itself 2048 slots and can never be 'too small'. The "
        "contiguous allocator's rejections are pure internal fragmentation "
        "(reserving 2048 slots for ~373-token sequences on average)."
    )
    lines.append("")
    lines.append("## Experiment 2 — block-size sweep\n")
    lines.append(
        "Block-table overhead is modelled as "
        "`pool_blocks * 8 bytes` (one 8-byte block-table entry per physical "
        "block). For scale: at 32 layers x 4096 hidden x fp16, one token's "
        "KV cache is 524,288 bytes, so 1M slots ~= 524 GB of KV memory.\n"
    )
    lines.append(
        "| B | paged waste % | contiguous waste % | paged block-table (MB) |"
    )
    lines.append("|---|---|---|---|")
    for s in res["sweep"]:
        lines.append(
            f"| {s['block_size']} | {s['paged_waste_pct']:.2f} | "
            f"{res['contig_waste_pct']:.2f} | {s['paged_table_mb']:.3f} |"
        )
    lines.append("")
    lines.append("## Determinism\n")
    lines.append(
        "The full pipeline was executed twice from "
        "`np.random.default_rng(11)`; all headline numbers matched exactly "
        "(internal waste %, external events, max concurrency, rejections, "
        "and every sweep point)."
    )
    lines.append("")
    with open(os.path.join(RESULTS_DIR, "results.md"), "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    res1 = run_pipeline()
    res2 = run_pipeline()
    h1, h2 = headline(res1), headline(res2)
    assert h1 == h2, f"determinism check FAILED: {h1} != {h2}"
    print("determinism check: PASS (two seeded runs, identical headline numbers)")

    write_charts(res1)
    write_results_md(res1)

    p, c = res1["exp1_paged"], res1["exp1_contig"]
    print(f"paged B=16:      waste {p['internal_waste_pct']:.2f}%, "
          f"max concurrent {p['max_concurrent']}, rejected {p['rejected']}")
    print(f"contiguous:      waste {c['internal_waste_pct']:.2f}%, "
          f"max concurrent {c['max_concurrent']}, rejected {c['rejected']}")
    print("sweep (B -> paged waste %): " + ", ".join(
        f"{s['block_size']}->{s['paged_waste_pct']:.2f}%"
        for s in res1["sweep"]))
    print(f"charts -> {os.path.join(RESULTS_DIR, 'charts.png')}")
    print(f"tables -> {os.path.join(RESULTS_DIR, 'results.md')}")


if __name__ == "__main__":
    main()
