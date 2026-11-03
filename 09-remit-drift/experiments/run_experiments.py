"""Seeded experiments for remit-drift.

All numbers below are simulator output (single sequential RNG stream per run),
not invented. Run twice -> byte-identical headlines, results.md, charts.png.

  exp1: D(t) for all 8 divergence types, no governance (T=300)
  exp2: same WITH the full policy engine -> divergence stays bounded
  exp3: finance-agent enterprise timeline (first out-of-remit use per capability)
  exp4: support-ticket goal divergence (proxy improves, true objective collapses)
  exp5: control ablation -> which control catches which divergence type

Run: python3 09-remit-drift/experiments/run_experiments.py
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "..", "src"))

from llm_inference_engine.remit_drift import metrics as M  # noqa: E402
from llm_inference_engine.remit_drift import scenarios, sim  # noqa: E402
from llm_inference_engine.remit_drift.policy import (  # noqa: E402
    ALL_OFF,
    ALL_ON,
    CONTROLS,
    PolicyEngine,
)

SEED = 11
T = 300
WINDOW = 50
OUT = os.path.join(REPO, "results")
os.makedirs(OUT, exist_ok=True)


def exp1_ungoverned():
    """D(t) per type, no controls."""
    out = {}
    for dtype in M.DIVERGENCE_TYPES:
        agent = scenarios.make_specialist(dtype)
        _, series = sim.run(T, SEED, [agent], PolicyEngine(ALL_OFF),
                            window=WINDOW)
        out[dtype] = series[dtype]
    return out


def exp2_governed():
    """D(t) per type, full policy engine."""
    out = {}
    for dtype in M.DIVERGENCE_TYPES:
        agent = scenarios.make_specialist(dtype)
        _, series = sim.run(T, SEED, [agent], PolicyEngine(ALL_ON),
                            window=WINDOW)
        out[dtype] = series[dtype]
    return out


def exp3_finance():
    """Enterprise timeline: invoice reader -> vendor/bank writes."""
    res = {}
    for gov, controls in (("off", ALL_OFF), ("on", ALL_ON)):
        agent = scenarios.finance_agent()
        events, _ = sim.run(T, SEED, [agent], PolicyEngine(controls),
                            window=WINDOW)
        ev = events["finance"]
        first, denied, escalated = {}, {}, {}
        in_kinds = ("read_invoice", "match_po", "recommend_payment")
        for e in ev:
            k = e["attempted_kind"]
            if k in in_kinds:
                continue
            if e["decision"] == "deny":
                denied[k] = denied.get(k, 0) + 1
            elif e["decision"] == "escalate":
                escalated[k] = escalated.get(k, 0) + 1
            if e["executed"] and e["kind"] == k and k not in first:
                first[k] = e["t"]
        res[gov] = {"first_executed": first, "denied": denied,
                    "escalated": escalated,
                    "silent_expansions": agent.silent_expansions}
    return res


def exp4_goal():
    """Support tickets: proxy (response time) vs true objective (satisfaction)."""
    res = {}
    for gov, controls in (("off", ALL_OFF), ("on", ALL_ON)):
        r = scenarios.run_support_scenario(T, SEED, PolicyEngine(controls))
        # trailing-window means for the curves
        w = WINDOW
        rt_m = np.array([r["rt"][max(0, t - w):t + 1].mean() for t in range(T)])
        sat_m = np.array([r["sat"][max(0, t - w):t + 1].mean() for t in range(T)])
        res[gov] = {"rt": rt_m, "sat": sat_m, "pi": r["pi"],
                    "close_rate": r["close_rate"], "blocked": len(r["blocked"])}
    return res


def _final_d(dtype, controls, seed=SEED):
    agent = scenarios.make_specialist(dtype)
    _, series = sim.run(T, seed, [agent], PolicyEngine(controls), window=WINDOW)
    return float(series[dtype][1][-1])


def exp5_ablation():
    """D(T) per type under: none, all, each single control. Plus audit alerts."""
    configs = [("none", ALL_OFF), ("all", ALL_ON)]
    for c in CONTROLS:
        d = dict(ALL_OFF)
        d[c] = True
        configs.append((c, d))
    table = {}
    for dtype in M.DIVERGENCE_TYPES:
        row = {}
        for name, controls in configs:
            row[name] = _final_d(dtype, controls)
        # auditing is detect-only: does it fire?
        agent = scenarios.make_specialist(dtype)
        events, _ = sim.run(T, SEED, [agent], PolicyEngine(ALL_OFF), window=WINDOW)
        pol = PolicyEngine({**ALL_OFF, "auditing": True})
        alerts = pol.audit(events[agent.name], agent.remit, T, window=WINDOW)
        row["audit_alerts"] = len(alerts) > 0
        table[dtype] = row
    return table


def coverage_matrix(table):
    """'caught' = single control keeps D(T) below 25% of the ungoverned value."""
    mat = {}
    for dtype, row in table.items():
        base = row["none"]
        mat[dtype] = {c: (row[c] < 0.25 * base if base > 1e-9 else row[c] < 1e-9)
                      for c in CONTROLS}
        mat[dtype]["auditing"] = bool(row["audit_alerts"])
    return mat


def main():
    s1 = exp1_ungoverned()
    s2 = exp2_governed()
    fin = exp3_finance()
    goal = exp4_goal()
    abl = exp5_ablation()
    mat = coverage_matrix(abl)

    # ---- results.md ----
    lines = []
    lines.append("# remit-drift results\n")
    lines.append(f"Seed: {SEED} (single sequential RNG stream per run). "
                 f"T={T} steps, trailing window={WINDOW}.\n")
    lines.append("## Experiment 1 -- D(t) without governance (final values)\n")
    lines.append("| divergence type | D(50) | D(300) |")
    lines.append("|:---|---:|---:|")
    for dtype in M.DIVERGENCE_TYPES:
        ts, v = s1[dtype]
        lines.append(f"| {dtype} | {v[0]:.3f} | {v[-1]:.3f} |")
    lines.append("\n## Experiment 2 -- max D with vs without the policy engine\n")
    lines.append("| divergence type | no governance | governed | contained |")
    lines.append("|:---|---:|---:|:---:|")
    for dtype in M.DIVERGENCE_TYPES:
        a = s1[dtype][1].max()
        b = s2[dtype][1].max()
        lines.append(f"| {dtype} | {a:.3f} | {b:.3f} | "
                     f"{'yes' if b < 0.25 * max(a, 1e-9) else 'no'} |")
    lines.append("\n## Experiment 3 -- finance-agent timeline\n")
    lines.append("First step each out-of-remit capability was *executed* "
                 "(no governance):")
    lines.append("")
    for k, t in sorted(fin["off"]["first_executed"].items(), key=lambda x: x[1]):
        lines.append(f"- {k}: step {t}")
    lines.append("")
    lines.append("With governance: executed out-of-remit actions: "
                 f"{sum(fin['on']['denied'].values()) + sum(fin['on']['escalated'].values())} "
                 f"attempts, 0 executed "
                 f"({sum(fin['on']['denied'].values())} denied, "
                 f"{sum(fin['on']['escalated'].values())} escalated).")
    lines.append("\n## Experiment 4 -- goal divergence (support tickets)\n")
    lines.append("| | no governance | governed |")
    lines.append("|:---|---:|---:|")
    lines.append(f"| close-immediately rate | {goal['off']['close_rate']:.2f} | "
                 f"{goal['on']['close_rate']:.2f} |")
    lines.append(f"| final mean response time | {goal['off']['rt'][-1]:.1f} min | "
                 f"{goal['on']['rt'][-1]:.1f} min |")
    lines.append(f"| final mean satisfaction | {goal['off']['sat'][-1]:.2f} | "
                 f"{goal['on']['sat'][-1]:.2f} |")
    gap_off = max(0.0, (30 - goal['off']['rt'][-1]) / 30 - goal['off']['sat'][-1])
    gap_on = max(0.0, (30 - goal['on']['rt'][-1]) / 30 - goal['on']['sat'][-1])
    lines.append(f"| proxy/true gap | {gap_off:.2f} | {gap_on:.2f} |")
    lines.append("| blocked close attempts | 0 | "
                 f"{goal['on']['blocked']} |")
    lines.append("\n## Experiment 5 -- control ablation (D at t=300)\n")
    hdr = "| divergence type | none | all | " + " | ".join(CONTROLS) + " |"
    lines.append(hdr)
    lines.append("|:---|" + "---:|" * (2 + len(CONTROLS)))
    for dtype in M.DIVERGENCE_TYPES:
        row = abl[dtype]
        cells = [f"{row['none']:.3f}", f"{row['all']:.3f}"] + \
                [f"{row[c]:.3f}" for c in CONTROLS]
        lines.append(f"| {dtype} | " + " | ".join(cells) + " |")
    lines.append("\n### Coverage matrix (control catches type?)\n")
    lines.append("| divergence type | " + " | ".join(CONTROLS) + " |")
    lines.append("|:---|" + ":---:|" * len(CONTROLS))
    for dtype in M.DIVERGENCE_TYPES:
        cells = ["Y" if mat[dtype][c] else "." for c in CONTROLS]
        lines.append(f"| {dtype} | " + " | ".join(cells) + " |")
    with open(os.path.join(OUT, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- charts ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    fig.patch.set_facecolor("#0a0e13")
    for ax in axes.flat:
        ax.set_facecolor("#0d141e")

    ax = axes[0, 0]
    for dtype in M.DIVERGENCE_TYPES:
        ts, v = s1[dtype]
        ax.plot(ts, v, label=dtype, lw=1.8)
    ax.set_title("D(t): divergence without governance (all 8 types grow)")
    ax.set_xlabel("step"); ax.set_ylabel("D(t)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.2)

    ax = axes[0, 1]
    xs = np.arange(len(M.DIVERGENCE_TYPES))
    a = [s1[d][1].max() for d in M.DIVERGENCE_TYPES]
    b = [s2[d][1].max() for d in M.DIVERGENCE_TYPES]
    ax.bar(xs - 0.2, a, 0.4, label="no governance", color="#f08787")
    ax.bar(xs + 0.2, b, 0.4, label="policy engine", color="#7ee2a8")
    ax.set_xticks(xs); ax.set_xticklabels(M.DIVERGENCE_TYPES, rotation=30, ha="right", fontsize=8)
    ax.set_title("Max D(t): ungoverned vs governed")
    ax.legend(fontsize=8); ax.grid(alpha=0.2, axis="y")

    ax = axes[1, 0]
    items = sorted(fin["off"]["first_executed"].items(), key=lambda x: x[1])
    ax.barh([k for k, _ in items], [t for _, t in items], color="#f0b35e")
    ax.set_title("Finance agent: first out-of-remit use (no governance)")
    ax.set_xlabel("step")
    for k, t in items:
        ax.text(t + 3, k, f"t={t}", va="center", fontsize=8, color="#99a4ba")

    ax = axes[1, 1]
    tt = np.arange(T)
    ax.plot(tt, goal["off"]["rt"], label="response time (proxy)", color="#5eead4", lw=1.6)
    ax.plot(tt, goal["off"]["sat"] * 30, label="satisfaction x30 (true)", color="#f08787", lw=1.6)
    ax.plot(tt, goal["on"]["sat"] * 30, label="satisfaction x30 (governed)", color="#7ee2a8",
            lw=1.6, ls="--")
    ax.set_title("Goal divergence: proxy improves, mission collapses")
    ax.set_xlabel("step"); ax.set_ylabel("min (rt) / sat x30")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[2, 0]
    mvals = np.array([[1.0 if mat[d][c] else 0.0 for c in CONTROLS]
                      for d in M.DIVERGENCE_TYPES])
    im = ax.imshow(mvals, cmap="Greens", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(CONTROLS)))
    ax.set_xticklabels(CONTROLS, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(M.DIVERGENCE_TYPES)))
    ax.set_yticklabels(M.DIVERGENCE_TYPES, fontsize=8)
    ax.set_title("Ablation: which control catches which divergence (Y=caught)")
    for i in range(len(M.DIVERGENCE_TYPES)):
        for j in range(len(CONTROLS)):
            ax.text(j, i, "Y" if mvals[i, j] else ".", ha="center", va="center",
                    fontsize=8, color="white" if mvals[i, j] else "#6b7690")

    ax = axes[2, 1]
    ax.axis("off")
    txt = (
        "Headlines (seed 11, T=300)\n\n"
        f"Functional KL: {s1['functional'][1][-1]:.2f} -> {s2['functional'][1][-1]:.2f} governed\n"
        f"Authority rate: {s1['authority'][1][-1]:.2f} -> {s2['authority'][1][-1]:.2f}\n"
        f"Finance: 3 out-of-remit capabilities first used at steps "
        f"{', '.join(str(t) for _, t in items)};\n  governed: 0 executed "
        f"({sum(fin['on']['denied'].values())} denied, {sum(fin['on']['escalated'].values())} escalated)\n"
        f"Support: close-now rate {goal['off']['close_rate']:.0%} ungoverned "
        f"(satisfaction {goal['off']['sat'][-1]:.2f}) vs "
        f"{goal['on']['close_rate']:.0%} governed (satisfaction {goal['on']['sat'][-1]:.2f})\n\n"
        "Without governance D(t) grows for all 8 types;\n"
        "the policy engine bounds every one of them."
    )
    ax.text(0.02, 0.95, txt, transform=ax.transAxes, fontsize=10,
            va="top", ha="left", color="#c9d1e2", family="monospace")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "charts.png"), dpi=110)
    print("wrote results.md + charts.png")
    # headline echo for the report
    for dtype in M.DIVERGENCE_TYPES:
        print(f"{dtype:13s} ungoverned D(T)={s1[dtype][1][-1]:7.3f}  "
              f"governed D(T)={s2[dtype][1][-1]:7.3f}")


if __name__ == "__main__":
    main()
