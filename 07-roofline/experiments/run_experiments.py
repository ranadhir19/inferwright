"""Roofline experiments for LLM inference ops.

Fully analytical (no RNG; numpy seed set for form): run twice and the
outputs are byte-identical.

Experiments:
  1. Decode step (B=1, S=2048, LLaMA-7B-ish shapes), A100 and H100.
  2. Prefill step (B=8, S=2048), A100 and H100.
  3. Fusion estimates: FlashAttention-style attention fusion (decode and
     prefill) plus fused dequant+GEMM (INT8 weights) on decode.
  4. Charts + results.md with full tables.
"""

from __future__ import annotations

import hashlib
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm_inference_engine.roofline import (  # noqa: E402
    A100,
    H100,
    LayerConfig,
    classify,
    fuse_dequant_gemm,
    fuse_flashattention,
    layer_ops,
    step_summary,
)

np.random.seed(7)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

DECODE_CFG = LayerConfig(d_model=4096, n_heads=32, ffn_mult=2.0, batch=1, seq=2048, dtype_bytes=2)
PREFILL_CFG = LayerConfig(d_model=4096, n_heads=32, ffn_mult=2.0, batch=8, seq=2048, dtype_bytes=2)

HW = {"A100": A100, "H100": H100}
DTYPE = "fp16"


def fmt_flops(x: float) -> str:
    return f"{x / 1e9:.2f} GFLOP" if x >= 1e9 else f"{x / 1e6:.2f} MFLOP"


def fmt_bytes(x: float) -> str:
    return f"{x / 1e9:.3f} GB" if x >= 1e9 else f"{x / 1e6:.2f} MB"


def table(classified) -> list[list[str]]:
    rows = []
    for c in classified:
        rows.append(
            [
                c.name,
                fmt_flops(c.flops),
                fmt_bytes(c.bytes),
                f"{c.intensity:.2f}",
                c.bound,
                f"{c.pct_peak:.2f}%",
                f"{c.time_s * 1e6:.2f} us" if c.time_s < 1e-3 else f"{c.time_s * 1e3:.2f} ms",
            ]
        )
    return rows


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    out: dict = {}

    md = ["# Roofline analysis results", ""]
    md.append("Shapes: d_model=4096, n_heads=32, ffn_mult=2.0 (LLaMA-7B-ish), fp16.")
    md.append("Hardware values are published specs (see module docstring).")
    md.append("")

    header = ["op", "FLOPs", "bytes", "intensity (FLOP/B)", "bound", "% of peak", "est. time"]

    # --- Experiments 1 & 2: decode and prefill on both GPUs ---
    for mode, cfg in (("decode", DECODE_CFG), ("prefill", PREFILL_CFG)):
        for hw_name, hw in HW.items():
            cl = classify(layer_ops(cfg, mode), hw, DTYPE)
            s = step_summary(cl)
            out[f"{mode}_{hw_name}"] = {
                "rows": [[c.name, c.flops, c.bytes, c.intensity, c.bound, c.pct_peak, c.time_s] for c in cl],
                "summary": s,
                "ridge": hw.ridge_point(DTYPE),
            }
            md.append(f"## {mode} on {hw.name} (ridge = {hw.ridge_point(DTYPE):.1f} FLOP/byte)")
            md.append("")
            md.append(md_table(header, table(cl)))
            md.append("")
            step_us = s["step_time_s"] * 1e6
            step_str = f"{step_us / 1e3:.2f} ms" if step_us >= 1000 else f"{step_us:.1f} us"
            md.append(
                f"Step: {fmt_flops(s['total_flops'])}, {fmt_bytes(s['total_bytes'])}, "
                f"step intensity {s['step_intensity']:.2f} FLOP/byte, "
                f"est. step time {step_str}; "
                f"{s['n_memory_bound']}/{len(cl)} ops memory-bound; "
                f"dominated by {s['dominant_op']} "
                f"({100 * s['dominant_share']:.1f}% of step time)."
            )
            md.append("")

    # --- Experiment 3: fusion estimates ---
    md.append("## Fusion estimates")
    md.append("")
    for mode, cfg in (("decode", DECODE_CFG), ("prefill", PREFILL_CFG)):
        hw = A100
        base = layer_ops(cfg, mode)
        base_cl = classify(base, hw, DTYPE)
        base_s = step_summary(base_cl)

        fused = fuse_flashattention(base)
        fused_cl = classify(fused, hw, DTYPE)
        fused_s = step_summary(fused_cl)

        attn_names = {"attn_qk", "softmax", "attn_av"}
        b_i = sum(c.intensity * c.flops for c in base_cl if c.name in attn_names) / sum(
            c.flops for c in base_cl if c.name in attn_names
        )
        b_t = sum(c.time_s for c in base_cl if c.name in attn_names)
        f = next(c for c in fused_cl if c.name == "attn_fused")
        f_t = f.time_s
        speedup = base_s["step_time_s"] / fused_s["step_time_s"]
        md.append(f"### FlashAttention-style fusion, {mode} (A100)")
        md.append("")
        md.append(
            f"- Attention intensity: {b_i:.2f} -> {f.intensity:.2f} FLOP/byte "
            f"(flop-weighted before; bound {f.bound})."
        )
        md.append(f"- Attention time: {b_t * 1e6:.2f} us -> {f_t * 1e6:.2f} us.")
        md.append(
            f"- Step time: {base_s['step_time_s'] * 1e6:.2f} us -> "
            f"{fused_s['step_time_s'] * 1e6:.2f} us "
            f"(estimated step speedup {speedup:.3f}x)."
        )
        md.append("")
        out[f"flash_{mode}"] = {"before_s": base_s["step_time_s"], "after_s": fused_s["step_time_s"], "speedup": speedup}

    # Fused dequant+GEMM on decode (INT8 weights, 1 byte/param)
    hw = A100
    base = layer_ops(DECODE_CFG, "decode")
    base_s = step_summary(classify(base, hw, DTYPE))
    dq = fuse_dequant_gemm(base, quant_bytes=1, dtype_bytes=2)
    dq_s = step_summary(classify(dq, hw, DTYPE))
    dq_speedup = base_s["step_time_s"] / dq_s["step_time_s"]
    proj_i = [c.intensity for c in classify(base, hw, DTYPE) if c.name in ("qkv_proj", "out_proj", "ffn_gate_up", "ffn_down")]
    proj_i2 = [c.intensity for c in classify(dq, hw, DTYPE) if c.name.endswith("_dequant")]
    md.append("### Fused dequant+GEMM (INT8 weights), decode (A100)")
    md.append("")
    md.append(
        f"- GEMM op intensity: {[f'{v:.2f}' for v in proj_i]} -> "
        f"{[f'{v:.2f}' for v in proj_i2]} FLOP/byte (still memory-bound; "
        "dequant ALU cost neglected)."
    )
    md.append(
        f"- Step time: {base_s['step_time_s'] * 1e6:.2f} us -> "
        f"{dq_s['step_time_s'] * 1e6:.2f} us "
        f"(estimated step speedup {dq_speedup:.3f}x)."
    )
    md.append("")
    out["dequant_decode"] = {"before_s": base_s["step_time_s"], "after_s": dq_s["step_time_s"], "speedup": dq_speedup}

    with open(os.path.join(RESULTS, "results.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    make_charts(out)

    # --- Determinism fingerprint ---
    digest = hashlib.sha256()
    for fn in ("results.md", "charts.png"):
        with open(os.path.join(RESULTS, fn), "rb") as f:
            digest.update(f.read())
    fingerprint = digest.hexdigest()
    with open(os.path.join(RESULTS, "fingerprint.txt"), "w") as f:
        f.write(fingerprint + "\n")
    print("fingerprint:", fingerprint)

    # Console headline
    d = out["decode_A100"]["summary"]
    print(
        f"decode A100: {d['n_memory_bound']}/7 ops memory-bound, "
        f"dominated by {d['dominant_op']} ({100 * d['dominant_share']:.1f}% of step, "
        f"{d['step_time_s'] * 1e6:.1f} us/step)"
    )
    p = out["prefill_A100"]["summary"]
    print(
        f"prefill A100: {p['n_memory_bound']}/7 ops memory-bound, "
        f"dominated by {p['dominant_op']} ({100 * p['dominant_share']:.1f}% of step, "
        f"{p['step_time_s'] * 1e3:.2f} ms/step)"
    )
    print(f"FlashAttention fusion step speedup: decode {out['flash_decode']['speedup']:.3f}x, "
          f"prefill {out['flash_prefill']['speedup']:.3f}x")
    print(f"INT8 fused-dequant step speedup (decode): {out['dequant_decode']['speedup']:.3f}x")
    return out


def make_charts(out: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # (a) Roofline log-log plot, A100, decode op points + ceilings
    ax = axes[0]
    peak, bw, ridge = A100.peak(DTYPE), A100.memory_bw, A100.ridge_point(DTYPE)
    xs = np.logspace(-2, 5, 400)
    ax.loglog(xs, np.minimum(peak, xs * bw), "k-", lw=2, label="roofline")
    ax.axhline(peak, ls="--", color="gray", lw=1)
    ax.axvline(ridge, ls=":", color="gray", lw=1)
    ax.text(ridge * 1.15, peak * 0.55, f"ridge {ridge:.0f} FLOP/B", fontsize=9)
    ax.text(0.02, peak * 1.15, f"compute ceiling {peak / 1e12:.0f} TFLOPS", fontsize=9)
    rows = out["decode_A100"]["rows"]
    offsets = [(8, 8), (8, -22), (-64, 10), (8, -34), (-70, -18), (10, 20), (-66, -34)]
    for (name, flops, nbytes, intensity, bound, pct, t), (dx, dy) in zip(rows, offsets):
        y = min(peak, intensity * bw)
        ax.plot(intensity, y, "o", ms=8)
        ax.annotate(
            f"{name}\n{pct:.1f}% peak",
            (intensity, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
        )
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Attainable performance (FLOP/s)")
    ax.set_title("Roofline, A100 (fp16) - decode step, B=1 S=2048")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1e-2, 1e5)

    # (b) Per-op % of peak bars, decode vs prefill (A100)
    ax = axes[1]
    d_rows = {r[0]: r[5] for r in out["decode_A100"]["rows"]}
    p_rows = {r[0]: r[5] for r in out["prefill_A100"]["rows"]}
    names = list(d_rows)
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, [d_rows[n] for n in names], w, label="decode (B=1)")
    ax.bar(x + w / 2, [p_rows[n] for n in names], w, label="prefill (B=8)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% of A100 fp16 peak")
    ax.set_title("Per-op % of peak: decode vs prefill (A100)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "charts.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
