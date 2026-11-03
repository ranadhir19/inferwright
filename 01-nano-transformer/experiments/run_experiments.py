"""Experiments for 01-nano-transformer.

1. Numerical verification (assert-level, tabulated):
   - RMSNorm vs naive loop implementation
   - RoPE vs direct complex-exponential rotation
   - GQA attention vs full-head attention with tied weights
   - Incremental KV-cache decode vs full-sequence forward (exact match)
2. Benchmarks (median of reps, time.perf_counter):
   - Per-token decode latency vs sequence length, WITH KV cache vs WITHOUT
     (full-context recompute)
   - Prefill latency vs prompt length
   - KV cache memory vs sequence length (analytic, fp16)
3. Charts + results.md written to 01-nano-transformer/results/.

All RNG is seeded (np.random.default_rng(7)); only wall-clock timings vary
between runs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_inference_engine.transformer import (
    KVCache,
    RMSNorm,
    TransformerConfig,
    TransformerModel,
    apply_rope,
    build_rope_cache,
    rmsnorm_naive,
    rope_complex_reference,
)
from llm_inference_engine.transformer.model import repeat_kv, sdpa

RESULTS_DIR = REPO_ROOT / "01-nano-transformer" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 7
SEQ_LENS = [32, 64, 128, 256, 512]
REPS = 15
WARMUP = 5


# ---------------------------------------------------------------------------
# 1. Numerical verification
# ---------------------------------------------------------------------------


def verify_rmsnorm(rng: np.random.Generator) -> float:
    norm = RMSNorm(128)
    norm.weight = rng.standard_normal(128).astype(np.float32)
    x = (rng.standard_normal((64, 128)) * 3).astype(np.float32)
    got = norm(x)
    ref = rmsnorm_naive(x, norm.weight, norm.eps)
    diff = float(np.max(np.abs(got - ref)))
    assert diff < 1e-6, f"RMSNorm mismatch: {diff}"
    return diff


def verify_rope(rng: np.random.Generator) -> float:
    head_dim, t = 16, 24
    cos, sin = build_rope_cache(64, head_dim, theta=10000.0)
    x = rng.standard_normal((t, 4, head_dim)).astype(np.float32)
    got = apply_rope(x, cos[:t][:, None, :], sin[:t][:, None, :])
    angles = np.arccos(cos[:t].astype(np.float64))  # recover rotation angles
    # recover signed angles from cos/sin
    angles = np.arctan2(sin[:t].astype(np.float64), cos[:t].astype(np.float64))
    ref = rope_complex_reference(x, angles[:, None, :])
    diff = float(np.max(np.abs(got - ref)))
    assert diff < 1e-5, f"RoPE mismatch: {diff}"
    return diff


def verify_gqa(rng: np.random.Generator) -> float:
    """GQA attention must equal full-head attention when KV weights are tied."""
    n_heads, n_kv_heads, head_dim, t = 8, 2, 16, 32
    n_rep = n_heads // n_kv_heads
    q = rng.standard_normal((t, n_heads, head_dim)).astype(np.float32)
    k = rng.standard_normal((t, n_kv_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((t, n_kv_heads, head_dim)).astype(np.float32)
    got = sdpa(q, k, v, n_rep=n_rep)
    # Full-head path: explicitly repeat KV heads, then plain attention (n_rep=1).
    ref = sdpa(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), n_rep=1)
    diff = float(np.max(np.abs(got - ref)))
    assert diff == 0.0, f"GQA mismatch: {diff}"
    return diff


def verify_cache_exactness(model: TransformerModel, rng: np.random.Generator) -> float:
    cfg = model.config
    tokens = rng.integers(0, cfg.vocab_size, size=64)
    full = model.forward(tokens)
    cache = KVCache(cfg)
    inc = np.stack([model.step(int(tok), cache) for tok in tokens])
    diff = float(np.max(np.abs(full - inc)))
    assert diff < 1e-6, f"cache/full mismatch: {diff}"
    return diff


# ---------------------------------------------------------------------------
# 2. Benchmarks
# ---------------------------------------------------------------------------


def median_time(fn, reps: int = REPS, warmup: int = WARMUP) -> float:
    """Median wall-clock time of ``fn()`` over reps (warmup excluded)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def median_return(fn, reps: int = REPS, warmup: int = WARMUP) -> float:
    """Median of the values returned by ``fn()`` over reps (warmup excluded).

    Use when ``fn`` times an inner region itself (e.g. per-token latency over
    several decode steps, excluding untimed prefill).
    """
    for _ in range(warmup):
        fn()
    return float(np.median([fn() for _ in range(reps)]))


DECODE_STEPS = 16  # consecutive timed steps per rep for the cached path


def bench_decode(model: TransformerModel, rng: np.random.Generator) -> dict:
    """Per-token decode latency vs context length, with/without KV cache.

    WITH cache: prefill to L once (untimed) and snapshot the filled prefix.
    Each rep restores the snapshot into a reusable cache (fast memcpy, outside
    the timed region), then times DECODE_STEPS consecutive step() calls and
    reports the mean per-token latency. WITHOUT cache: time one full forward
    pass over L+1 tokens (full-context recompute per step).
    """
    cfg = model.config
    rows = []
    for L in SEQ_LENS:
        ctx = rng.integers(0, cfg.vocab_size, size=L)
        step_toks = rng.integers(0, cfg.vocab_size, size=DECODE_STEPS * (WARMUP + REPS))
        tok_iter = iter(step_toks.tolist())

        # Prefill once (untimed); snapshot the L filled rows per layer.
        cache0 = KVCache(cfg)
        model.prefill(ctx, cache0)
        snap = [
            (cache0.k[li][:L].copy(), cache0.v[li][:L].copy())
            for li in range(cfg.n_layers)
        ]
        c2 = KVCache(cfg)

        def decode_cached():
            for li, (sk, sv) in enumerate(snap):
                c2.k[li][:L] = sk
                c2.v[li][:L] = sv
            c2.seq_len = L
            t0 = time.perf_counter()
            for _ in range(DECODE_STEPS):
                model.step(int(next(tok_iter)), c2)
            return (time.perf_counter() - t0) / DECODE_STEPS

        t_cached = median_return(decode_cached)

        # WITHOUT cache: recompute the full context forward pass per step.
        full_ctx = np.concatenate([ctx, step_toks[:1]])
        t_recompute = median_time(lambda: model.forward(full_ctx))

        rows.append((L, t_cached, t_recompute))
    return rows


def bench_prefill(model: TransformerModel, rng: np.random.Generator) -> list:
    cfg = model.config
    rows = []
    for L in SEQ_LENS:
        toks = rng.integers(0, cfg.vocab_size, size=L)
        t = median_time(lambda: model.forward(toks))
        rows.append((L, t))
    return rows


def kv_cache_bytes(cfg: TransformerConfig, seq_len: int) -> int:
    # 2 (k+v) * n_layers * n_kv_heads * head_dim * seq_len * 2 bytes (fp16)
    return KVCache.bytes_for(cfg, seq_len, bytes_per_elem=2)


# ---------------------------------------------------------------------------
# 3. Charts + results.md
# ---------------------------------------------------------------------------


def make_charts(decode_rows, cfg) -> Path:
    seq = [r[0] for r in decode_rows]
    t_cache = [r[1] * 1e3 for r in decode_rows]
    t_recompute = [r[2] * 1e3 for r in decode_rows]
    mem_kb = [kv_cache_bytes(cfg, L) / 1024 for L in seq]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(seq, t_cache, "o-", label="with KV cache")
    ax.plot(seq, t_recompute, "s-", label="recompute (no cache)")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("per-token decode latency (ms)")
    ax.set_title("(a) Decode latency vs sequence length")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(seq, [str(s) for s in seq])
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(seq, mem_kb, "o-", color="tab:green")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("KV cache size (KiB, fp16)")
    ax.set_title("(b) KV cache memory vs sequence length")
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq, [str(s) for s in seq])
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"nano-transformer: d_model={cfg.d_model}, layers={cfg.n_layers}, "
        f"heads={cfg.n_heads}, kv_heads={cfg.n_kv_heads} (numpy CPU)"
    )
    fig.tight_layout()
    path = RESULTS_DIR / "charts.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_results_md(verify: dict, decode_rows, prefill_rows, cfg: TransformerConfig) -> Path:
    lines = []
    lines.append("# 01-nano-transformer: results")
    lines.append("")
    lines.append(
        f"Config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
        f"n_heads={cfg.n_heads}, n_kv_heads={cfg.n_kv_heads}, "
        f"vocab_size={cfg.vocab_size}, ffn_mult={cfg.ffn_mult}, "
        f"max_seq_len={cfg.max_seq_len}, rope_theta={cfg.rope_theta}. "
        f"Seed {SEED}; weights random-init (float32); CPU numpy."
    )
    lines.append("")
    lines.append("## Numerical verification")
    lines.append("")
    lines.append("| Check | Max abs diff | Status |")
    lines.append("|---|---|---|")
    for name, diff, tol in verify["rows"]:
        tol_str = "== 0 (bit-exact)" if tol == 0 else f"< {tol:g}"
        lines.append(f"| {name} | {diff:.3e} | pass ({tol_str}) |")
    lines.append("")
    lines.append("## Decode latency vs sequence length (median of %d reps)" % REPS)
    lines.append("")
    lines.append("| seq_len | with KV cache (ms/token) | recompute, no cache (ms/token) | speedup |")
    lines.append("|---|---|---|---|")
    for L, tc, tr in decode_rows:
        lines.append(
            f"| {L} | {tc*1e3:.3f} | {tr*1e3:.3f} | {tr/tc:.1f}x |"
        )
    lines.append("")
    lines.append("## Prefill latency vs prompt length (median of %d reps)" % REPS)
    lines.append("")
    lines.append("| prompt_len | prefill latency (ms) |")
    lines.append("|---|---|")
    for L, t in prefill_rows:
        lines.append(f"| {L} | {t*1e3:.3f} |")
    lines.append("")
    lines.append("## KV cache memory vs sequence length (analytic, fp16)")
    lines.append("")
    lines.append(
        "Formula: 2 (k+v) * n_layers * n_kv_heads * head_dim * seq_len * 2 bytes."
    )
    lines.append("")
    lines.append("| seq_len | KV cache bytes | KiB |")
    lines.append("|---|---|---|")
    for L in SEQ_LENS:
        b = kv_cache_bytes(cfg, L)
        lines.append(f"| {L} | {b:,} | {b/1024:.1f} |")
    lines.append("")
    lines.append("![benchmark charts](charts.png)")
    lines.append("")
    path = RESULTS_DIR / "results.md"
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(SEED)
    # max_seq_len=1024 so a decode step at position 512 (after a 512-token
    # prefill) stays inside the RoPE cache.
    cfg = TransformerConfig(max_seq_len=1024)
    model = TransformerModel(cfg, np.random.default_rng(SEED))

    print("=== Numerical verification (seed 7) ===")
    vrows = []
    d = verify_rmsnorm(rng)
    vrows.append(("RMSNorm vs naive loop", d, 1e-6))
    print(f"RMSNorm vs naive loop:            max abs diff = {d:.3e}")
    d = verify_rope(rng)
    vrows.append(("RoPE vs complex-exponential rotation", d, 1e-5))
    print(f"RoPE vs complex-exponential:      max abs diff = {d:.3e}")
    d = verify_gqa(rng)
    vrows.append(("GQA vs full-head (tied weights)", d, 0.0))
    print(f"GQA vs full-head tied weights:    max abs diff = {d:.3e}")
    d = verify_cache_exactness(model, rng)
    vrows.append(("KV-cache decode vs full forward", d, 1e-6))
    print(f"KV-cache decode vs full forward:  max abs diff = {d:.3e}")

    print("\n=== Benchmarks (median of %d reps, ms) ===" % REPS)
    decode_rows = bench_decode(model, rng)
    for L, tc, tr in decode_rows:
        print(
            f"seq {L:>3}: cache {tc*1e3:8.3f} ms/token | "
            f"recompute {tr*1e3:8.3f} ms/token | speedup {tr/tc:5.1f}x"
        )
    prefill_rows = bench_prefill(model, rng)
    for L, t in prefill_rows:
        print(f"prefill len {L:>3}: {t*1e3:8.3f} ms")

    chart_path = make_charts(decode_rows, cfg)
    md_path = write_results_md({"rows": vrows}, decode_rows, prefill_rows, cfg)
    print(f"\nWrote {chart_path}")
    print(f"Wrote {md_path}")

    L = 512
    tc = next(r[1] for r in decode_rows if r[0] == L)
    tr = next(r[2] for r in decode_rows if r[0] == L)
    print(
        f"\nHEADLINE seq={L}: decode {tc*1e3:.3f} ms/token with cache "
        f"vs {tr*1e3:.3f} ms/token recompute (speedup {tr/tc:.1f}x)"
    )


if __name__ == "__main__":
    main()
