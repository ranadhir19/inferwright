"""05-quant-lab experiments (seeded, CPU-only, numpy + matplotlib).

Pipeline
--------
1. Synthetic reconstruction benchmark: quantize outlier-structured weight
   matrices with each scheme; report relative-L2 and max-abs error.
2. Train the tiny char-level transformer LM once (seed 41), save weights,
   record baseline eval perplexity. Then evaluate every scheme on the frozen
   trained weights -- the whole eval sweep runs TWICE (weights reloaded from
   disk the second time) and the two passes must agree.
3. Per-layer sensitivity: quantize one scope at a time (INT4, group 32).
4. Pareto chart (bits/param vs perplexity), sensitivity bar chart, and a
   Markdown report with all tables.

Run:  python3 05-quant-lab/experiments/run_experiments.py   (from repo root)
"""

import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from llm_inference_engine.quant_lab import metrics, schemes, tiny_lm
from llm_inference_engine.quant_lab.eval import (
    evaluate,
    quantize_model,
    quantized_bits_per_param,
)

SEED = 41
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SCHEMES = [
    ("INT8 per-channel", schemes.quantize_int8_per_channel, {}),
    ("FP8 E4M3 (direct cast)", schemes.quantize_fp8_e4m3, {}),
    ("INT4 group-128", schemes.quantize_int4_per_group, {"group_size": 128}),
    ("INT4 group-32", schemes.quantize_int4_per_group, {"group_size": 32}),
    ("INT8 per-tensor", schemes.quantize_int8_per_tensor, {}),
]

N_TRAIN_STEPS = 600
TRAIN_BATCH = 16


# ----------------------------------------------------------------------------
# experiment 1: synthetic reconstruction with outlier structure
# ----------------------------------------------------------------------------

def outlier_weights(rng, shape):
    """Mimic real LLM weight distributions: small dense values + outliers.

    Base ~ N(0, 0.03); 0.2% of entries scaled x25 (rare large weights);
    ~1.5% of columns scaled x12 (outlier channels, as seen in real LLMs).
    """
    w = rng.normal(0.0, 0.03, shape).astype(np.float32)
    rare = rng.random(shape) < 0.002
    w[rare] *= 25.0
    n_cols = max(1, shape[1] * 3 // 200)
    cols = rng.choice(shape[1], size=n_cols, replace=False)
    w[:, cols] *= 12.0
    return w


def experiment_1(rng):
    print("\n=== Experiment 1: synthetic reconstruction benchmark ===", flush=True)
    rows = []
    for shape in [(1024, 1024), (4096, 1024)]:
        w = outlier_weights(rng, shape)
        for name, fn, kw in SCHEMES:
            wq, meta = fn(w, **kw)
            rep = metrics.reconstruction_report(w, wq)
            rows.append({
                "shape": f"{shape[0]}x{shape[1]}",
                "scheme": name,
                "rel_l2": rep["relative_l2"],
                "snr_db": rep["snr_db"],
                "max_abs": rep["max_abs"],
                "bits": schemes.effective_bits_per_param(
                    meta, w.size),
            })
            print(f"  {shape[0]}x{shape[1]} {name:22s} "
                  f"relL2={rep['relative_l2']:.4f} maxAbs={rep['max_abs']:.4f}",
                  flush=True)
    return rows


# ----------------------------------------------------------------------------
# experiment 2: train once, evaluate twice
# ----------------------------------------------------------------------------

def train_model():
    print("\n=== Experiment 2: training tiny LM (seed 41) ===", flush=True)
    text = tiny_lm.build_corpus(seed=SEED, target_chars=12000)
    ids, c2i, vocab = tiny_lm.encode_corpus(text)
    print(f"  corpus: {len(text)} chars, vocab={len(vocab)} "
          f"({''.join(vocab)})", flush=True)
    cfg = tiny_lm.TinyLMConfig(vocab_size=len(vocab))
    split = int(0.9 * len(ids))
    train_ids, eval_ids = ids[:split], ids[split:]
    print(f"  config: d_model={cfg.d_model} layers={cfg.n_layers} "
          f"heads={cfg.n_heads} d_ff={cfg.d_ff} ctx={cfg.seq_len} "
          f"params={tiny_lm.count_params(tiny_lm.init_params(np.random.default_rng(0), cfg))}",
          flush=True)
    t0 = time.time()
    params, history = tiny_lm.train(
        cfg, train_ids, n_steps=N_TRAIN_STEPS, batch_size=TRAIN_BATCH,
        lr=3e-3, seed=SEED, log_every=100)
    dt = time.time() - t0
    print(f"  training done: {N_TRAIN_STEPS} steps in {dt:.1f}s "
          f"({dt / N_TRAIN_STEPS:.3f}s/step), final loss={history[-1]:.4f}",
          flush=True)
    weights_path = RESULTS / "tiny_lm_weights.npz"
    np.savez(weights_path, **params)
    print(f"  weights saved to {weights_path}", flush=True)
    return params, cfg, eval_ids, history, dt, len(vocab)


def eval_sweep(params, cfg, eval_ids, label):
    """One full eval pass: baseline + every scheme. Returns list of rows."""
    print(f"\n--- eval pass {label} ---", flush=True)
    rows = []
    base = evaluate(params, cfg, eval_ids)
    rows.append({"scheme": "FP32 baseline", "perplexity": base["perplexity"],
                 "bits_per_param": 32.0})
    print(f"  {'FP32 baseline':22s} ppl={base['perplexity']:.4f}", flush=True)
    for name, fn, kw in SCHEMES:
        t0 = time.time()
        qparams, metas = quantize_model(params, fn, scope="all", **kw)
        res = evaluate(qparams, cfg, eval_ids)
        bpp = quantized_bits_per_param(qparams, metas)
        rows.append({"scheme": name, "perplexity": res["perplexity"],
                     "bits_per_param": bpp})
        print(f"  {name:22s} ppl={res['perplexity']:.4f} "
              f"bits/param={bpp:.3f} ({time.time()-t0:.1f}s)", flush=True)
    return rows


def check_determinism(rows1, rows2):
    assert len(rows1) == len(rows2)
    diffs = [abs(a["perplexity"] - b["perplexity"])
             for a, b in zip(rows1, rows2)]
    assert [a["scheme"] for a in rows1] == [b["scheme"] for b in rows2]
    ok = all(d == 0.0 for d in diffs)
    print(f"\n  determinism check (pass1 vs pass2): "
          f"{'IDENTICAL' if ok else 'MISMATCH! max diff=' + str(max(diffs))}",
          flush=True)
    if not ok:
        raise RuntimeError("eval passes disagree")
    return ok


# ----------------------------------------------------------------------------
# experiment 3: per-layer sensitivity
# ----------------------------------------------------------------------------

def experiment_3(params, cfg, eval_ids, baseline_ppl):
    print("\n=== Experiment 3: per-layer sensitivity (INT4 group-32) ===",
          flush=True)
    rows = []
    for scope in ["l0", "l1", "head"]:
        qparams, _ = quantize_model(
            params, schemes.quantize_int4_per_group, scope=scope,
            group_size=32)
        res = evaluate(qparams, cfg, eval_ids)
        delta = res["perplexity"] - baseline_ppl
        rows.append({"scope": scope, "perplexity": res["perplexity"],
                     "delta": delta})
        print(f"  quantize {scope:6s} only: ppl={res['perplexity']:.4f} "
              f"(delta {delta:+.4f})", flush=True)
    return rows


# ----------------------------------------------------------------------------
# charts + report
# ----------------------------------------------------------------------------

def make_charts(eval_rows, sens_rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Pareto frontier: bits/param vs perplexity (log y).
    pts = [(r["bits_per_param"], r["perplexity"], r["scheme"])
           for r in eval_rows]
    pts_sorted = sorted(pts, key=lambda p: p[0])
    frontier, best = [], float("inf")
    for bpp, ppl, name in pts_sorted:
        if ppl < best:
            best = ppl
            frontier.append((bpp, ppl))
    for bpp, ppl, name in pts:
        ax1.scatter([bpp], [ppl], s=60, zorder=3)
        ax1.annotate(name, (bpp, ppl), textcoords="offset points",
                     xytext=(6, 4), fontsize=8)
    fx = [p[0] for p in frontier]
    fy = [p[1] for p in frontier]
    ax1.plot(fx, fy, "--", color="gray", alpha=0.7, label="Pareto frontier")
    ax1.set_xlabel("bits / param (quantized weights, fp16 scales)")
    ax1.set_ylabel("eval perplexity (log scale)")
    ax1.set_yscale("log")
    ax1.set_title("(a) Memory vs quality: quantization Pareto frontier")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # (b) Per-layer sensitivity.
    scopes = [r["scope"] for r in sens_rows]
    deltas = [r["delta"] for r in sens_rows]
    colors = ["tab:red" if d == max(deltas) else "tab:blue" for d in deltas]
    ax2.bar(scopes, deltas, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("quantized scope (INT4 group-32, rest fp32)")
    ax2.set_ylabel("perplexity increase vs fp32 baseline")
    ax2.set_title("(b) Per-layer sensitivity to INT4 quantization")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("05-quant-lab: weight quantization workbench (tiny char-LM, "
                 "seed 41)", fontsize=11)
    fig.tight_layout()
    out = RESULTS / "charts.png"
    fig.savefig(out, dpi=150)
    print(f"\n  charts saved to {out}", flush=True)


def write_report(exp1_rows, eval_rows, sens_rows, train_info, history):
    L = []
    A = L.append
    A("# 05-quant-lab: weight quantization workbench -- results\n")
    A("Seeded end-to-end (`np.random.default_rng(41)`); numpy + matplotlib "
      "only, CPU.\n")
    A("## Setup\n")
    A(f"* Tiny char-level transformer: d_model=64, 2 layers, 4 heads, "
      f"d_ff=128, context 64, vocab {train_info['vocab_size']} chars, "
      f"{train_info['n_params']} params.")
    A(f"* Corpus: {train_info['corpus_chars']} chars of generated repetitive "
      f"text (90% train / 10% eval).")
    A(f"* Training: Adam lr=3e-3, batch 16, {train_info['n_steps']} steps, "
      f"{train_info['train_time_s']:.0f}s CPU; final train loss "
      f"{history[-1]:.4f}.")
    A("* Quantization scope: all Linear weight matrices "
      "(`wq/wk/wv/wo/w1/w2/head_w`); embeddings, norms and biases stay fp32.")
    A("* Bits/param counts quantized weights only, assuming fp16 scales "
      "(standard GPTQ-style accounting).\n")
    A("## Experiment 1: synthetic reconstruction (outlier-structured weights)\n")
    A("Weights ~ N(0, 0.03) with 0.2% of entries x25 and ~1.5% of columns x12 "
      "(outlier channels).\n")
    A("| shape | scheme | rel. L2 error | max-abs error | bits/param |")
    A("|---|---|---|---|---|")
    for r in exp1_rows:
        A(f"| {r['shape']} | {r['scheme']} | {r['rel_l2']:.4f} | "
          f"{r['max_abs']:.4f} | {r['bits']:.3f} |")
    A("\nReading: per-channel / per-group schemes isolate outlier channels "
      "into their own scale, so one hot channel cannot blow up the error of "
      "the whole tensor -- per-tensor INT8 pays the full outlier tax on both "
      "metrics.")
    A("\n## Experiment 2: trained-model perplexity (eval run twice)\n")
    A("| scheme | eval perplexity | bits/param |")
    A("|---|---|---|")
    for r in eval_rows:
        A(f"| {r['scheme']} | {r['perplexity']:.4f} | {r['bits_per_param']:.3f} |")
    A("\nThe eval sweep was executed twice (training once; weights reloaded "
      "from disk for pass 2). Both passes produced bit-identical "
      "perplexities -- the numbers above are deterministic, not noise.")
    A("\n## Experiment 3: per-layer sensitivity (INT4 group-32, one scope)\n")
    A("| quantized scope | eval perplexity | delta vs baseline |")
    A("|---|---|---|")
    for r in sens_rows:
        A(f"| {r['scope']} | {r['perplexity']:.4f} | {r['delta']:+.4f} |")
    A("\n## Scale limits\n")
    A(f"Tiny model ({train_info['n_params']} params), character-level, CPU, "
      f"synthetic repetitive text. The *methodology* -- granularity "
      f"ablations, Pareto analysis, sensitivity sweeps -- transfers to real "
      f"LLMs; the absolute perplexities do not.")
    out = RESULTS / "results.md"
    out.write_text("\n".join(L) + "\n")
    print(f"  report saved to {out}", flush=True)


def main():
    rng = np.random.default_rng(SEED)
    exp1_rows = experiment_1(rng)

    params, cfg, eval_ids, history, train_s, vocab_size = train_model()
    n_params = tiny_lm.count_params(params)

    pass1 = eval_sweep(params, cfg, eval_ids, "1 (fresh weights)")
    # Reload from disk and evaluate again: training once, eval twice.
    params2 = dict(np.load(RESULTS / "tiny_lm_weights.npz"))
    pass2 = eval_sweep(params2, cfg, eval_ids, "2 (reloaded weights)")
    check_determinism(pass1, pass2)

    baseline_ppl = pass1[0]["perplexity"]
    sens_rows = experiment_3(params, cfg, eval_ids, baseline_ppl)

    make_charts(pass1, sens_rows)
    write_report(exp1_rows, pass1, sens_rows,
                 {"vocab_size": vocab_size, "n_params": n_params,
                  "corpus_chars": len(tiny_lm.build_corpus(seed=SEED)),
                  "n_steps": N_TRAIN_STEPS, "train_time_s": train_s},
                 history)
    print("\nALL EXPERIMENTS DONE", flush=True)


if __name__ == "__main__":
    main()
