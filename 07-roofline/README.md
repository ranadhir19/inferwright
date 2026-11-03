# 07 — Roofline performance model for LLM inference ops

## Problem

LLM serving has two latency regimes with opposite bottlenecks. Decode (one
token at a time) feels slow because each step re-reads the entire weight
matrix for a handful of FLOPs; prefill (the prompt pass) is a different
animal. Without a quantitative model, "make it faster" devolves into
guessing which kernel to optimize.

## Current gap

Most roofline treatments of transformers stop at a single GEMM or quote
aggregate TFLOPS. This project builds a per-op roofline for a full decoder
layer — QKV projection, QK^T, softmax, AV, output projection, SwiGLU FFN —
with explicit byte accounting (weights, activations, KV-cache reads), so
each op gets an arithmetic intensity, a bound label, a % of peak, and an
estimated time. Fusion is modeled as a transform on the byte traffic, not
hand-waving.

## Solution

`src/llm_inference_engine/roofline/` (numpy, CPU-only, analytical):

- `hardware.py` — `Hardware` dataclass (peak FLOPS per dtype, memory
  bandwidth). A100 (312 dense FP16 TFLOPS, 2039 GB/s), H100 (989 dense
  FP16 TFLOPS, 3350 GB/s) from published specs; an illustrative CPU
  reference.
- `ops.py` — FLOP/byte accounting per op for a layer config
  (d_model, n_heads, ffn_mult, batch, seq, dtype). Decode reads the KV
  cache for one query token; prefill materializes the SxS score matrix.
- `model.py` — roofline classification
  (attainable = min(peak, intensity * bw), % of peak, memory/compute-bound
  vs the ridge point) plus two fusion transforms:
  - FlashAttention-style: QK^T + softmax + AV fused; the score matrix is
    never materialized, so bytes drop from O(B*H*S^2) to O(B*H*S).
  - Fused dequant+GEMM: weights stored quantized (e.g. INT8) and
    dequantized on the fly; weight-read bytes scale by the byte ratio.

`experiments/run_experiments.py` is deterministic (analytical; two runs
produce identical SHA-256 fingerprints of the outputs).

## Results

Shapes: d_model=4096, n_heads=32, ffn_mult=2.0 (LLaMA-7B-ish), fp16.
Full tables in `results/results.md`; charts in `results/charts.png`.

**Decode (B=1, S=2048), A100:** all 7/7 ops memory-bound at ~1.0 FLOP/byte,
each at ~0.65% of peak. Estimated step time 181.3 us, dominated by
`ffn_gate_up` (65.9 us, 36.3% of step). On H100: same 7/7 memory-bound at
~0.34% of peak, 110.4 us/step.

**Prefill (B=8, S=2048), A100:** 3/7 ops memory-bound — `attn_qk` and
`attn_av` at 113.8 FLOP/byte (74.4% of peak) and `softmax` at 1.25
FLOP/byte (0.82% of peak) — while all four projection GEMMs are
compute-bound at 100% of peak. Step time 22.10 ms, dominated by
`ffn_gate_up` (7.05 ms, 31.9%). On H100: 8.28 ms/step.

**Fusion estimates (A100):**

- FlashAttention-style, decode: estimated step speedup 1.001x — the SxS
  score matrix is 1x2048 per head, so there is almost no traffic to kill.
- FlashAttention-style, prefill: attention intensity 112.7 -> 1034
  FLOP/byte (memory-bound -> compute-bound); attention time 4.48 ms ->
  1.78 ms; estimated step speedup 1.139x.
- Fused dequant+GEMM (INT8 weights), decode: GEMM intensity 1.0 -> 2.0
  FLOP/byte (still memory-bound); estimated step speedup 1.831x.

## Insights

- Decode is memory-bound — bandwidth is the wall. At ~1 FLOP/byte against
  a 153 FLOP/byte ridge point, no amount of extra FLOPS helps; only less
  traffic (quantized weights, larger batches) or more bandwidth does.
- Prefill is compute-bound for the projections but the attention ops stay
  memory-bound because of the materialized SxS matrix — which is exactly
  what FlashAttention removes.
- Fusion moves the ceiling, not the wall: FlashAttention shifts attention
  from the bandwidth ceiling to the compute ceiling (big win in prefill,
  ~nothing in decode), while weight quantization raises effective
  intensity against the unchanged bandwidth wall (the real lever for
  decode latency: 1.83x estimated step speedup).

## Layout

- `src/llm_inference_engine/roofline/` — the model
- `experiments/run_experiments.py` — all four experiments
- `results/results.md`, `results/charts.png` — generated outputs
- `tests/test_smoke.py` — sanity tests
