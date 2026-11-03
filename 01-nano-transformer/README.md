# 01-nano-transformer

A LLaMA-style decoder-only transformer implemented from scratch in numpy
(CPU only): RMSNorm, RoPE rotary embeddings, grouped-query attention (GQA),
SwiGLU feed-forward blocks, and an incremental KV cache with
`forward(tokens)` full-pass and `step(token, cache)` single-token decode,
plus greedy and temperature/top-k sampling in `generate()`.

Config used throughout: d_model=128, n_layers=4, n_heads=8, n_kv_heads=2,
vocab_size=512, ffn_mult=2, rope_theta=10000. All weights are
**random-initialized** (seed 7); nothing here is trained. This project
demonstrates inference mechanics and measures them — it says nothing about
model quality.

## Problem

Every production LLM serving stack (vLLM, TensorRT-LLM, llama.cpp) rests on a
small set of inference primitives: normalized residual blocks, rotary position
embeddings, grouped-query attention, gated FFNs, and incremental decoding
against a KV cache. These are usually hidden inside framework kernels, so it
is hard to verify what each piece computes, what it costs, and why the KV
cache matters so much.

## Current gap

Most "build a transformer from scratch" material stops at a training loop for
a toy model. It rarely (a) verifies each inference primitive against an
independent reference implementation with assert-level tolerances,
(b) proves that incremental cached decoding is numerically identical to a
full forward pass, or (c) measures the actual latency and memory scaling of
KV-cache vs recompute decoding on the same code.

## Solution

`src/llm_inference_engine/transformer/model.py` implements the full
inference path in ~350 lines of numpy with no ML framework:

- `TransformerConfig`, `RMSNorm`, `build_rope_cache` / `apply_rope`
  (LLaMA-style halves pairing), `repeat_kv` for GQA, SwiGLU FFN.
- `KVCache`: preallocated per-layer k/v buffers with `append`, `advance`,
  `get`, and an analytic `bytes_for` footprint helper.
- `TransformerModel.forward(tokens)` (causally masked full pass),
  `step(token, cache)` (single-token incremental decode attending over the
  cached prefix), `prefill`, and `generate` (greedy or temperature/top-k).
- `experiments/run_experiments.py` (seeded, `np.random.default_rng(7)`):
  four assert-level numerical verifications, latency benchmarks (median of
  15 reps via `time.perf_counter`), analytic KV-memory accounting, and
  chart + markdown report generation into `results/`.

## Results

Numerical verification (seed 7; identical across repeated runs):

| Check | Max abs diff |
|---|---|
| RMSNorm vs naive row-loop implementation | 4.768e-07 |
| RoPE vs direct complex-exponential rotation | 2.384e-07 |
| GQA attention vs full-head attention, tied weights | 0.0 (bit-exact) |
| Incremental KV-cache decode vs full-sequence forward | 0.0 (bit-exact) |

Benchmarks (numpy CPU, 2-core VM; medians of 15 reps; the experiment script
was run twice — verification diffs were identical both times, timings showed
ordinary run-to-run jitter of up to ~2x on this shared machine):

| seq_len | decode with KV cache (ms/token) | decode recompute, no cache (ms/token) | speedup |
|---|---|---|---|
| 32 | 1.844 | 21.243 | 11.5x |
| 64 | 3.614 | 34.564 | 9.6x |
| 128 | 2.399 | 165.936 | 69.2x |
| 256 | 4.140 | 485.117 | 117.2x |
| 512 | 4.718 | 645.682 | 136.8x |

Prefill (full forward pass) latency: 13.5 ms at 32 tokens, 79.5 ms at 128,
557.3 ms at 512. KV-cache footprint (analytic, fp16):
`2 * n_layers * n_kv_heads * head_dim * seq_len * 2 bytes` = 16 KiB at
seq 32, 64 KiB at 128, 256 KiB at 512 — linear in sequence length.

Charts: `results/charts.png` (decode latency vs seq_len, both curves; KV
memory vs seq_len). Full tables: `results/results.md`. Smoke tests
(`tests/test_smoke.py`): 7 passed.

## Insights

- **The KV cache is the whole game for decode.** Cached decode stays nearly
  flat (~1-5 ms/token) while full-context recompute grows steeply with
  sequence length (~21 ms at 32 tokens, ~646 ms at 512) — a 137x gap at
  seq 512 on identical hardware and code. Prefill is the price of admission;
  decode is where the cache pays off.
- **Exactness is achievable, not approximate.** Cached incremental decode
  reproduces full-forward logits bit-for-bit (max diff 0.0), because each
  cached key is rotated by RoPE at its own absolute position and attention
  is evaluated over the identical prefix in the identical order. There is no
  mathematical reason for "close enough" here.
- **GQA is pure bookkeeping.** With tied weights, grouped-query attention is
  bit-identical to full multi-head attention (diff 0.0) — repeating KV heads
  is a memory/compute tradeoff, not an approximation.
- **KV memory is linear and plannable.** 256 KiB at seq 512 for this nano
  config (fp16); the formula has no surprises, which is exactly why serving
  systems can budget it per request.
- **Benchmark on a quiet machine if you want tight numbers.** Medians over
  15 reps still jittered ~2x run-to-run on a shared 2-core VM; the scaling
  shapes and orders of magnitude were stable, the last digits were not.
