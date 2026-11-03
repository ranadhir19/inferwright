# 03 — Speculative Decoding Simulator

## Problem

Vanilla autoregressive decoding is latency-bound: each token requires a full
forward pass of the large target model, and tokens are generated strictly one
at a time. This makes the per-token latency equal to the cost of the most
expensive model you own, with no way to amortize it.

## Current gap

Speculative decoding (draft a few tokens with a cheap model, verify them in
one target pass) is known to speed up inference, but most teams deploy it
with hand-picked draft lengths and no quantitative model of when it helps,
when it is neutral, and when it actively *hurts*. Without a model, you
cannot pick the draft length for your draft model quality, nor can you
predict whether a given draft/target pair is worth the engineering effort.

## Solution

A numpy-only (CPU) simulator for the draft + target speculative sampling
loop, built around the closed-form analytical model:

- Expected accepted tokens per target forward pass:
  `E(k, alpha) = (1 - alpha^(k+1)) / (1 - alpha)`, where `k` is the draft
  length and `alpha` the per-token acceptance probability.
- Expected wall-clock speedup vs vanilla AR: `S = E / (1 + k*c)`, where
  `c = (draft cost per token) / (target cost per token)`.
- An optimal-k selector that maximizes `S` over candidate draft lengths
  (`k = 0`, i.e. plain AR, is always allowed, so it never recommends a
  slowdown).
- A Monte Carlo token-level simulator of the geometric acceptance process
  that validates the formula empirically, plus wall-clock tokens/sec
  estimates from a target forward latency.

Everything is seeded (`np.random.default_rng(21)`) and runs twice with
bit-identical headlines. Module: `src/llm_inference_engine/spec_decode/`
(`expected_speedup`, `optimal_k`, `simulate`).

## Results

Formula validation: Monte Carlo (100k trials/cell, 120 cells across
k=1..8, alpha in {0.3, 0.5, 0.7, 0.85, 0.95}, c in {0.05, 0.1, 0.2}) matches
the analytical formula with **max absolute deviation 0.01786 tokens** (max
speedup deviation 0.01051).

Speedup sweep highlights (k=1..8):

| alpha | c=0.05 | c=0.1 | c=0.2 |
|-------|--------|-------|-------|
| 0.3   | k=2 (1.26x) | k=1 (1.18x) | k=1 (1.08x) |
| 0.5   | k=3 (1.63x) | k=2 (1.46x) | k=1 (1.25x) |
| 0.7   | k=6 (2.35x) | k=4 (1.98x) | k=3 (1.58x) |
| 0.85  | k=8 (3.66x) | k=7 (2.85x) | k=5 (2.08x) |
| 0.95  | k=8 (5.28x) | k=8 (4.11x) | k=8 (2.84x) |

Best cell in the sweep: **5.282x speedup at k=8, alpha=0.95, c=0.05**.

End-to-end scenario (target forward 40 ms, draft cost ratio 0.1,
alpha=0.7, optimal k=4): speculative decoding achieves **49.5 tok/s vs
25.0 tok/s vanilla AR — a 1.979x speedup**. The Monte Carlo throughput
agrees with the analytical 1.981x.

Failure case (alpha=0.3, k=8, c=0.1): **0.791x speedup — a slowdown**
(19.8 tok/s vs 25.0 tok/s vanilla). The draft compute is pure overhead
when most drafts are rejected.

Full tables and charts: `results/results.md`, `results/charts.png`.

## Insights

- **Speculation is a bet on alpha.** The acceptance probability dominates
  everything: at alpha=0.95 you get 5.3x; at alpha=0.3 you are fighting for
  1.2x.
- **Large k with low alpha HURTS.** The denominator `1 + k*c` grows linearly
  with draft length while accepted tokens saturate at ~1/(1-alpha). Drafting
  8 tokens at alpha=0.3 costs 1.8x the compute per cycle for only ~1.42
  accepted tokens: net slowdown of ~21%.
- **Optimal draft length grows with draft quality and shrinks with draft
  cost.** Cheap, accurate drafts justify k=7..8; mediocre or expensive
  drafts justify k=1..2 — and below some alpha threshold the optimum is
  k=0, i.e. do not speculate at all.
- The closed form is exact for the geometric acceptance process (Monte
  Carlo agrees to within 0.018 tokens), so draft-length tuning can be done
  analytically from a measured alpha — no GPU required.
