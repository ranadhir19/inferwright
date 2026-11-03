# 06-moe-router: Mixture-of-Experts expert-routing simulator

## Problem

A mixture-of-experts (MoE) layer promises the capacity of a huge model at
the compute cost of a small one: each token is processed by only `top_k`
of `E` experts. In practice the router must *dispatch* tokens to experts
that live on different devices, and each expert has a fixed per-step
capacity. When too many tokens pick the same popular expert, the overflow
is silently **dropped** — those tokens skip the expert entirely. Dropping
is the hidden tax of MoE: it never appears in the parameter count, but it
directly reduces the tokens that receive full expert computation.

## Current gap

Most MoE discussions focus on gating (top-k softmax) and on the parameter
savings. What is rarely quantified in one place is the three-way tradeoff
that actually governs MoE serving:

1. **Capacity factor (CF)** sets `expert_capacity = ceil(CF * n_tokens /
   n_experts)`. CF < 1 under-provisions; CF > 1 wastes memory and compute
   on padding.
2. **Load balancing** (auxiliary loss) decides whether capacity is spread
   evenly or piled onto a few popular experts.
3. **Expert parallelism** replaces the drop problem with a communication
   problem: all-to-all dispatch/combine volume that does not shrink as you
   add devices.

## Solution

A numpy-only simulator (`src/llm_inference_engine/moe_router/`) with three
pieces:

- `gating.py` — `topk_gate` (softmax + top-k + renormalize), `noisy_gate`
  (logit noise to simulate imperfect routing), `balanced_gate` (models the
  steady-state effect of an aux load-balancing loss as reduced logit skew).
- `router.py` — `route` dispatches tokens in order, drops assignments beyond
  `expert_capacity`, and reports drop rate, per-expert counts, imbalance
  ratio (max/mean), and processed tokens.
- `parallel.py` — `shard_experts`, `alltoall_bytes`
  (`n_tokens * top_k * d_model * bytes_per_elem * 2`, dispatch + combine),
  `device_load` (per-device compute slots and comm share).

Experiments are seeded (`np.random.default_rng(51)`); two consecutive runs
produce identical headlines, charts, and `results/results.md`.

## Results

Setup: 4096 tokens, 8 experts, skewed expert popularity
(0.30 / 0.20 / 0.15 / 0.10 / 0.08 / 0.07 / 0.05 / 0.05).

**Experiment 1 — drop rate vs capacity factor (skewed gating):**

| CF | top-1 drop % | top-1 processed | top-2 drop % | top-2 processed |
|----|--------------|-----------------|--------------|-----------------|
| 0.50 | 77.5 | 921 / 4096 | 83.2 | 1044 / 4096 |
| 1.00 | 61.0 | 1598 / 4096 | 71.8 | 1665 / 4096 |
| 1.50 | 48.5 | 2110 / 4096 | 62.4 | 2102 / 4096 |
| 2.00 | 38.1 | 2536 / 4096 | 53.0 | 2556 / 4096 |

Even at CF=2.0 (double the "fair share" capacity per expert), skewed
routing still drops 38% of top-1 assignments — the most popular expert
alone attracts ~2584 tokens against a capacity of 1024.

**Experiment 2 — load-balancing loss effect at CF=1.0:**

| top-k | gating | drop % | imbalance max/mean | processed |
|-------|--------|--------|--------------------|-----------|
| 1 | skewed | 61.0 | 2.56 | 1599 |
| 1 | balanced | 9.3 | 1.10 | 3714 |
| 2 | skewed | 71.5 | 1.75 | 1693 |
| 2 | balanced | 50.0 | 1.00 | 2251 |

An idealized aux load-balancing loss cuts top-1 drops from 61.0% to 9.3%
and more than doubles processed tokens (1599 -> 3714). Note the top-2
balanced case: perfectly even routing (imbalance 1.00) still drops 50%
because each token consumes *two* capacity slots — top-k multiplies the
capacity requirement.

**Experiment 3 — expert parallelism (E=16, d_model=4096, fp16, top-2):**

| devices D | experts/device | all-to-all total | comm/device | token slots/device |
|-----------|----------------|------------------|-------------|--------------------|
| 1 | 16 | 128.0 MiB | 128.0 MiB | 8192 |
| 2 | 8 | 128.0 MiB | 64.0 MiB | 4096 |
| 4 | 4 | 128.0 MiB | 32.0 MiB | 2048 |
| 8 | 2 | 128.0 MiB | 16.0 MiB | 1024 |

Total all-to-all volume is independent of D: every token->expert vector is
sent twice (dispatch + combine) regardless of sharding. Per-device compute
shrinks as 1/D while total communication stays at 128.0 MiB/step, so the
comm/compute ratio worsens with more devices.

## Insights

1. **Dropping is the hidden tax of MoE.** With realistic skewed expert
   popularity, the default CF=1.0 throws away 61% of top-1 expert
   assignments. The parameter-count story ("only 2 of 8 experts active")
   says nothing about how many tokens actually got expert compute.
2. **Capacity factor trades memory/compute for quality, with diminishing
   returns.** Raising CF from 1.0 to 2.0 costs 2x the expert activation
   memory/compute but only cuts top-1 drops from 61% to 38% under skew.
   Throwing capacity at a routing problem is expensive and incomplete.
3. **Load balancing beats raw capacity.** The aux-loss-idealized router
   achieves at CF=1.0 (9.3% drops) what skewed gating cannot reach even at
   CF=2.0 (38.1% drops) — and it does so without any extra memory. This is
   why production MoE training treats the balancing loss as first-class.
4. **top-k multiplies the capacity bill.** Balanced top-2 routing is
   perfectly even yet still drops 50% at CF=1.0, because each token books
   two expert slots. Capacity planning must be done per *assignment*, not
   per token.
5. **Expert parallelism converts the drop problem into a comm problem.**
   Sharding 16 experts over 8 devices leaves only 1024 compute slots per
   device but still moves 128 MiB of all-to-all traffic per step — the
   network, not the experts, becomes the scaling bottleneck.
