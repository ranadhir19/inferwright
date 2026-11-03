# 08 — sched-lab: LLM request-scheduling simulator

An advanced, numpy-only (CPU) discrete-event simulator comparing four LLM
serving schedulers — static FCFS batching, continuous iteration-level
batching, chunked prefill, and priority preemption — on latency (TTFT,
TPOT), throughput, and GPU utilization.

## Problem

How a serving engine schedules requests across GPU iterations decides the
latency every user feels. A single 4k-token prompt prefetched in one
iteration stalls every in-flight decode; a fixed batch that runs to
completion head-of-line-blocks everything behind it; and under KV-cache
pressure, someone has to decide which request loses its slot. These
tradeoffs are usually argued from intuition. sched-lab measures them.

## Current gap

Most scheduling discussions compare policies qualitatively ("continuous
batching is better") or on production systems where the workload, the
hardware, and the scheduler change at once. There was no small,
reproducible harness that isolates the *scheduling policy* — same seeded
workload, same analytical GPU cost model, same KV budget — and reports
TTFT/TBOT tails, throughput, and utilization per policy.

## Solution

`src/llm_inference_engine/sched_lab/sim.py` implements an
iteration-stepped discrete-event simulator behind one `Scheduler`
interface:

- **static-fcfs** — fixed FCFS batches run to completion; if the queue head
  does not fit the KV budget, nothing behind it is admitted (genuine
  head-of-line blocking).
- **continuous** — vLLM-style iteration-level batching: waiting requests
  join freed KV slots every iteration; a full prompt prefetches in one
  iteration.
- **chunked-prefill** — Sarathi-style: each prompt is processed in
  512-token chunks interleaved with decode steps, bounding per-iteration
  latency.
- **priority-preempt** — short requests are admitted ahead of long ones and
  preempt them under KV pressure. A preempted request loses its KV cache
  and must recompute `prompt_len + generated_so_far` tokens on
  rescheduling — the recompute cost is charged honestly as extra batch
  tokens in later iterations, and counted in the metrics.

GPU model: each iteration costs `t = T0 + K * batch_tokens`
(`T0 = 2.0 ms`, `K = 0.012 ms/token`). KV cache is a hard 16384-token
budget; each admitted request reserves `prompt_len + gen_len` up front.
Workloads are Poisson arrivals with a mixed shape (chat prompts 32–512
tokens, document prompts 1024–4096). Everything is seeded
(`np.random.default_rng(61)`); three consecutive runs produce
byte-identical headlines, `results.md`, and `charts.png`.

## Results

All numbers below are simulator output (seed 61), not invented.

**Experiment 1 — policy x load, 1500 requests per cell.**
At high load (6 req/s), continuous batching delivers TTFT mean 15.0 ms /
p99 57.5 ms versus static-fcfs 844.6 ms / p99 2437.1 ms — a 56x mean and
42x p99 reduction from iteration-level scheduling alone. Throughput is
identical across policies (1860.8 gen tok/s): every policy sustains the
offered load; scheduling moves latency, not capacity, at this scale.
Chunked prefill tracks continuous within ~20% on TTFT p99 (69.1 vs
57.5 ms). Priority preemption fires only 3 times at this load and matches
continuous (p99 57.1 ms).

**Experiment 2 — burst stress: 10 x 4096-token prompts at t=60 s amid pure
chat traffic (3 req/s).** Chunked prefill holds chat TTFT p99 at 12.0 ms
— the same as continuous batching — while bounding worst-case iteration
latency to 25.7 ms versus 149.5 ms for continuous (82.8% lower), which is
what protects in-flight decodes from TBT spikes. Static-fcfs collapses:
chat TTFT p99 1032.5 ms, 98.8% worse than chunked prefill, because one
batch's giant prefills head-of-line-block everything behind them.

**Experiment 3 — preemption cost (6 req/s, 30% long requests, burst at
t=30 s).** Priority preemption performs 14 preemptions costing 43,598
recompute tokens — 4.88% extra compute — in exchange for cutting
short-request TTFT p99 from 362.4 ms to 44.0 ms (87.9% lower) versus
continuous batching. Overall TTFT p99 is unchanged (2078.9 vs 2078.6 ms):
preemption redistributes latency from short requests to long ones; it does
not create capacity.

## Insights

- **Iteration-level admission is the dominant win.** Moving from
  run-to-completion batches to per-iteration admission cuts high-load TTFT
  p99 by 42x. Everything else is second-order.
- **Chunking bounds the worst iteration, not the average wait.** A
  synchronized long-prompt burst creates one giant iteration under
  continuous batching (concentrated pain: few requests hurt badly) versus
  many bounded iterations under chunking (spread pain: more requests hurt
  mildly). At p99 the two can tie; the win shows up in worst-case
  iteration latency (82.8% lower) and in TBT stability for decodes already
  in flight.
- **Preemption is a latency redistribution, priced in recompute.** 4.88%
  extra compute bought an 87.9% short-request p99 improvement here — a good
  trade when short requests are the product, but the preempted longs pay
  the full recompute of prompt plus generated tokens.
- **KV pressure, not iteration cost, is the cliff.** As long as every
  policy sustains the offered token rate, throughput is policy-independent;
  latency diverges only when queueing starts. Size the KV budget for the
  burst, not the average.

## Honest note on the model

The iteration-cost model is analytical (`t = 2.0 + 0.012 * batch_tokens`
ms), calibrated to the *shape* of real serving curves — fixed launch
overhead plus roughly linear token scaling — not measured on a GPU.
Absolute millisecond values are illustrative; the relative policy
differences are the finding. The simulator also assumes `gen_len` is known
at admission for KV reservation (real schedulers reserve incrementally)
and models preemption as full recompute rather than KV swap to CPU.

## Reproduce

```bash
python3 08-sched-lab/experiments/run_experiments.py  # ~10 s, seed 61
python3 08-sched-lab/tests/test_smoke.py
```

Outputs: `08-sched-lab/results/charts.png`, `08-sched-lab/results/results.md`.
Run twice; the headlines are identical.
