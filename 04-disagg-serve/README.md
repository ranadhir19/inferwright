# 04 — Disaggregated Prefill/Decode Serving Simulator

## Problem

In a standard (colocated) LLM serving setup, the same workers handle both
prefill (processing the prompt, compute-bound) and decode (generating tokens
one by one, memory-bandwidth-bound). A new request's long prefill is batched
with other requests' decode steps, so one 2048-token prompt stalls every
in-flight token stream on that worker, and new requests queue behind whatever
the worker is already doing. The question: how much do TTFT (time to first
token) and TBT (time between tokens) improve if prefill and decode run on
separate, specialized pools, and what does the KV-cache transfer between the
pools cost?

## Current gap

Most treatments of disaggregated serving (e.g. Splitwise-style architectures)
argue from hardware intuition or production traces. There is no small,
reproducible model that isolates the *scheduling and interference* mechanics:
head-of-line blocking of prefills behind decodes, KV-cache capacity limits,
and the explicit transfer delay of moving the prompt KV cache over a finite
interconnect. This project builds exactly that model in numpy on CPU.

## Solution

A discrete-event simulator (`src/llm_inference_engine/disagg_serve/sim.py`):

- **Colocated:** 4 identical workers, continuous batching. One iteration costs
  `t = 2.0 ms + 0.009 ms * batch_tokens`, subject to an 8192-token KV cap per
  worker. A request's full prompt prefills in a single iteration alongside
  ongoing decodes, so long prefills directly inflate everyone else's token
  latency (interference), and arrivals queue behind busy workers.
- **Disaggregated:** 1 compute-optimized prefill worker (800k tok/s) plus
  3 bandwidth-optimized decode workers (`t = 1.5 ms + 0.002 ms/token`,
  same KV cap). After prefill, the prompt KV cache
  (`2 * 32 layers * 4096 d_model * prompt_len * 2 bytes`, fp16) transfers over
  a 200 GB/s interconnect (+1 ms fixed latency) before decode starts.
  Same total worker count (4) in both architectures.
- **Workload:** Poisson arrivals at 4/10/16 req/s; prompt mix short (128) /
  long (2048) in short-heavy (80/20), mixed (50/50), long-heavy (20/80);
  `gen_len ~ Normal(256, 64)`. Seeded `np.random.default_rng(31)`.

Experiments (`experiments/run_experiments.py`) compare TTFT/TBT/throughput
across the workload x rate grid, sweep prompt length to find the TTFT
crossover, and sweep interconnect bandwidth to find where transfer overhead
erases the win.

## Results

All numbers from seeded simulation runs (two runs, byte-identical headlines).

**Headlines**

- Long prompts, high load (long-heavy, 16 req/s): disaggregated mean TTFT
  **17.1 ms vs colocated 40.4 ms**; p99 **166.2 ms vs 382.5 ms**.
- Mixed workload, high load: disaggregated TBT p99 **1.51 ms vs 2.05 ms**;
  throughput 3861 vs 3852 gen tok/s (nearly identical until saturation).
- Short prompts, low load: disaggregated TTFT 4.9 ms vs colocated 7.1 ms.
- **Crossover:** disaggregation starts winning on mean TTFT at
  **prompt_len ~81 tokens** (bracketed between 64 and 128). Below that, the
  KV-transfer tax dominates and colocated wins.
- **Bandwidth:** the mean-TTFT win is erased **below ~74 GB/s**
  (bracketed between 50 and 100 GB/s). At 5 GB/s, disaggregated mean TTFT is
  174.5 ms vs colocated 17.8 ms — the transfer alone adds ~215 ms per
  long prompt.
- TBT is 1.50–1.51 ms (disaggregated) vs 2.01–2.13 ms (colocated) across the
  whole grid: the decode pool's interference-free iterations give a steady
  ~25% lower token latency.
- Under saturation (mixed/high, long-heavy/high) colocated TTFT p99 explodes
  (186–383 ms) while disaggregated degrades gracefully (13–166 ms).

**Key honesty note:** this models scheduling and interference effects, not
real GPU kernels. Per-iteration time is the linear model
`t = T0 + K * batch_tokens`; prefill is not chunked; hardware parameters
(prefill throughput, interconnect bandwidth/latency, worker counts) are
modeling choices, documented in `results/results.md`. Absolute millisecond
values should not be read as predictions for any real GPU.

## Insights

1. **Disaggregation's TTFT win comes from two places:** a faster prefill pool
   (no waiting behind decode iterations) and the removal of prefill/decode
   interference. Either alone is worth little; together they give 1.4–2.4x
   mean TTFT improvements in this model.
2. **The transfer tax is real but small at datacenter interconnect speeds.**
   At 200 GB/s it only outweighs the win for prompts under ~81 tokens. The
   architecture is far more sensitive to interconnect *bandwidth* than to
   prompt length: halving bandwidth below ~74 GB/s flips the result, while
   prompt length only matters at the short end.
3. **TBT is where disaggregation never loses in this model.** Decode-only
   iterations are both faster per step and immune to prefill stalls, so token
   pacing is uniformly better — the tail-latency story for interactive use.
4. **Saturation behavior differs qualitatively.** When colocated workers fill
   their KV caps with long prompts, queueing compounds (prefill iterations
   grow, which slows decodes, which slows admissions). The disaggregated
   pools saturate independently and more gracefully.
5. **Utilization view:** at equal throughput the disaggregated pools run at
   lower mean utilization (e.g. 0.69 vs 0.97 mixed/high) — the prefill worker
   is rarely the bottleneck, so the same hardware has more headroom.

## Layout

- `src/llm_inference_engine/disagg_serve/sim.py` — the simulator
- `experiments/run_experiments.py` — experiment harness (seed 31)
- `results/charts.png`, `results/results.md` — charts and full tables
- `tests/test_smoke.py` — formula, determinism, conservation tests
  (`python3 tests/test_smoke.py`)
