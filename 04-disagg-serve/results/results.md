# 04-disagg-serve: results

Seed: 31 (np.random.default_rng). All runs deterministic.

## Model parameters

- Colocated: 4 workers, per-iteration t = 2.0 ms + 0.009 ms/token, KV cap 8192 tokens/worker.
- Disaggregated: 1 prefill worker(s) at 800,000 tok/s + 3 decode workers (t = 1.5 ms + 0.002 ms/token, KV cap 8192 tokens/worker).
- KV cache: 2 * 32 layers * 4096 d_model * prompt_len * 2 bytes (fp16). Interconnect 200 GB/s + 1.0 ms fixed transfer latency.
- Workload: Poisson arrivals; prompt mix short=128 / long=2048 tokens; gen_len ~ Normal(256, 64) clipped at >= 1.

## Experiment 1: workload x arrival rate

| workload | rate | arch | TTFT mean (ms) | TTFT p99 (ms) | TBT mean (ms) | TBT p99 (ms) | throughput (tok/s) | util |
|---|---|---|---|---|---|---|---|---|
| short-heavy | low (4/s) | colocated | 7.1 | 22.1 | 2.01 | 2.02 | 1032.2 | 0.49 |
| short-heavy | low (4/s) | disaggregated | 4.9 | 11.7 | 1.50 | 1.51 | 1032.5 | 0.31 |
| short-heavy | med (10/s) | colocated | 7.6 | 22.4 | 2.03 | 2.03 | 2535.0 | 0.88 |
| short-heavy | med (10/s) | disaggregated | 5.2 | 11.8 | 1.50 | 1.51 | 2538.9 | 0.55 |
| short-heavy | high (16/s) | colocated | 7.2 | 22.3 | 2.05 | 2.05 | 4000.8 | 0.97 |
| short-heavy | high (16/s) | disaggregated | 5.0 | 11.8 | 1.51 | 1.51 | 4011.4 | 0.66 |
| mixed | low (4/s) | colocated | 11.5 | 22.3 | 2.02 | 2.02 | 1065.1 | 0.52 |
| mixed | low (4/s) | disaggregated | 6.8 | 11.8 | 1.50 | 1.51 | 1065.6 | 0.32 |
| mixed | med (10/s) | colocated | 12.3 | 22.4 | 2.05 | 2.04 | 2483.6 | 0.86 |
| mixed | med (10/s) | disaggregated | 7.1 | 11.9 | 1.50 | 1.51 | 2486.9 | 0.56 |
| mixed | high (16/s) | colocated | 16.7 | 186.3 | 2.09 | 2.05 | 3852.2 | 0.97 |
| mixed | high (16/s) | disaggregated | 8.3 | 13.2 | 1.51 | 1.51 | 3861.4 | 0.69 |
| long-heavy | low (4/s) | colocated | 17.2 | 22.3 | 2.02 | 2.02 | 1044.2 | 0.51 |
| long-heavy | low (4/s) | disaggregated | 9.2 | 11.9 | 1.50 | 1.51 | 1044.9 | 0.35 |
| long-heavy | med (10/s) | colocated | 18.0 | 22.4 | 2.07 | 2.03 | 2475.9 | 0.88 |
| long-heavy | med (10/s) | disaggregated | 9.5 | 12.3 | 1.50 | 1.51 | 2478.5 | 0.61 |
| long-heavy | high (16/s) | colocated | 40.4 | 382.5 | 2.13 | 2.05 | 4203.9 | 0.98 |
| long-heavy | high (16/s) | disaggregated | 17.1 | 166.2 | 1.51 | 1.51 | 4215.1 | 0.72 |

## Experiment 2a: TTFT vs prompt length (rate 6/s)

| prompt_len | colocated TTFT mean/p99 (ms) | disaggregated TTFT mean/p99 (ms) |
|---|---|---|
| 64 | 3.0 / 4.6 | 3.1 / 4.2 |
| 128 | 3.6 / 5.1 | 3.3 / 4.5 |
| 256 | 4.6 / 6.3 | 3.8 / 5.0 |
| 512 | 6.9 / 8.5 | 4.7 / 5.9 |
| 1024 | 11.6 / 13.2 | 6.8 / 7.9 |
| 2048 | 20.8 / 22.4 | 10.8 / 11.9 |
| 4096 | 248.3 / 911.3 | 115.3 / 480.2 |

Crossover (mean TTFT): disaggregation starts winning at prompt_len ~81 tokens (bracketed between 64 and 128).

## Experiment 2b: TTFT vs interconnect bandwidth (long-heavy, 10/s)

| bandwidth (GB/s) | colocated TTFT mean/p99 (ms) | disaggregated TTFT mean/p99 (ms) |
|---|---|---|
| 5 | 17.8 / 22.4 | 174.5 / 221.3 |
| 10 | 17.9 / 27.9 | 92.5 / 114.5 |
| 20 | 18.2 / 22.4 | 47.2 / 60.2 |
| 50 | 19.9 / 44.7 | 24.0 / 31.0 |
| 100 | 18.1 / 22.4 | 13.8 / 17.9 |
| 200 | 18.3 / 22.4 | 9.7 / 12.0 |
| 400 | 19.0 / 22.5 | 8.1 / 10.2 |

The disaggregation win on mean TTFT is erased below ~74 GB/s (bracketed between 50 and 100 GB/s); at 5 GB/s the KV transfer alone adds ~215 ms per long prompt.

## Honesty notes

- This is a scheduling/interference model, not real GPU kernels: per-iteration time is the linear model t = T0 + K * batch_tokens.
- Prefill is not chunked; a full prompt is processed in one iteration.
- Both architectures get the same total worker count (4 colocated vs 1 prefill + 3 decode); the split is a modeling choice.
- Decode-pool routing is join-shortest-queue by assigned KV tokens.
