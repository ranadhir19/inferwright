# sched-lab results

Seed: 61 (`np.random.default_rng(61)`, single sequential stream).

GPU model (analytical): `t = 2.0 + 0.012 * batch_tokens` ms per iteration; KV-cache token cap 16384; per-request reservation `prompt_len + gen_len`.

Workload: Poisson arrivals; 85% chat (prompt 32-512, gen 32-512) + 15% documents (prompt 1024-4096, gen 128-1024).

## Experiment 1 -- policy x load (1500 requests per cell)

| rate (req/s) | policy | TTFT mean (ms) | TTFT p99 (ms) | TPOT mean (ms) | TPOT p99 (ms) | throughput (tok/s) | GPU util | preemptions |
|:---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | static-fcfs | 285.5 | 1688.7 | 2.01 | 2.04 | 320.7 | 0.002 | 0 |
| 1.0 | continuous | 12.2 | 52.0 | 2.03 | 2.22 | 320.7 | 0.002 | 0 |
| 1.0 | chunked-prefill | 13.6 | 66.1 | 2.03 | 2.22 | 320.7 | 0.001 | 0 |
| 1.0 | priority-preempt | 12.2 | 52.0 | 2.03 | 2.22 | 320.7 | 0.002 | 0 |
| 3.0 | static-fcfs | 635.2 | 2001.9 | 2.03 | 2.13 | 976.5 | 0.007 | 0 |
| 3.0 | continuous | 13.5 | 53.3 | 2.08 | 2.44 | 976.5 | 0.004 | 0 |
| 3.0 | chunked-prefill | 15.1 | 67.5 | 2.08 | 2.44 | 976.5 | 0.001 | 0 |
| 3.0 | priority-preempt | 13.5 | 53.3 | 2.08 | 2.44 | 976.5 | 0.004 | 0 |
| 6.0 | static-fcfs | 844.6 | 2437.1 | 2.08 | 2.15 | 1858.2 | 0.019 | 0 |
| 6.0 | continuous | 15.0 | 57.5 | 2.14 | 2.47 | 1860.8 | 0.006 | 0 |
| 6.0 | chunked-prefill | 16.1 | 69.1 | 2.14 | 2.48 | 1860.8 | 0.002 | 0 |
| 6.0 | priority-preempt | 14.9 | 57.1 | 2.14 | 2.59 | 1860.8 | 0.006 | 3 |

## Experiment 2 -- burst of 10 x 4096-token prompts at t=60 s amid pure chat traffic (3 req/s, no background documents)

| policy | chat TTFT mean (ms) | chat TTFT p99 (ms) | burst TTFT mean (ms) | max iteration latency (ms) | throughput (tok/s) |
|:---|---:|---:|---:|---:|---:|
| static-fcfs | 382.5 | 1032.5 | 1758.5 | 149.5 | 819.4 |
| continuous | 8.2 | 12.0 | 966.4 | 149.5 | 823.4 |
| chunked-prefill | 8.2 | 12.0 | 997.5 | 25.7 | 823.4 |
| priority-preempt | 8.2 | 12.0 | 966.4 | 149.5 | 823.4 |

## Experiment 3 -- preemption cost: priority-preempt vs continuous (6 req/s, 30% long, burst at t=30 s)

| policy | short TTFT mean (ms) | short TTFT p99 (ms) | all TTFT p99 (ms) | throughput (tok/s) | preemptions | recompute tokens | extra compute (%) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| continuous | 20.2 | 362.4 | 2078.6 | 2112.2 | 0 | 0 | 0.00 |
| priority-preempt | 10.0 | 44.0 | 2078.9 | 2112.2 | 14 | 43598 | 4.88 |

## Headlines

- high load (6 req/s): continuous TTFT mean 15.0 ms / p99 57.5 ms vs static-fcfs 844.6 ms / p99 2437.1 ms
- burst of 10x4096-token prompts amid chat traffic: chunked-prefill holds chat TTFT p99 at 12.0 ms (same as continuous 12.0 ms) while bounding worst-case iteration latency to 25.7 ms vs 149.5 ms (82.8% lower); static-fcfs collapses to p99 1032.5 ms (98.8% worse than chunked-prefill)
- priority-preempt: 14 preemptions cost 43598 recompute tokens (4.88% extra compute) for 87.9% lower short-request TTFT p99 (362.4 -> 44.0 ms)

Note: the iteration-cost model is analytical, calibrated to the shape of real serving curves (fixed launch overhead + linear token scaling), not measured on a GPU. Absolute millisecond values are illustrative; relative policy differences are the finding.
