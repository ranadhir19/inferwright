# llm-mesh: experiment results (seed 7)

All numbers are simulator output, not invented.

## Experiment 1 -- mesh vs single frontier model (400 requests)

| system | mean quality | total cost (USD) | p95 latency (ms) |
|---|---|---|---|
| single (frontier-1) | 0.9150 | 5.0022 | 4955.5 |
| mesh (quality-first) | 0.9430 | 1.9168 | 6394.6 |
| mesh / single | 1.031x | 0.383x | 1.290x |

## Experiment 2 -- semantic router accuracy

TF-IDF Rocchio router trained on 72 labeled queries, tested on 24 held-out queries: accuracy **1.0000**.

## Experiment 3 -- parallel vs sequential orchestration

200 two-subtask requests (vision chart + chat report): sequential mean wall latency 6078.0 ms, parallel 3706.9 ms -- **1.64x speedup** (p95 6946.9 vs 4486.5 ms).

## Experiment 4 -- self-improving router (UCB1, 3000 requests)

Mean regret vs oracle: first 100-request window 0.1566, last window 0.0296. Final oracle-model selection share:
- reasoning: reasoner-r1 selected 53.3% of the time
- coding: coder-cl selected 53.1% of the time
- vision: vision-qw selected 61.9% of the time
- chat: frontier-1 selected 47.1% of the time

## Experiment 5 -- failover under vision-qw outage (400 requests)

Static per-task mapping without health checks: success rate 0.750. Mesh with re-routing: success rate 1.000, mean quality 0.9353.

## Experiment 6 -- routing-policy Pareto (400 requests)

| policy | mean quality | total cost (USD) |
|---|---|---|
| quality-first | 0.9439 | 1.9171 |
| cost-first | 0.7815 | 0.3305 |
| latency-first | 0.7825 | 0.3305 |
| balanced | 0.8473 | 0.4271 |
