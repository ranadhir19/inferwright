"""Disaggregated prefill/decode LLM serving simulator (CPU/numpy only).

Discrete-event simulation that quantifies TTFT vs TBT tradeoffs between two
serving architectures:

- **Colocated**: a pool of identical workers. Each request occupies a worker
  for its prefill *and* all decode steps. A new request's prefill is batched
  with ongoing decodes on one iteration, so long prefills stall everyone
  else's token stream (head-of-line / interference effects).
- **Disaggregated**: a compute-optimized prefill pool plus a
  bandwidth-optimized decode pool. After prefill, the prompt KV cache is
  copied to a decode worker over an interconnect of finite bandwidth, which
  adds an explicit transfer delay before the first decode token.

The model captures *scheduling and interference effects* (queueing behind
decode batches, KV-cache capacity limits, KV transfer overhead), not real GPU
kernels. Per-iteration time on a worker is the linear model
``t = T0 + K * batch_tokens`` subject to a KV-token memory cap.

Public API:
    kv_cache_bytes, transfer_delay_s, make_workload,
    ColocatedConfig, DisaggConfig, Request, RunResult,
    run_colocated, run_disaggregated
"""

from .sim import (
    ColocatedConfig,
    DisaggConfig,
    Request,
    RunResult,
    kv_cache_bytes,
    make_workload,
    run_colocated,
    run_disaggregated,
    transfer_delay_s,
)

__all__ = [
    "ColocatedConfig",
    "DisaggConfig",
    "Request",
    "RunResult",
    "kv_cache_bytes",
    "make_workload",
    "run_colocated",
    "run_disaggregated",
    "transfer_delay_s",
]
