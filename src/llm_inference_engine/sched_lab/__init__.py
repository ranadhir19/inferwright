"""sched_lab: advanced LLM request-scheduling simulator (numpy, CPU only).

Compares four serving policies -- static FCFS batching, continuous
iteration-level batching, chunked prefill, and priority preemption -- on
latency (TTFT, TPOT), throughput, and GPU utilization, using an analytical
per-iteration cost model ``t = T0 + K * batch_tokens`` and a hard KV-cache
token budget with per-request reservations.

Public API:
    SimConfig, Req, Scheduler, StaticFCFSScheduler, ContinuousScheduler,
    ChunkedPrefillScheduler, PriorityPreemptScheduler, POLICY_NAMES,
    make_scheduler, Simulator, SimResult, summarize,
    make_workload, add_burst, run_policy
"""

from .sim import (
    AGE_S,
    CHUNK_TOKENS,
    K_MS_PER_TOKEN,
    KV_CAP_TOKENS,
    MAX_BATCH,
    POLICY_NAMES,
    SHORT_TOTAL_TOKENS,
    T0_MS,
    ChunkedPrefillScheduler,
    ContinuousScheduler,
    PriorityPreemptScheduler,
    Req,
    Scheduler,
    SimConfig,
    Simulator,
    SimResult,
    StaticFCFSScheduler,
    add_burst,
    make_scheduler,
    make_workload,
    run_policy,
    summarize,
)

__all__ = [
    "AGE_S",
    "CHUNK_TOKENS",
    "K_MS_PER_TOKEN",
    "KV_CAP_TOKENS",
    "MAX_BATCH",
    "POLICY_NAMES",
    "SHORT_TOTAL_TOKENS",
    "T0_MS",
    "ChunkedPrefillScheduler",
    "ContinuousScheduler",
    "PriorityPreemptScheduler",
    "Req",
    "Scheduler",
    "SimConfig",
    "Simulator",
    "SimResult",
    "StaticFCFSScheduler",
    "add_burst",
    "make_scheduler",
    "make_workload",
    "run_policy",
    "summarize",
]
