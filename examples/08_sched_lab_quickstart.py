"""Quickstart: request scheduling policies compared.

Static FCFS batching vs continuous (iteration-level) batching at high load:
TTFT, tail latency, and throughput.
"""
import numpy as np

from llm_inference_engine.sched_lab import make_workload, run_policy

rng = np.random.default_rng(61)
specs = make_workload(rng, n=1500, arrival_rate=6.0)

for policy in ("static-fcfs", "continuous", "chunked-prefill"):
    res = run_policy(policy, specs)
    ttft = np.asarray(res.ttft_ms)
    print(f"{policy:15s} mean TTFT={ttft.mean():7.1f} ms  "
          f"p99 TTFT={np.percentile(ttft, 99):7.1f} ms  "
          f"throughput={res.total_gen_tokens / res.makespan_s:.0f} tok/s")

print("Scheduling moves latency, not capacity: throughput is ~identical, "
      "TTFT differs by 50x+.")
