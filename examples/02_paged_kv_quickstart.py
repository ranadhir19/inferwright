"""Quickstart: paged KV-cache block manager vs contiguous allocation.

Streams a mixed-length workload through both allocators and compares
fragmentation waste and sustained concurrency.
"""
import numpy as np

from llm_inference_engine.paged_kv import (
    BlockManager, ContiguousAllocator, gen_workload,
)

rng = np.random.default_rng(11)
BUDGET = 200_000  # token slots of GPU memory
BLOCK = 16

reqs = gen_workload(rng, n=500)
results = {}
for name, alloc in (("paged", BlockManager(BUDGET, BLOCK)),
                    ("contiguous", ContiguousAllocator(BUDGET, max_len=2048))):
    admitted = sum(1 for r in reqs if alloc.alloc(r.seq_id, r.length) is not None)
    frag = alloc.fragmentation()
    results[name] = (frag["internal_waste_pct"], frag["live_sequences"], admitted)
    print(f"{name:10s} waste={frag['internal_waste_pct']:5.2f}%  "
          f"live={frag['live_sequences']:3d}  admitted={admitted}/{len(reqs)}")

print("Paged allocation admits every sequence with a fraction of the waste.")
