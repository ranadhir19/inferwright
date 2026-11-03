"""vLLM-style paged KV-cache block manager.

A paged allocator splits the KV-cache memory pool into fixed-size blocks
(``block_size`` tokens each). A sequence of ``n`` tokens occupies
``ceil(n / block_size)`` blocks, which need not be contiguous in physical
memory; a per-sequence block table maps logical block order to physical
block ids. Because any free block can serve any sequence, the paged scheme
suffers only *internal* fragmentation (the unused tail of each sequence's
last block) and never *external* fragmentation.

The naive baseline, :class:`ContiguousAllocator`, reserves ``max_len``
slots contiguously per sequence with first-fit placement, mirroring
pre-paged-attention serving stacks. It wastes the gap between a sequence's
actual length and ``max_len`` (internal fragmentation) and additionally
fails allocations when free memory is split into pieces smaller than
``max_len`` even though total free memory suffices (external
fragmentation).

Everything here is a CPU/numpy-level simulation of *memory management
policy*: no attention math, no GPU. The "GPU memory budget" is an
analytical token-slot count, and timings are an abstract decode model
(1 token per time unit).

Public API:
    BlockManager, ContiguousAllocator, gen_workload, Request
"""

from llm_inference_engine.paged_kv.blocks import BlockManager, ContiguousAllocator
from llm_inference_engine.paged_kv.workload import Request, gen_workload

__all__ = [
    "BlockManager",
    "ContiguousAllocator",
    "Request",
    "gen_workload",
]
