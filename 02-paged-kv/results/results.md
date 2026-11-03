# 02-paged-kv — results

Seeded simulation (`np.random.default_rng(11)`), memory-management policy only: no attention math, no GPU. Analytical model: 1M token slots, sequence lengths from a 70/20/10 mixture of N(128,32) / N(512,128) / N(2048,256) clipped to [1, 2048], arrivals uniform over 600 time units, 1 token decoded per time unit, 2000 requests (measured mean length 373.0 tokens). Internal waste is the per-admission mean of (reserved_but_unused / reserved) over all admitted sequences.

## Experiment 1 — fixed budget, B=16 vs contiguous

| allocator | internal waste % (per-admission mean) | external frag events | max concurrent seqs | admitted | rejected |
|---|---|---|---|---|---|
| paged (B=16) | 1.90 | 0 | 817 | 2000 | 0 |
| contiguous (max_len=2048) | 81.51 | 0 | 488 | 1397 | 603 |

External fragmentation events are 0 for both allocators here: the paged allocator's blocks are fungible by construction, and the contiguous baseline reserves a uniform 2048 slots, so every freed interval is itself 2048 slots and can never be 'too small'. The contiguous allocator's rejections are pure internal fragmentation (reserving 2048 slots for ~373-token sequences on average).

## Experiment 2 — block-size sweep

Block-table overhead is modelled as `pool_blocks * 8 bytes` (one 8-byte block-table entry per physical block). For scale: at 32 layers x 4096 hidden x fp16, one token's KV cache is 524,288 bytes, so 1M slots ~= 524 GB of KV memory.

| B | paged waste % | contiguous waste % | paged block-table (MB) |
|---|---|---|---|
| 8 | 0.87 | 81.51 | 1.000 |
| 16 | 1.90 | 81.51 | 0.500 |
| 32 | 3.88 | 81.51 | 0.250 |
| 64 | 7.50 | 81.51 | 0.125 |

## Determinism

The full pipeline was executed twice from `np.random.default_rng(11)`; all headline numbers matched exactly (internal waste %, external events, max concurrency, rejections, and every sweep point).
