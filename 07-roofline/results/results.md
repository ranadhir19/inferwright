# Roofline analysis results

Shapes: d_model=4096, n_heads=32, ffn_mult=2.0 (LLaMA-7B-ish), fp16.
Hardware values are published specs (see module docstring).

## decode on NVIDIA A100 80GB (ridge = 153.0 FLOP/byte)

| op | FLOPs | bytes | intensity (FLOP/B) | bound | % of peak | est. time |
| --- | --- | --- | --- | --- | --- | --- |
| qkv_proj | 100.66 MFLOP | 100.70 MB | 1.00 | memory-bound | 0.65% | 49.39 us |
| attn_qk | 16.78 MFLOP | 16.92 MB | 0.99 | memory-bound | 0.65% | 8.30 us |
| softmax | 0.33 MFLOP | 0.26 MB | 1.25 | memory-bound | 0.82% | 0.13 us |
| attn_av | 16.78 MFLOP | 16.92 MB | 0.99 | memory-bound | 0.65% | 8.30 us |
| out_proj | 33.55 MFLOP | 33.57 MB | 1.00 | memory-bound | 0.65% | 16.46 us |
| ffn_gate_up | 134.22 MFLOP | 134.26 MB | 1.00 | memory-bound | 0.65% | 65.85 us |
| ffn_down | 67.11 MFLOP | 67.13 MB | 1.00 | memory-bound | 0.65% | 32.92 us |

Step: 369.43 MFLOP, 369.75 MB, step intensity 1.00 FLOP/byte, est. step time 181.3 us; 7/7 ops memory-bound; dominated by ffn_gate_up (36.3% of step time).

## decode on NVIDIA H100 80GB (ridge = 295.2 FLOP/byte)

| op | FLOPs | bytes | intensity (FLOP/B) | bound | % of peak | est. time |
| --- | --- | --- | --- | --- | --- | --- |
| qkv_proj | 100.66 MFLOP | 100.70 MB | 1.00 | memory-bound | 0.34% | 30.06 us |
| attn_qk | 16.78 MFLOP | 16.92 MB | 0.99 | memory-bound | 0.34% | 5.05 us |
| softmax | 0.33 MFLOP | 0.26 MB | 1.25 | memory-bound | 0.42% | 0.08 us |
| attn_av | 16.78 MFLOP | 16.92 MB | 0.99 | memory-bound | 0.34% | 5.05 us |
| out_proj | 33.55 MFLOP | 33.57 MB | 1.00 | memory-bound | 0.34% | 10.02 us |
| ffn_gate_up | 134.22 MFLOP | 134.26 MB | 1.00 | memory-bound | 0.34% | 40.08 us |
| ffn_down | 67.11 MFLOP | 67.13 MB | 1.00 | memory-bound | 0.34% | 20.04 us |

Step: 369.43 MFLOP, 369.75 MB, step intensity 1.00 FLOP/byte, est. step time 110.4 us; 7/7 ops memory-bound; dominated by ffn_gate_up (36.3% of step time).

## prefill on NVIDIA A100 80GB (ridge = 153.0 FLOP/byte)

| op | FLOPs | bytes | intensity (FLOP/B) | bound | % of peak | est. time |
| --- | --- | --- | --- | --- | --- | --- |
| qkv_proj | 1649.27 GFLOP | 637.53 MB | 2586.95 | compute-bound | 100.00% | 5.29 ms |
| attn_qk | 274.88 GFLOP | 2.416 GB | 113.78 | memory-bound | 74.36% | 1.18 ms |
| softmax | 5.37 GFLOP | 4.295 GB | 1.25 | memory-bound | 0.82% | 2.11 ms |
| attn_av | 274.88 GFLOP | 2.416 GB | 113.78 | memory-bound | 74.36% | 1.18 ms |
| out_proj | 549.76 GFLOP | 301.99 MB | 1820.44 | compute-bound | 100.00% | 1.76 ms |
| ffn_gate_up | 2199.02 GFLOP | 805.31 MB | 2730.67 | compute-bound | 100.00% | 7.05 ms |
| ffn_down | 1099.51 GFLOP | 469.76 MB | 2340.57 | compute-bound | 100.00% | 3.52 ms |

Step: 6052.68 GFLOP, 11.341 GB, step intensity 533.68 FLOP/byte, est. step time 22.10 ms; 3/7 ops memory-bound; dominated by ffn_gate_up (31.9% of step time).

## prefill on NVIDIA H100 80GB (ridge = 295.2 FLOP/byte)

| op | FLOPs | bytes | intensity (FLOP/B) | bound | % of peak | est. time |
| --- | --- | --- | --- | --- | --- | --- |
| qkv_proj | 1649.27 GFLOP | 637.53 MB | 2586.95 | compute-bound | 100.00% | 1.67 ms |
| attn_qk | 274.88 GFLOP | 2.416 GB | 113.78 | memory-bound | 38.54% | 721.17 us |
| softmax | 5.37 GFLOP | 4.295 GB | 1.25 | memory-bound | 0.42% | 1.28 ms |
| attn_av | 274.88 GFLOP | 2.416 GB | 113.78 | memory-bound | 38.54% | 721.17 us |
| out_proj | 549.76 GFLOP | 301.99 MB | 1820.44 | compute-bound | 100.00% | 555.87 us |
| ffn_gate_up | 2199.02 GFLOP | 805.31 MB | 2730.67 | compute-bound | 100.00% | 2.22 ms |
| ffn_down | 1099.51 GFLOP | 469.76 MB | 2340.57 | compute-bound | 100.00% | 1.11 ms |

Step: 6052.68 GFLOP, 11.341 GB, step intensity 533.68 FLOP/byte, est. step time 8.28 ms; 3/7 ops memory-bound; dominated by ffn_gate_up (26.8% of step time).

## Fusion estimates

### FlashAttention-style fusion, decode (A100)

- Attention intensity: 0.99 -> 1.01 FLOP/byte (flop-weighted before; bound memory-bound).
- Attention time: 16.72 us -> 16.46 us.
- Step time: 181.34 us -> 181.08 us (estimated step speedup 1.001x).

### FlashAttention-style fusion, prefill (A100)

- Attention intensity: 112.69 -> 1034.00 FLOP/byte (flop-weighted before; bound compute-bound).
- Attention time: 4476.12 us -> 1779.25 us.
- Step time: 22096.50 us -> 19399.62 us (estimated step speedup 1.139x).

### Fused dequant+GEMM (INT8 weights), decode (A100)

- GEMM op intensity: ['1.00', '1.00', '1.00', '1.00'] -> ['2.00', '2.00', '2.00', '2.00'] FLOP/byte (still memory-bound; dequant ALU cost neglected).
- Step time: 181.34 us -> 99.06 us (estimated step speedup 1.831x).

