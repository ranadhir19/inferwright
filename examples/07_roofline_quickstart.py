"""Quickstart: roofline analysis of a transformer decode step.

Classifies each op of a 7B-ish decoder layer as memory- or compute-bound on
an A100, and shows what kernel fusion buys.
"""
from llm_inference_engine.roofline import (
    A100, LayerConfig, classify, fuse_dequant_gemm, fuse_flashattention,
    layer_ops, step_summary,
)

cfg = LayerConfig(d_model=4096, n_heads=32, ffn_mult=2.0, batch=1, seq=2048)
ops = layer_ops(cfg, mode="decode")
classified = classify(ops, A100)
s = step_summary(classified)
n = len(classified)

print(f"decode step on {A100.name}: "
      f"{s['n_memory_bound']}/{n} ops memory-bound, "
      f"step time ~{s['step_time_s'] * 1e6:.1f} us")
print(f"dominant op: {s['dominant_op']} ({s['dominant_share']:.1%} of step)")

base = s["step_time_s"]
fa = step_summary(classify(fuse_flashattention(ops), A100))["step_time_s"]
dq = step_summary(classify(fuse_dequant_gemm(ops, quant_bytes=1, dtype_bytes=2), A100))["step_time_s"]
print(f"FlashAttention-style fusion: {base / fa:.3f}x step speedup")
print(f"fused INT8 dequant+GEMM:      {base / dq:.3f}x step speedup")
print("Decode is memory-bound: bandwidth is the wall, fusion moves the ceiling.")
