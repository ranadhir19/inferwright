"""Smoke tests for the roofline model.

Run with: python3 tests/test_smoke.py   (also pytest-compatible)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm_inference_engine.roofline import (  # noqa: E402
    A100,
    LayerConfig,
    classify,
    fuse_dequant_gemm,
    fuse_flashattention,
    layer_ops,
    step_summary,
)

CFG = LayerConfig(d_model=4096, n_heads=32, ffn_mult=2.0, batch=1, seq=2048, dtype_bytes=2)


def test_matmul_intensity_grows_with_batch_times_seq():
    # QKV GEMM intensity should scale with B*q: more work per weight read.
    small = next(o for o in layer_ops(CFG, "decode") if o.name == "qkv_proj")
    big_cfg = LayerConfig(d_model=4096, n_heads=32, ffn_mult=2.0, batch=8, seq=2048, dtype_bytes=2)
    big = next(o for o in layer_ops(big_cfg, "prefill") if o.name == "qkv_proj")
    assert big.intensity > small.intensity
    # Roughly 8x batch * 2048x seq worth of growth (activation bytes dilute it).
    assert big.intensity / small.intensity > 100


def test_classification_threshold_at_ridge():
    ridge = A100.ridge_point("fp16")
    mem = classify(layer_ops(CFG, "decode"), A100, "fp16")
    assert all(c.bound == "memory-bound" for c in mem), "decode must be memory-bound"
    assert all(c.intensity < ridge for c in mem)
    pre = classify(layer_ops(CFG, "prefill"), A100, "fp16")
    proj = [c for c in pre if c.name in ("qkv_proj", "out_proj", "ffn_gate_up", "ffn_down")]
    assert all(c.bound == "compute-bound" for c in proj)
    assert all(c.intensity >= ridge for c in proj)
    assert all(abs(c.pct_peak - 100.0) < 1e-9 for c in proj)


def test_attainable_is_min_of_ceilings():
    cl = classify(layer_ops(CFG, "decode"), A100, "fp16")
    for c in cl:
        expected = min(A100.peak("fp16"), c.intensity * A100.memory_bw)
        assert abs(c.attainable_flops - expected) / expected < 1e-12
        assert abs(c.time_s - c.flops / c.attainable_flops) < 1e-18


def test_fusion_never_increases_bytes():
    for mode in ("decode", "prefill"):
        ops = layer_ops(CFG, mode)
        fused = fuse_flashattention(ops)
        b0 = sum(o.bytes for o in ops)
        b1 = sum(o.bytes for o in fused)
        assert b1 <= b0, f"fusion increased bytes in {mode}"
        f0 = sum(o.flops for o in ops)
        f1 = sum(o.flops for o in fused)
        assert f0 == f1, "fusion must not change FLOPs"


def test_flashattention_drops_score_matrix_traffic():
    # Prefill: score matrix is 4 * B*H*S^2 bytes of the attention traffic.
    ops = layer_ops(CFG, "prefill")
    attn = [o for o in ops if o.name in ("attn_qk", "softmax", "attn_av")]
    b0 = sum(o.bytes for o in attn)
    fused = fuse_flashattention(ops)
    f = next(o for o in fused if o.name == "attn_fused")
    softmax = next(o for o in attn if o.name == "softmax")
    assert abs(f.bytes - (b0 - 4 * softmax.bytes / 2)) < 1e-6
    assert f.intensity > max(o.intensity for o in attn)


def test_dequant_reduces_only_weight_bytes():
    ops = layer_ops(CFG, "decode")
    dq = fuse_dequant_gemm(ops, quant_bytes=1, dtype_bytes=2)
    for orig, new in zip(ops, dq):
        if orig.weight_bytes > 0:
            assert new.weight_bytes == orig.weight_bytes / 2
            assert new.bytes < orig.bytes
            assert new.flops == orig.flops
        else:
            assert new is orig or (new.bytes == orig.bytes and new.flops == orig.flops)


def test_step_summary_dominant_op():
    cl = classify(layer_ops(CFG, "decode"), A100, "fp16")
    s = step_summary(cl)
    assert s["dominant_op"] == "ffn_gate_up"
    assert abs(s["step_time_s"] - sum(c.time_s for c in cl)) < 1e-12
    assert s["n_memory_bound"] == 7


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
