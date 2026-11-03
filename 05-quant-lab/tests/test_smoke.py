"""Smoke tests for quant_lab: error bounds, FP8 grid membership, determinism.

Run with ``python3 -m pytest 05-quant-lab/tests/test_smoke.py`` from the repo
root, or directly: ``python3 05-quant-lab/tests/test_smoke.py``.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from llm_inference_engine.quant_lab import metrics, schemes

RNG = np.random.default_rng(7)


def test_int8_per_channel_roundtrip_bounded():
    w = RNG.normal(0, 1, (64, 128)).astype(np.float32)
    wq, meta = schemes.quantize_int8_per_channel(w)
    assert wq.shape == w.shape and wq.dtype == np.float32
    assert metrics.relative_l2_error(w, wq) < 0.02
    assert meta["scales"].shape == (64,)


def test_int8_per_tensor_roundtrip_bounded():
    w = RNG.normal(0, 1, (64, 128)).astype(np.float32)
    wq, _ = schemes.quantize_int8_per_tensor(w)
    assert metrics.relative_l2_error(w, wq) < 0.02


def test_int4_per_group_roundtrip_bounded():
    w = RNG.normal(0, 1, (64, 128)).astype(np.float32)
    wq, meta = schemes.quantize_int4_per_group(w, group_size=32)
    assert metrics.relative_l2_error(w, wq) < 0.10
    assert meta["scales"].shape == (64, 4)
    # padding path: in_features not divisible by group_size
    w2 = RNG.normal(0, 1, (8, 100)).astype(np.float32)
    wq2, _ = schemes.quantize_int4_per_group(w2, group_size=32)
    assert wq2.shape == w2.shape


def test_fp8_e4m3_roundtrip_bounded():
    w = RNG.normal(0, 1, (64, 128)).astype(np.float32)
    wq, _ = schemes.quantize_fp8_e4m3(w)
    assert metrics.relative_l2_error(w, wq) < 0.06


def test_fp8_grid_membership():
    grid = set(schemes.fp8_e4m3_grid_values().tolist())
    # 256 patterns - 2 NaN (S.1111.111) - 1 duplicate zero (+0.0 == -0.0)
    assert len(grid) == 253
    edge = np.array([0.0, -0.0, 1e-4, -1e-4, 0.5, -2.0, 100.0,
                     447.9, 448.0, 449.0, 1e6, -1e6, np.float32(2.0) ** -9],
                    dtype=np.float32)
    w = np.concatenate([RNG.normal(0, 2, 5000).astype(np.float32), edge])
    wq, _ = schemes.quantize_fp8_e4m3(w)
    missing = [v for v in np.unique(wq).tolist() if v not in grid]
    assert not missing, f"off-grid values: {missing[:5]}"
    # saturation behavior (E4M3 has no infinities)
    assert float(wq[-4]) == 448.0   # 449.0 saturates
    assert float(wq[-3]) == 448.0   # 1e6 saturates
    assert float(wq[-2]) == -448.0  # -1e6 saturates
    assert float(schemes.quantize_fp8_e4m3(
        np.array([448.0], dtype=np.float32))[0][0]) == 448.0


def test_determinism():
    w = RNG.normal(0, 1, (32, 64)).astype(np.float32)
    for fn, kw in [(schemes.quantize_int8_per_channel, {}),
                   (schemes.quantize_int8_per_tensor, {}),
                   (schemes.quantize_int4_per_group, {"group_size": 32}),
                   (schemes.quantize_fp8_e4m3, {})]:
        a, ma = fn(w, **kw)
        b, mb = fn(w, **kw)
        assert np.array_equal(a, b), f"{fn.__name__} not deterministic"
        assert np.array_equal(ma["scales"], mb["scales"])


def test_zero_tensor_safe():
    w = np.zeros((16, 32), dtype=np.float32)
    for fn, kw in [(schemes.quantize_int8_per_channel, {}),
                   (schemes.quantize_int8_per_tensor, {}),
                   (schemes.quantize_int4_per_group, {"group_size": 32}),
                   (schemes.quantize_fp8_e4m3, {})]:
        wq, _ = fn(w, **kw)
        assert np.all(wq == 0.0) and np.all(np.isfinite(wq))


def test_effective_bits_per_param():
    # (64,128) INT4 g32 -> 64*4 scales of fp16: 4 + 16*256/8192 = 4.5
    _, meta = schemes.quantize_int4_per_group(
        np.zeros((64, 128), dtype=np.float32), group_size=32)
    assert abs(schemes.effective_bits_per_param(meta, 64 * 128) - 4.5) < 1e-9
    _, meta8 = schemes.quantize_int8_per_channel(
        np.zeros((64, 128), dtype=np.float32))
    bpp8 = schemes.effective_bits_per_param(meta8, 64 * 128)
    assert abs(bpp8 - (8 + 16 * 64 / 8192)) < 1e-9


def test_metrics_sanity():
    w = RNG.normal(0, 1, (32, 32)).astype(np.float32)
    assert metrics.relative_l2_error(w, w) == 0.0
    assert metrics.max_abs_error(w, w) == 0.0
    assert metrics.snr_db(w, w) == float("inf")
    wq, _ = schemes.quantize_int8_per_channel(w)
    rep = metrics.reconstruction_report(w, wq)
    assert set(rep) == {"relative_l2", "snr_db", "max_abs", "mean_abs"}
    assert rep["snr_db"] > 30.0  # INT8 on clean gaussian is high-SNR


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} smoke tests passed")
