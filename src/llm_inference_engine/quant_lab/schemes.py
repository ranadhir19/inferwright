"""Weight quantization schemes (numpy simulation, CPU only).

Each scheme is a pure function ``(weights) -> (dequantized_weights, metadata)``
where ``dequantized_weights`` is the float32 reconstruction (what the model
would actually compute with) and ``metadata`` carries the scales and
bookkeeping needed for bit-rate accounting.

Conventions
-----------
* Linear weights use the ``(out_features, in_features)`` layout, i.e.
  ``y = x @ W.T``. "Per-channel" therefore means per output channel (per row),
  matching standard practice.
* Symmetric quantization: no zero-point, range clipped to the signed grid.
* Scale storage for bit-rate accounting is assumed fp16 (16 bits per scale),
  the standard choice in e.g. GPTQ/AWQ checkpoints. See
  :func:`effective_bits_per_param`.

FP8 E4M3 follows the OCP 8-bit floating point specification: 1 sign bit,
4 exponent bits, 3 mantissa bits, exponent bias 7. The only NaN pattern is
``S.1111.111``; there are no infinities, so out-of-range values saturate to
+/-448. (AMD's FNUZ variant instead uses bias 8 / max 240; it is *not* what is
simulated here.)
"""

import numpy as np

_TINY = 1e-12

# OCP FP8 E4M3 constants.
E4M3_BIAS = 7
E4M3_MAX = 448.0            # 2**8 * 1.75, largest finite value
E4M3_MIN_NORMAL = 2.0 ** -6
E4M3_MIN_SUBNORMAL = 2.0 ** -9


def _as_f32(a):
    return np.asarray(a, dtype=np.float32)


def quantize_int8_per_channel(w):
    """Symmetric INT8, one scale per output channel (row).

    Returns (dequantized float32 weights, metadata).
    """
    w = _as_f32(w)
    if w.ndim != 2:
        raise ValueError("per-channel INT8 expects a 2D (out, in) weight matrix")
    amax = np.abs(w).max(axis=1)
    scale = np.maximum(amax / 127.0, _TINY).astype(np.float32)
    q = np.clip(np.round(w / scale[:, None]), -128, 127)
    wq = (q * scale[:, None]).astype(np.float32)
    meta = {
        "scheme": "int8_per_channel",
        "bits": 8,
        "granularity": "per_output_channel",
        "scales": scale,
        "n_scales": int(scale.size),
    }
    return wq, meta


def quantize_int8_per_tensor(w):
    """Symmetric INT8, one scale for the whole tensor (baseline)."""
    w = _as_f32(w)
    amax = float(np.abs(w).max()) if w.size else 0.0
    scale = np.float32(max(amax / 127.0, _TINY))
    q = np.clip(np.round(w / scale), -128, 127)
    wq = (q * scale).astype(np.float32)
    meta = {
        "scheme": "int8_per_tensor",
        "bits": 8,
        "granularity": "per_tensor",
        "scales": np.array([scale], dtype=np.float32),
        "n_scales": 1,
    }
    return wq, meta


def quantize_int4_per_group(w, group_size=32):
    """Symmetric INT4, one scale per group of ``group_size`` input features.

    Groups run along the last axis; the axis is zero-padded when it is not a
    multiple of ``group_size`` (padding is discarded after dequantization).
    Grid is [-8, 7].
    """
    w = _as_f32(w)
    if w.ndim != 2:
        raise ValueError("per-group INT4 expects a 2D (out, in) weight matrix")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    out_f, in_f = w.shape
    n_groups = (in_f + group_size - 1) // group_size
    padded = np.zeros((out_f, n_groups * group_size), dtype=np.float32)
    padded[:, :in_f] = w
    g = padded.reshape(out_f, n_groups, group_size)
    amax = np.abs(g).max(axis=2)
    scale = np.maximum(amax / 7.0, _TINY).astype(np.float32)
    q = np.clip(np.round(g / scale[:, :, None]), -8, 7)
    wq = (q * scale[:, :, None]).reshape(out_f, n_groups * group_size)[:, :in_f]
    wq = wq.astype(np.float32)
    meta = {
        "scheme": "int4_per_group",
        "bits": 4,
        "granularity": f"per_group_g{group_size}",
        "group_size": group_size,
        "scales": scale,
        "n_scales": int(scale.size),
    }
    return wq, meta


def _to_e4m3_grid(x):
    """Round an array to the nearest OCP FP8 E4M3 grid value (ties round up)."""
    xf = np.asarray(x, dtype=np.float64).reshape(-1)
    sign = np.where(xf < 0.0, -1.0, 1.0)
    ax = np.abs(xf)

    # Normal-path decomposition: ax = (2m) * 2**(e-1), 2m in [1, 2).
    m, e = np.frexp(ax)
    E = (e - 1 + E4M3_BIAS).astype(np.int64)          # exponent field
    frac = 2.0 * m - 1.0                             # in [0, 1)
    q = np.floor(frac * 8.0 + 0.5).astype(np.int64)  # mantissa 0..8
    carry = q == 8
    q[carry] = 0
    E[carry] += 1
    normal_val = (1.0 + q / 8.0) * np.power(2.0, E - E4M3_BIAS)

    # Subnormal path (E < 1, i.e. ax < 2**-6): multiples of 2**-9.
    qd = np.floor(ax / E4M3_MIN_SUBNORMAL + 0.5).astype(np.int64)
    sub_val = np.minimum(qd, 7) * E4M3_MIN_SUBNORMAL
    sub_val = sub_val + (qd == 8) * E4M3_MIN_NORMAL  # rounds up to min normal

    val = np.where(E < 1, sub_val, normal_val)
    # NaN pattern (E == 15, q == 7) and exponent overflow saturate; E4M3 has
    # no infinities, matching hardware saturating casts.
    bad = ((E == 15) & (q == 7)) | (E > 15) | (ax > E4M3_MAX)
    val = np.where(bad, E4M3_MAX, val)
    val[ax == 0.0] = 0.0
    return (sign * val).reshape(np.shape(x)).astype(np.float32)


def quantize_fp8_e4m3(w):
    """Direct cast of float32 weights onto the OCP FP8 E4M3 grid (no scale).

    Real FP8 inference always pairs the format with a per-tensor/per-block
    scale because the grid's dynamic range is narrow (2**-9 .. 448); this
    scheme deliberately simulates the *unscaled* cast so the workbench can
    measure what happens without that scale.
    """
    w = _as_f32(w)
    wq = _to_e4m3_grid(w)
    meta = {
        "scheme": "fp8_e4m3",
        "bits": 8,
        "granularity": "direct_cast_no_scale",
        "scales": np.zeros(0, dtype=np.float32),
        "n_scales": 0,
    }
    return wq, meta


def fp8_e4m3_grid_values():
    """All 255 finite OCP FP8 E4M3 grid values, decoded from bit patterns.

    Independent implementation of the grid (bit-pattern decode rather than
    rounding), used to verify grid membership of quantized outputs.
    """
    vals = set()
    for b in range(256):
        s = (b >> 7) & 1
        E = (b >> 3) & 0xF
        M = b & 0x7
        if E == 15 and M == 7:
            continue  # NaN
        if E == 0:
            v = ((-1.0) ** s) * (2.0 ** -6) * (M / 8.0)
        else:
            v = ((-1.0) ** s) * (2.0 ** (E - E4M3_BIAS)) * (1.0 + M / 8.0)
        vals.add(float(np.float32(v)))
    return np.array(sorted(vals), dtype=np.float32)


def effective_bits_per_param(meta, n_params):
    """Effective storage bits per parameter, assuming fp16 scales.

    ``bits + 16 * n_scales / n_params``. Scale dtype is an accounting
    assumption only; dequantization itself uses float32 math.
    """
    if n_params <= 0:
        raise ValueError("n_params must be positive")
    return float(meta["bits"]) + 16.0 * meta["n_scales"] / n_params
