"""Roofline classification and kernel-fusion transforms.

Roofline: for an op with arithmetic intensity I (FLOP/byte) on hardware
with peak FLOPS P and bandwidth Bw, the attainable performance is

    attainable = min(P, I * Bw)

An op is *memory-bound* when I < ridge point (P / Bw), i.e. the slanted
bandwidth ceiling binds, and *compute-bound* otherwise. Op time is
estimated as flops / attainable, so step time is the sum over ops.

Fusion transforms rewrite the OpStats list to model what a fused kernel
would do to traffic, then re-classification shows the shift:

- ``fuse_flashattention``: QK^T + softmax + AV become one kernel that never
  materializes the (B, H, S_q, S_k) score matrix. Bytes drop from
  O(B*H*S^2) to O(B*H*S): only Q, K, V reads and the output write remain.
  FLOPs are unchanged (same math, less traffic).
- ``fuse_dequant_gemm``: weights are stored at ``quant_bytes`` per
  parameter (e.g. INT8 -> 1) and dequantized on the fly in registers.
  Weight-read bytes scale by quant_bytes / dtype_bytes; activation bytes
  and FLOPs are unchanged (dequant ALU cost is neglected, documented
  here as a limitation).
"""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import Hardware
from .ops import OpStats

MEMORY_BOUND = "memory-bound"
COMPUTE_BOUND = "compute-bound"


@dataclass(frozen=True)
class ClassifiedOp:
    name: str
    flops: float
    bytes: float
    intensity: float
    attainable_flops: float
    pct_peak: float
    bound: str
    time_s: float


def classify(ops: list[OpStats], hw: Hardware, dtype: str = "fp16") -> list[ClassifiedOp]:
    """Apply the roofline model to each op."""
    peak = hw.peak(dtype)
    bw = hw.memory_bw
    ridge = hw.ridge_point(dtype)
    out = []
    for op in ops:
        intensity = op.intensity
        attainable = min(peak, intensity * bw)
        bound = MEMORY_BOUND if intensity < ridge else COMPUTE_BOUND
        out.append(
            ClassifiedOp(
                name=op.name,
                flops=op.flops,
                bytes=op.bytes,
                intensity=intensity,
                attainable_flops=attainable,
                pct_peak=100.0 * attainable / peak,
                bound=bound,
                time_s=op.flops / attainable,
            )
        )
    return out


def step_summary(classified: list[ClassifiedOp]) -> dict:
    """Aggregate one layer step: totals, dominant op, bound mix."""
    total_flops = sum(c.flops for c in classified)
    total_bytes = sum(c.bytes for c in classified)
    total_time = sum(c.time_s for c in classified)
    dominant = max(classified, key=lambda c: c.time_s)
    n_mem = sum(1 for c in classified if c.bound == MEMORY_BOUND)
    return {
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "step_intensity": total_flops / total_bytes,
        "step_time_s": total_time,
        "dominant_op": dominant.name,
        "dominant_time_s": dominant.time_s,
        "dominant_share": dominant.time_s / total_time,
        "n_memory_bound": n_mem,
        "n_compute_bound": len(classified) - n_mem,
    }


def fuse_flashattention(ops: list[OpStats]) -> list[OpStats]:
    """Fuse attn_qk + softmax + attn_av into one kernel.

    The fused kernel streams K/V tiles and never materializes the score
    matrix: bytes = Q + K + V reads + output write. FLOPs unchanged.
    """
    names = {"attn_qk", "softmax", "attn_av"}
    fused_parts = [op for op in ops if op.name in names]
    rest = [op for op in ops if op.name not in names]
    if len(fused_parts) != 3:
        raise ValueError("expected attn_qk, softmax, attn_av ops to fuse")

    flops = sum(op.flops for op in fused_parts)
    # attn_qk bytes = Q + K + scores; softmax bytes = 2 * scores;
    # attn_av bytes = scores + V + O. Fused bytes = Q + K + V + O, so
    # subtract the 4 score-matrix terms (2 in softmax, 1 each in qk/av).
    # The score matrix size is recovered from the softmax op (2x its bytes).
    softmax_op = next(op for op in fused_parts if op.name == "softmax")
    score_matrix_bytes = softmax_op.bytes / 2.0
    total_bytes = sum(op.bytes for op in fused_parts)
    fused_bytes = total_bytes - 4.0 * score_matrix_bytes
    fused = OpStats(
        name="attn_fused",
        flops=flops,
        bytes=fused_bytes,
        weight_bytes=0.0,
    )
    # Keep position of the first attention op for stable ordering.
    first_idx = min(ops.index(op) for op in fused_parts)
    return rest[:first_idx] + [fused] + rest[first_idx:]


def fuse_dequant_gemm(ops: list[OpStats], quant_bytes: int, dtype_bytes: int) -> list[OpStats]:
    """Model fused dequant+GEMM: weights stored at quant_bytes/param.

    Weight-read bytes scale by quant_bytes / dtype_bytes; everything else
    (activations, FLOPs) is unchanged. Applies to ops that carry
    weight_bytes > 0.
    """
    scale = quant_bytes / dtype_bytes
    out = []
    for op in ops:
        if op.weight_bytes > 0:
            new_weight = op.weight_bytes * scale
            out.append(
                OpStats(
                    name=op.name + "_dequant",
                    flops=op.flops,
                    bytes=new_weight + op.activation_bytes,
                    weight_bytes=new_weight,
                )
            )
        else:
            out.append(op)
    return out
