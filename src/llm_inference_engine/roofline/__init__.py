"""Roofline performance model for LLM inference ops (numpy, CPU-only).

Analytical model: per-op FLOP/byte accounting for a transformer decoder
layer, roofline classification (attainable = min(peak, intensity * bw)),
and kernel-fusion transforms that recompute intensity and shift the bound.

Public API:
    hardware: Hardware, A100, H100, CPU_REFERENCE
    ops:      LayerConfig, OpStats, layer_ops
    model:    classify, step_summary, fuse_flashattention, fuse_dequant_gemm
"""

from .hardware import A100, CPU_REFERENCE, H100, ALL_HARDWARE, Hardware
from .model import (
    COMPUTE_BOUND,
    MEMORY_BOUND,
    ClassifiedOp,
    classify,
    fuse_dequant_gemm,
    fuse_flashattention,
    step_summary,
)
from .ops import LayerConfig, OpStats, layer_ops

__all__ = [
    "Hardware",
    "A100",
    "H100",
    "CPU_REFERENCE",
    "ALL_HARDWARE",
    "LayerConfig",
    "OpStats",
    "layer_ops",
    "ClassifiedOp",
    "classify",
    "step_summary",
    "fuse_flashattention",
    "fuse_dequant_gemm",
    "MEMORY_BOUND",
    "COMPUTE_BOUND",
]
