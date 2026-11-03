"""Quickstart: weight quantization schemes compared.

Quantizes a synthetic weight matrix with outlier structure (like real LLM
weights) and compares reconstruction error across schemes.
"""
import numpy as np

from llm_inference_engine.quant_lab.metrics import relative_l2_error
from llm_inference_engine.quant_lab.schemes import (
    quantize_fp8_e4m3, quantize_int4_per_group, quantize_int8_per_channel,
    quantize_int8_per_tensor,
)

rng = np.random.default_rng(41)
# Mostly small values + rare large outliers, mimicking LLM weight matrices.
w = rng.normal(0, 0.02, size=(256, 256))
w[rng.random(w.shape) < 0.01] *= 20

for name, fn, bits in [
    ("INT8 per-tensor", quantize_int8_per_tensor, "8.00"),
    ("INT8 per-channel", quantize_int8_per_channel, "8.03"),
    ("FP8 E4M3", quantize_fp8_e4m3, "8.00"),
    ("INT4 group-128", lambda x: quantize_int4_per_group(x, 128), "4.22"),
    ("INT4 group-32", lambda x: quantize_int4_per_group(x, 32), "4.50"),
]:
    wq, _ = fn(w)
    print(f"{name:16s} rel-L2={relative_l2_error(w, wq):.4f}  ~{bits} bits/param")

print("Per-channel/per-group granularity exists to tame outliers.")
