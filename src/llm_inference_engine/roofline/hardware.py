"""Hardware specifications for the roofline model.

All peak-FLOPS and memory-bandwidth numbers below are *published
specifications* (vendor datasheets), not measured values:

- A100 80GB: 312 dense FP16 TFLOPS, 2039 GB/s HBM2e memory bandwidth.
- H100 80GB: 989 dense FP16 TFLOPS, 3350 GB/s HBM3 memory bandwidth.
- CPU reference: illustrative values (2 FP32 TFLOPS, 100 GB/s), not a
  specific part.

Sparsity-multiplied figures (e.g. the 2x structured-sparsity numbers) are
intentionally excluded: this model uses dense tensor-core throughput.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Hardware:
    """A compute device described by its roofline parameters."""

    name: str
    peak_flops: dict = field(default_factory=dict)  # dtype -> dense peak FLOPS
    memory_bw_GBs: float = 0.0  # published DRAM/HBM bandwidth, GB/s
    note: str = ""

    @property
    def memory_bw(self) -> float:
        """Memory bandwidth in bytes/second."""
        return self.memory_bw_GBs * 1e9

    def peak(self, dtype: str = "fp16") -> float:
        """Dense peak FLOPS for the given dtype."""
        if dtype not in self.peak_flops:
            raise KeyError(f"{self.name} has no published peak for dtype '{dtype}'")
        return self.peak_flops[dtype]

    def ridge_point(self, dtype: str = "fp16") -> float:
        """Ridge point (FLOP/byte): intensity above this is compute-bound."""
        return self.peak(dtype) / self.memory_bw


A100 = Hardware(
    name="NVIDIA A100 80GB",
    peak_flops={"fp16": 312e12, "fp32": 19.5e12, "int8": 624e12},
    memory_bw_GBs=2039,
    note=(
        "Published specs: 312 dense FP16 TFLOPS (tensor cores, no sparsity), "
        "19.5 FP32 TFLOPS, 2039 GB/s HBM2e."
    ),
)

H100 = Hardware(
    name="NVIDIA H100 80GB",
    peak_flops={"fp16": 989e12, "fp32": 66.9e12, "int8": 1979e12},
    memory_bw_GBs=3350,
    note=(
        "Published specs: 989 dense FP16 TFLOPS (tensor cores, no sparsity), "
        "66.9 FP32 TFLOPS, 3350 GB/s HBM3."
    ),
)

CPU_REFERENCE = Hardware(
    name="CPU reference (illustrative)",
    peak_flops={"fp32": 2e12, "fp16": 4e12},
    memory_bw_GBs=100,
    note="Illustrative reference values, not a specific part.",
)

ALL_HARDWARE = {"A100": A100, "H100": H100, "CPU": CPU_REFERENCE}
