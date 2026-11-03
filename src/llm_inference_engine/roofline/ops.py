"""Per-op FLOP and byte accounting for one transformer decoder layer.

Conventions (standard roofline practice for GEMM-dominated layers):

- A multiply-add counts as 2 FLOPs.
- "bytes" is the traffic between DRAM/HBM and the chip: weight reads,
  activation reads/writes, and KV-cache reads. On-chip reuse (registers,
  shared memory) is assumed ideal, so this is a lower bound on traffic.
- In decode mode a single query token (q_len=1) reads the full KV cache
  of length S. In prefill mode the full SxS attention matrix is processed.

Layer is LLaMA-style: QKV proj, multi-head attention, output proj, and a
SwiGLU FFN (gate, up, down) with hidden dim = ffn_mult * d_model.
RMSNorm/RoPE/residuals are excluded (sub-percent of step time at these
shapes; noted rather than silently folded in).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerConfig:
    d_model: int
    n_heads: int
    ffn_mult: float
    batch: int
    seq: int
    dtype_bytes: int = 2  # fp16

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_hidden(self) -> int:
        return int(self.ffn_mult * self.d_model)


@dataclass(frozen=True)
class OpStats:
    """FLOP/byte accounting for one op. intensity = flops / bytes (FLOP/byte)."""

    name: str
    flops: float
    bytes: float
    weight_bytes: float = 0.0  # subset of `bytes` that is parameter reads

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes if self.bytes > 0 else float("inf")

    @property
    def activation_bytes(self) -> float:
        return self.bytes - self.weight_bytes


def layer_ops(cfg: LayerConfig, mode: str) -> list[OpStats]:
    """Return per-op stats for one decoder layer.

    mode="decode": one query token, KV cache of length cfg.seq is read.
    mode="prefill": full sequence processed, SxS score matrix materialized.
    """
    if mode not in ("decode", "prefill"):
        raise ValueError(f"mode must be 'decode' or 'prefill', got {mode!r}")

    B, S, d, H = cfg.batch, cfg.seq, cfg.d_model, cfg.n_heads
    h = cfg.ffn_hidden
    b = cfg.dtype_bytes
    q = 1 if mode == "decode" else S      # query length
    kv = S                                 # key/value length

    ops = [
        OpStats(
            name="qkv_proj",
            flops=6 * B * q * d * d,
            bytes=(3 * d * d + 4 * B * q * d) * b,
            weight_bytes=3 * d * d * b,
        ),
        OpStats(
            name="attn_qk",
            flops=2 * B * q * kv * d,
            bytes=(B * q * d + B * kv * d + B * H * q * kv) * b,
            weight_bytes=0.0,
        ),
        OpStats(
            name="softmax",
            # ~5 FLOPs per score element (max-subtract, exp, sum, normalize)
            flops=5 * B * H * q * kv,
            bytes=2 * B * H * q * kv * b,
            weight_bytes=0.0,
        ),
        OpStats(
            name="attn_av",
            flops=2 * B * q * kv * d,
            bytes=(B * H * q * kv + B * kv * d + B * q * d) * b,
            weight_bytes=0.0,
        ),
        OpStats(
            name="out_proj",
            flops=2 * B * q * d * d,
            bytes=(d * d + 2 * B * q * d) * b,
            weight_bytes=d * d * b,
        ),
        OpStats(
            name="ffn_gate_up",
            flops=4 * B * q * d * h,
            bytes=(2 * d * h + B * q * d + 2 * B * q * h) * b,
            weight_bytes=2 * d * h * b,
        ),
        OpStats(
            name="ffn_down",
            flops=2 * B * q * h * d,
            bytes=(h * d + B * q * h + B * q * d) * b,
            weight_bytes=h * d * b,
        ),
    ]
    return ops
