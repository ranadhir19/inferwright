"""LLM Inference Engine — advanced LLM inference systems from first principles.

Ten self-contained, seeded, reproducible projects (CPU/numpy only):

- ``transformer``    — LLaMA-style decoder inference from scratch
  (RMSNorm, RoPE, grouped-query attention, SwiGLU, KV cache)
- ``paged_kv``       — vLLM-style paged KV-cache block manager simulator
- ``spec_decode``    — speculative decoding simulator (draft + target)
- ``disagg_serve``   — disaggregated prefill/decode serving simulator
- ``quant_lab``      — weight quantization workbench (INT8/FP8/INT4/FP4)
- ``moe_router``     — mixture-of-experts routing simulator
- ``roofline``       — roofline performance model for inference ops
- ``sched_lab``      — advanced request scheduling simulator
- ``remit_drift``    — divergence of remit in multi-agent systems:
  remit as a first-class object, 8 drift types, policy engine
- ``llm_mesh``       — LLM mesh: model registry, semantic router,
  orchestrator with parallel dispatch, UCB1 self-improving routing

All benchmarks are simulation-based with seeded RNG: they quantify
architectural and algorithmic tradeoffs, not production GPU behavior.
"""

__version__ = "0.1.0"

__all__ = [
    "transformer",
    "paged_kv",
    "spec_decode",
    "disagg_serve",
    "quant_lab",
    "moe_router",
    "roofline",
    "sched_lab",
    "remit_drift",
    "llm_mesh",
]

from llm_inference_engine import (
    disagg_serve,
    llm_mesh,
    moe_router,
    paged_kv,
    quant_lab,
    remit_drift,
    roofline,
    sched_lab,
    spec_decode,
    transformer,
)
