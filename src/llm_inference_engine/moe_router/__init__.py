"""Mixture-of-experts (MoE) expert-routing simulator.

Simulates the three forces that decide whether an MoE layer is fast or
wasteful:

- **Gating** (``gating.py``): softmax top-k routing from token->expert
  logits, with noisy-gating and load-balancing-loss-idealized variants.
- **Dispatch** (``router.py``): per-expert capacity from the capacity
  factor CF; tokens beyond capacity are *dropped* and counted.
- **Expert parallelism** (``parallel.py``): all-to-all communication
  volume and per-device compute load when experts are sharded across
  devices.

Public API
----------
- :func:`topk_gate` — softmax top-k gating, renormalized weights
- :func:`noisy_gate` — gating with perturbed logits (imperfect balance)
- :func:`balanced_gate` — gating with skew reduced, modeling an aux
  load-balancing loss
- :func:`route` — capacity-limited dispatch with drop statistics
- :func:`shard_experts` — expert -> device assignment
- :func:`alltoall_bytes` — all-to-all bytes per MoE step
- :func:`device_load` — per-device compute/comm breakdown
"""

from .gating import balanced_gate, noisy_gate, softmax, topk_gate
from .parallel import alltoall_bytes, device_load, shard_experts
from .router import route

__all__ = [
    "topk_gate",
    "noisy_gate",
    "balanced_gate",
    "softmax",
    "route",
    "shard_experts",
    "alltoall_bytes",
    "device_load",
]
