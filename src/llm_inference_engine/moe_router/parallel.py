"""Expert-parallel (EP) sharding and all-to-all communication cost model.

With expert parallelism, the E experts are sharded across D devices. Every
step requires an *all-to-all* exchange: each device sends the tokens (and
gating weights) destined for other devices' experts, receives its own,
computes, then a second all-to-all combines the results back. Hence the
factor of 2 (dispatch + combine) in :func:`alltoall_bytes`.
"""

from __future__ import annotations

import numpy as np


def shard_experts(n_experts: int, n_devices: int) -> np.ndarray:
    """Assign each expert to a device (contiguous shards).

    Returns an integer array of length ``n_experts`` with the device id
    of each expert. Raises if ``n_experts`` is not divisible by
    ``n_devices``.
    """
    if n_devices < 1:
        raise ValueError(f"n_devices must be >= 1, got {n_devices}")
    if n_experts % n_devices != 0:
        raise ValueError(
            f"n_experts ({n_experts}) must be divisible by n_devices ({n_devices})"
        )
    return np.repeat(np.arange(n_devices), n_experts // n_devices)


def alltoall_bytes(
    n_tokens: int,
    top_k: int,
    d_model: int,
    bytes_per_elem: int = 2,
) -> int:
    """Total all-to-all communication volume per MoE step, in bytes.

    Each token is sent to ``top_k`` experts as a ``d_model``-dimensional
    vector (dispatch), and the combined result comes back (combine):

        n_tokens * top_k * d_model * bytes_per_elem * 2
    """
    return int(n_tokens * top_k * d_model * bytes_per_elem * 2)


def device_load(
    n_tokens: int,
    top_k: int,
    n_experts: int,
    n_devices: int,
    d_model: int = 4096,
    bytes_per_elem: int = 2,
) -> dict:
    """Per-device compute load and communication share under expert parallelism.

    Assumes the ``n_tokens * top_k`` token->expert assignments are spread
    evenly across devices (idealized balanced case). Returns:

    - ``experts_per_device``
    - ``token_slots_per_device``: assignments each device must compute
    - ``comm_bytes_per_device``: device's share of the all-to-all volume
    - ``comm_bytes_per_slot``: communication bytes per compute assignment
      — the comm-vs-compute ratio, which grows with ``n_devices``
    """
    assignment = shard_experts(n_experts, n_devices)
    experts_per_device = int(n_experts / n_devices)
    total_slots = n_tokens * top_k
    per_device_slots = total_slots / n_devices
    total_comm = alltoall_bytes(n_tokens, top_k, d_model, bytes_per_elem)
    return {
        "n_devices": n_devices,
        "experts_per_device": experts_per_device,
        "expert_to_device": assignment,
        "token_slots_per_device": per_device_slots,
        "comm_bytes_per_device": total_comm / n_devices,
        "comm_bytes_per_slot": (total_comm / n_devices) / per_device_slots,
    }
