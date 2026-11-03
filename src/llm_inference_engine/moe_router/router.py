"""Token dispatch with expert capacity limits and dropping.

In production MoE systems (GShard, Switch, DeepSpeed-MoE) each expert has
a fixed *capacity* per step derived from the capacity factor CF::

    expert_capacity = ceil(CF * n_tokens / n_experts)

Tokens routed to an expert that is already full are *dropped* — they skip
the expert computation entirely (their residual stream passes through).
This module simulates that dispatch and counts the dropped tokens: the
hidden tax of MoE routing.
"""

from __future__ import annotations

import numpy as np


def route(
    expert_ids: np.ndarray,
    weights: np.ndarray,
    n_experts: int,
    capacity_factor: float,
) -> dict:
    """Dispatch tokens to experts, dropping assignments beyond capacity.

    Tokens are processed in index order; for each token, its ``top_k``
    expert assignments (ranked best-first) are attempted in order. An
    assignment is *routed* if the expert still has free capacity and
    *dropped* otherwise.

    Parameters
    ----------
    expert_ids:
        Integer array from a gating function, shape ``(n_tokens, top_k)``.
    weights:
        Renormalized gating weights, same shape (kept for API symmetry;
        dispatch does not depend on them).
    n_experts:
        Total number of experts.
    capacity_factor:
        Capacity factor CF; ``expert_capacity = ceil(CF * n_tokens / n_experts)``.

    Returns
    -------
    dict with:
        - ``n_tokens``, ``top_k``, ``n_experts``, ``capacity_factor``,
          ``expert_capacity``
        - ``per_expert_counts``: routed token counts per expert
        - ``routed_assignments``, ``dropped_assignments``
        - ``drop_rate``: fraction of token->expert assignments dropped
        - ``imbalance_ratio``: max/mean of per-expert routed counts
          (1.0 = perfectly balanced)
        - ``processed_tokens``: tokens with at least one surviving
          assignment (effective throughput proxy)
    """
    expert_ids = np.asarray(expert_ids, dtype=np.int64)
    if expert_ids.ndim != 2:
        raise ValueError(f"expert_ids must be 2D, got shape {expert_ids.shape}")
    if capacity_factor <= 0:
        raise ValueError(f"capacity_factor must be positive, got {capacity_factor}")
    n_tokens, top_k = expert_ids.shape
    capacity = int(np.ceil(capacity_factor * n_tokens / n_experts))

    counts = np.zeros(n_experts, dtype=np.int64)
    routed = 0
    dropped = 0
    token_survived = np.zeros(n_tokens, dtype=bool)

    for t in range(n_tokens):
        for e in expert_ids[t]:
            if not 0 <= e < n_experts:
                raise ValueError(f"expert id {e} out of range [0, {n_experts})")
            if counts[e] < capacity:
                counts[e] += 1
                routed += 1
                token_survived[t] = True
            else:
                dropped += 1

    total = n_tokens * top_k
    mean = counts.mean()
    return {
        "n_tokens": n_tokens,
        "top_k": top_k,
        "n_experts": n_experts,
        "capacity_factor": capacity_factor,
        "expert_capacity": capacity,
        "per_expert_counts": counts,
        "routed_assignments": routed,
        "dropped_assignments": dropped,
        "drop_rate": dropped / total,
        "imbalance_ratio": float(counts.max() / mean) if mean > 0 else 0.0,
        "processed_tokens": int(token_survived.sum()),
    }
