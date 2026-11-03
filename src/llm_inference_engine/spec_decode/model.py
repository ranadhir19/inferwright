"""Analytical model for speculative decoding performance.

Models the classic draft + target speculative sampling scheme (Leviathan et
al., 2023; Chen et al., 2023). Each decoding cycle works as follows:

1. A cheap draft model autoregressively proposes ``k`` tokens.
2. The expensive target model verifies the whole draft in ONE forward pass
   (parallel scoring of the k proposed positions, plus one bonus position).
3. Drafted tokens are accepted one-by-one while they "agree" with the target
   distribution; at the first rejection a corrected token is sampled from an
   adjusted distribution, ending the cycle. If all ``k`` drafts are accepted,
   the target's bonus token is emitted as well, so a cycle yields ``k + 1``
   tokens.

Parameters
----------
k : int
    Number of tokens drafted per cycle (draft length).
alpha : float
    Per-token acceptance probability, i.e. how well the draft model matches
    the target model. Higher alpha means a better draft model (or an easier
    task).
c : float
    Cost ratio ``c = (draft cost per token) / (target cost per token)``.
    Smaller c means a much cheaper draft model.

Key results
-----------
Expected accepted tokens per target forward pass::

    E(k, alpha) = (1 - alpha**(k+1)) / (1 - alpha)

This is the mean of a truncated geometric process: you accept tokens until
the first rejection (each rejection resampled into one final token), or all
``k`` drafts plus the bonus token.

Expected wall-clock speedup vs. vanilla autoregressive decoding::

    S(k, alpha, c) = E(k, alpha) / (1 + k * c)

The denominator is the per-cycle cost in units of one target forward pass:
one target pass plus ``k`` draft passes, each costing a fraction ``c`` of a
target pass.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "expected_accepted_tokens",
    "expected_speedup",
    "optimal_k",
    "tokens_per_second",
]


def expected_accepted_tokens(k: int, alpha: float) -> float:
    """Expected number of tokens accepted per target forward pass.

    Parameters
    ----------
    k : int
        Draft length (tokens proposed per cycle), must be >= 0.
    alpha : float
        Per-token acceptance probability in [0, 1].

    Returns
    -------
    float
        ``E(k, alpha) = (1 - alpha**(k+1)) / (1 - alpha)``. Edge cases:
        ``k = 0`` -> ``1.0`` (no drafting, plain AR token); ``alpha -> 1`` ->
        ``k + 1`` (perfect draft accepts everything plus the bonus token);
        ``alpha = 0`` -> ``1.0`` (every draft rejected, one resampled token).
    """
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError(f"k must be a non-negative integer, got {k!r}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
    if k == 0:
        return 1.0
    if alpha == 1.0 or (1.0 - alpha) < 1e-12:
        return float(k + 1)
    return float((1.0 - alpha ** (k + 1)) / (1.0 - alpha))


def expected_speedup(k: int, alpha: float, c: float) -> float:
    """Expected wall-clock speedup of speculative decoding vs vanilla AR.

    ``S = E(k, alpha) / (1 + k * c)``: the expected tokens per cycle divided
    by the per-cycle compute cost measured in target forward passes.
    ``k = 0`` gives exactly ``1.0`` (no speculation = vanilla AR).
    """
    if c < 0.0:
        raise ValueError(f"c must be non-negative, got {c!r}")
    return expected_accepted_tokens(k, alpha) / (1.0 + k * c)


def optimal_k(
    alpha: float,
    c: float,
    k_max: int = 8,
) -> tuple[int, float]:
    """Select the draft length ``k`` in ``[0, k_max]`` maximizing speedup.

    Returns
    -------
    (best_k, best_speedup)
        ``best_k`` is the smallest k attaining the maximum (deterministic
        tie-breaking). Note ``k = 0`` (plain AR, speedup 1.0) is always
        allowed, so the returned speedup is never below 1.0.
    """
    if k_max < 0:
        raise ValueError(f"k_max must be non-negative, got {k_max!r}")
    best_k, best_s = 0, 1.0
    for k in range(1, k_max + 1):
        s = expected_speedup(k, alpha, c)
        if s > best_s:
            best_k, best_s = k, s
    return best_k, best_s


def tokens_per_second(
    target_forward_ms: float,
    c: float,
    k: int,
    alpha: float,
) -> tuple[float, float]:
    """Wall-clock throughput for speculative vs. vanilla AR decoding.

    Parameters
    ----------
    target_forward_ms : float
        Latency of one target-model forward pass in milliseconds.
    c, k, alpha : float, int, float
        Cost ratio, draft length, per-token acceptance probability.

    Returns
    -------
    (speculative_tps, vanilla_tps)
        Tokens per second. Vanilla AR emits one token per target forward
        pass; speculative emits ``E(k, alpha)`` tokens per cycle, where a
        cycle costs ``(1 + k * c)`` target passes.
    """
    if target_forward_ms <= 0.0:
        raise ValueError(f"target_forward_ms must be positive, got {target_forward_ms!r}")
    cycle_ms = target_forward_ms * (1.0 + k * c)
    spec_tps = expected_accepted_tokens(k, alpha) / (cycle_ms / 1000.0)
    vanilla_tps = 1000.0 / target_forward_ms
    return spec_tps, vanilla_tps
