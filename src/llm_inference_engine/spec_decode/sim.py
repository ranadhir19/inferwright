"""Monte Carlo simulator for the speculative decoding draft/verify loop.

The simulator implements the token-level geometric acceptance process
directly, so the analytical formula in :mod:`.model` can be validated
empirically:

* The draft model proposes ``k`` tokens for one cycle.
* Each proposed token is independently accepted with probability ``alpha``.
* If the first rejection happens at draft position ``j`` (0-indexed), the
  cycle yields ``j`` accepted drafts plus one corrected (resampled) token:
  ``j + 1`` tokens in total.
* If all ``k`` drafts are accepted, the target model's bonus token is
  emitted: ``k + 1`` tokens in total.

Runs are vectorized over trials and seeded via ``numpy.random.Generator``
for reproducibility.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .model import expected_accepted_tokens, expected_speedup

__all__ = ["simulate", "simulate_throughput"]


def _accepted_counts(k: int, alpha: float, trials: int, rng: np.random.Generator) -> np.ndarray:
    """Accepted-token count per cycle for ``trials`` independent cycles."""
    if k == 0:
        return np.ones(trials, dtype=np.float64)
    # draws < alpha  <=>  draft token accepted at that position.
    rejected = rng.random((trials, k)) >= alpha
    any_rejected = rejected.any(axis=1)
    first_rejection = rejected.argmax(axis=1)  # 0 when no rejection; masked below
    return np.where(any_rejected, first_rejection + 1, k + 1).astype(np.float64)


def simulate(
    k: int,
    alpha: float,
    c: float,
    trials: int = 100_000,
    seed: Optional[int] = None,
) -> dict:
    """Monte Carlo estimate of per-cycle accepted tokens and speedup.

    Parameters
    ----------
    k, alpha, c : int, float, float
        Draft length, per-token acceptance probability, draft/target cost
        ratio (see :mod:`.model`).
    trials : int
        Number of independent draft/verify cycles to simulate.
    seed : int or None
        Seed for ``numpy.random.default_rng``; fixed seed -> identical result.

    Returns
    -------
    dict with keys:
        ``mean_accepted_tokens``  empirical mean tokens per cycle
        ``std_accepted_tokens``   empirical std of tokens per cycle
        ``se_accepted_tokens``    standard error of the mean
        ``analytical_accepted``   ``expected_accepted_tokens(k, alpha)``
        ``empirical_speedup``     empirical speedup vs vanilla AR
        ``analytical_speedup``    ``expected_speedup(k, alpha, c)``
        ``trials``, ``k``, ``alpha``, ``c``  echo of inputs
    """
    rng = np.random.default_rng(seed)
    counts = _accepted_counts(k, alpha, trials, rng)
    mean = float(counts.mean())
    std = float(counts.std(ddof=1)) if trials > 1 else 0.0
    return {
        "mean_accepted_tokens": mean,
        "std_accepted_tokens": std,
        "se_accepted_tokens": std / np.sqrt(trials) if trials > 0 else 0.0,
        "analytical_accepted": expected_accepted_tokens(k, alpha),
        "empirical_speedup": mean / (1.0 + k * c),
        "analytical_speedup": expected_speedup(k, alpha, c),
        "trials": trials,
        "k": k,
        "alpha": alpha,
        "c": c,
    }


def simulate_throughput(
    k: int,
    alpha: float,
    c: float,
    target_forward_ms: float,
    trials: int = 100_000,
    seed: Optional[int] = None,
) -> dict:
    """Monte Carlo wall-clock throughput (tokens/sec) for a scenario.

    Each simulated cycle costs ``target_forward_ms * (1 + k * c)`` ms and
    yields a random number of accepted tokens; throughput is total accepted
    tokens over total elapsed time. Also reports the vanilla AR throughput
    (one token per target forward pass) and the speedup ratio.
    """
    rng = np.random.default_rng(seed)
    counts = _accepted_counts(k, alpha, trials, rng)
    total_tokens = float(counts.sum())
    total_ms = trials * target_forward_ms * (1.0 + k * c)
    spec_tps = total_tokens / (total_ms / 1000.0)
    vanilla_tps = 1000.0 / target_forward_ms
    return {
        "speculative_tps": spec_tps,
        "vanilla_tps": vanilla_tps,
        "speedup": spec_tps / vanilla_tps,
        "analytical_speedup": expected_speedup(k, alpha, c),
        "trials": trials,
    }
