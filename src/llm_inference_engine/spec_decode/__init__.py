"""Speculative decoding simulator (project 03-spec-decode).

Quantifies latency/throughput wins of draft + target speculative sampling
vs. vanilla autoregressive decoding, using an analytical model validated by
Monte Carlo simulation (numpy, CPU only).

Public API
----------
``expected_speedup(k, alpha, c)``
    Expected wall-clock speedup vs vanilla AR, ``E/(1 + k*c)``.
``optimal_k(alpha, c, k_max=8)``
    Draft length maximizing expected speedup; returns ``(best_k,
    best_speedup)``. ``k = 0`` (plain AR) is always a candidate.
``simulate(k, alpha, c, trials=100_000, seed=None)``
    Monte Carlo simulation of the draft/verify loop: returns empirical
    mean/standard error of accepted tokens per cycle and empirical vs.
    analytical speedup.

Lower-level helpers (``expected_accepted_tokens``, ``tokens_per_second``,
``simulate_throughput``) are available in the ``model`` and ``sim``
submodules.
"""

from .model import (
    expected_accepted_tokens,
    expected_speedup,
    optimal_k,
    tokens_per_second,
)
from .sim import simulate, simulate_throughput

__all__ = [
    "expected_speedup",
    "optimal_k",
    "simulate",
    "expected_accepted_tokens",
    "tokens_per_second",
    "simulate_throughput",
]
