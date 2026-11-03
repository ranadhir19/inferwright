"""Synthetic request workload for the allocator simulations.

A workload is a list of :class:`Request` objects. Sequence lengths follow a
three-component mixture (short / medium / long), clipped to
``[1, max_len]``; arrivals are spread uniformly over ``arrival_window``
time units; each sequence completes ``length`` time units after arrival
(the analytical model decodes 1 token per time unit, so a sequence's
KV-cache lifetime equals its length).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Request:
    seq_id: int
    arrival: float
    length: int
    departure: float


def gen_workload(
    rng: np.random.Generator,
    n: int = 2000,
    arrival_window: float = 600.0,
    max_len: int = 2048,
    short: tuple[float, float, float] = (0.70, 128.0, 32.0),
    medium: tuple[float, float, float] = (0.20, 512.0, 128.0),
    long: tuple[float, float, float] = (0.10, 2048.0, 256.0),
) -> list[Request]:
    """Generate ``n`` requests with a mixed length distribution.

    Each tuple is ``(mixture_weight, mean, std)`` for a Gaussian component.
    Lengths are clipped to ``[1, max_len]`` and rounded to integers.
    Arrivals are uniform on ``[0, arrival_window]`` and sorted so the list
    is in arrival order; ``departure = arrival + length``.
    """
    weights = np.array([short[0], medium[0], long[0]])
    weights = weights / weights.sum()
    comp = rng.choice(3, size=n, p=weights)
    means = np.array([short[1], medium[1], long[1]])
    stds = np.array([short[2], medium[2], long[2]])
    lengths = rng.normal(means[comp], stds[comp])
    lengths = np.clip(lengths, 1, max_len).astype(int)

    arrivals = np.sort(rng.uniform(0.0, arrival_window, size=n))
    departures = arrivals + lengths.astype(float)

    return [
        Request(seq_id=i, arrival=float(a), length=int(L), departure=float(d))
        for i, (a, L, d) in enumerate(zip(arrivals, lengths, departures))
    ]
