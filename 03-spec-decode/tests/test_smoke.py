"""Smoke tests for 03-spec-decode.

Covers: formula edge cases (k=0, alpha->1, alpha=0), the optimal-k selector,
Monte Carlo convergence to the formula within tolerance, and determinism of
the seeded simulator.

Run with:  python3 -m pytest 03-spec-decode/tests/test_smoke.py -q
"""

import os
import sys

import numpy as np

try:
    import pytest
except ImportError:  # minimal shim so the suite runs without pytest installed
    from contextlib import contextmanager

    class _Approx:
        def __init__(self, expected, rel=None, abs=None):
            self.expected, self.rel, self.abs = expected, rel, abs
        def __eq__(self, actual):
            tol = self.abs if self.abs is not None else abs(self.expected) * (self.rel or 1e-12)
            return abs(actual - self.expected) <= tol

    @contextmanager
    def _raises(exc):
        try:
            yield
        except exc:
            return
        raise AssertionError(f"did not raise {exc}")

    pytest = type("pytest", (), {"approx": _Approx, "raises": _raises})

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from llm_inference_engine.spec_decode import (  # noqa: E402
    expected_accepted_tokens,
    expected_speedup,
    optimal_k,
    simulate,
)


# ---- formula edge cases ----------------------------------------------------
def test_k_zero_means_no_speculation():
    assert expected_accepted_tokens(0, 0.7) == 1.0
    assert expected_speedup(0, 0.7, 0.1) == 1.0
    assert expected_speedup(0, 0.95, 0.0) == 1.0


def test_alpha_one_limit():
    # As alpha -> 1, E -> k + 1 (all drafts accepted plus the bonus token).
    assert expected_accepted_tokens(8, 1.0) == 9.0
    assert expected_accepted_tokens(5, 1.0 - 1e-15) == pytest.approx(6.0, rel=1e-6)
    assert expected_accepted_tokens(3, 0.999999) == pytest.approx(4.0, abs=1e-3)


def test_alpha_zero_limit():
    # Every draft rejected: exactly the one resampled token.
    assert expected_accepted_tokens(8, 0.0) == pytest.approx(1.0)
    assert expected_accepted_tokens(1, 0.0) == pytest.approx(1.0)


def test_known_formula_value():
    # E(4, 0.7) = (1 - 0.7^5) / 0.3
    assert expected_accepted_tokens(4, 0.7) == pytest.approx(
        (1 - 0.7**5) / 0.3, rel=1e-12
    )
    # S = E / (1 + k*c)
    assert expected_speedup(4, 0.7, 0.1) == pytest.approx(
        expected_accepted_tokens(4, 0.7) / 1.4, rel=1e-12
    )


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        expected_accepted_tokens(-1, 0.5)
    with pytest.raises(ValueError):
        expected_accepted_tokens(4, 1.5)
    with pytest.raises(ValueError):
        expected_speedup(4, 0.7, -0.1)


# ---- optimal-k selector ----------------------------------------------------
def test_optimal_k_never_below_vanilla():
    for alpha in [0.0, 0.2, 0.5, 0.9]:
        for c in [0.05, 0.2, 1.0]:
            k, s = optimal_k(alpha, c, k_max=8)
            assert s >= 1.0  # k=0 (plain AR) is always available


def test_optimal_k_monotone_in_alpha():
    # Better draft models justify longer drafts: optimal k is
    # non-decreasing in alpha for fixed c.
    ks = [optimal_k(a, 0.1, k_max=12)[0]
          for a in np.linspace(0.05, 0.99, 25)]
    assert all(b >= a for a, b in zip(ks, ks[1:]))


def test_optimal_k_high_when_draft_cheap_and_good():
    k, s = optimal_k(0.95, 0.05, k_max=8)
    assert k == 8 and s > 5.0


# ---- Monte Carlo convergence -----------------------------------------------
def test_mc_converges_to_formula():
    for k, alpha, c in [(1, 0.5, 0.1), (4, 0.7, 0.1), (8, 0.95, 0.05),
                        (8, 0.3, 0.2)]:
        res = simulate(k, alpha, c, trials=200_000, seed=21)
        # ~5 standard errors of slack: 200k trials -> se ~ 0.004-0.007
        assert abs(res["mean_accepted_tokens"] - res["analytical_accepted"]) < 0.05
        assert abs(res["empirical_speedup"] - res["analytical_speedup"]) < 0.05


def test_mc_standard_error_consistent():
    res = simulate(4, 0.7, 0.1, trials=100_000, seed=21)
    assert res["se_accepted_tokens"] == pytest.approx(
        res["std_accepted_tokens"] / np.sqrt(100_000), rel=1e-12
    )


# ---- determinism ------------------------------------------------------------
def test_simulate_deterministic():
    r1 = simulate(4, 0.7, 0.1, trials=50_000, seed=21)
    r2 = simulate(4, 0.7, 0.1, trials=50_000, seed=21)
    assert r1["mean_accepted_tokens"] == r2["mean_accepted_tokens"]


def test_different_seeds_differ():
    r1 = simulate(4, 0.7, 0.1, trials=10_000, seed=1)
    r2 = simulate(4, 0.7, 0.1, trials=10_000, seed=2)
    assert r1["mean_accepted_tokens"] != r2["mean_accepted_tokens"]


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for k, v in fns:
        v()
    print(f"{len(fns)} tests passed")
