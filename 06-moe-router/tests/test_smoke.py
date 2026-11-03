"""Smoke tests for the moe_router module (stdlib + numpy only).

Run:  python 06-moe-router/tests/test_smoke.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from llm_inference_engine.moe_router import (
    alltoall_bytes,
    balanced_gate,
    device_load,
    noisy_gate,
    route,
    shard_experts,
    topk_gate,
)


def _skewed_logits(n_tokens=256, n_experts=8, seed=7):
    rng = np.random.default_rng(seed)
    bias = 2.0 * np.log(np.array([0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05]))
    return bias[None, :] + rng.normal(0.0, 1.0, size=(n_tokens, n_experts))


def test_conservation():
    """routed + dropped == total token->expert assignments."""
    logits = _skewed_logits()
    for top_k in (1, 2):
        eids, w = topk_gate(logits, top_k)
        s = route(eids, w, n_experts=8, capacity_factor=1.0)
        total = s["n_tokens"] * s["top_k"]
        assert s["routed_assignments"] + s["dropped_assignments"] == total, (
            f"top_k={top_k}: {s['routed_assignments']} + {s['dropped_assignments']} != {total}"
        )
        assert 0.0 <= s["drop_rate"] <= 1.0


def test_capacity_respected():
    """No expert receives more tokens than its capacity."""
    logits = _skewed_logits()
    for top_k in (1, 2):
        for cf in (0.5, 1.0, 2.0):
            eids, w = topk_gate(logits, top_k)
            s = route(eids, w, n_experts=8, capacity_factor=cf)
            assert (s["per_expert_counts"] <= s["expert_capacity"]).all(), (
                f"top_k={top_k} cf={cf}: capacity {s['expert_capacity']} exceeded: "
                f"{s['per_expert_counts']}"
            )
            # capacity formula itself
            assert s["expert_capacity"] == int(np.ceil(cf * 256 / 8))


def test_determinism():
    """Same seed -> identical routing, gating, and stats."""
    for seed in (51, 52):
        a = _skewed_logits(seed=seed)
        b = _skewed_logits(seed=seed)
        e1, w1 = topk_gate(a, 2)
        e2, w2 = topk_gate(b, 2)
        assert np.array_equal(e1, e2) and np.allclose(w1, w2)
        s1 = route(e1, w1, 8, 1.0)
        s2 = route(e2, w2, 8, 1.0)
        assert s1["drop_rate"] == s2["drop_rate"]
        assert np.array_equal(s1["per_expert_counts"], s2["per_expert_counts"])
    # noisy / balanced gates are deterministic given the same rng seed
    logits = _skewed_logits()
    e1, _ = noisy_gate(logits, 2, rng=np.random.default_rng(3))
    e2, _ = noisy_gate(logits, 2, rng=np.random.default_rng(3))
    assert np.array_equal(e1, e2)
    e1, _ = balanced_gate(logits, 2, rng=np.random.default_rng(3), noise_scale=0.1)
    e2, _ = balanced_gate(logits, 2, rng=np.random.default_rng(3), noise_scale=0.1)
    assert np.array_equal(e1, e2)


def test_gating_weights_renormalized():
    """Top-k weights sum to 1 per token and pick the argmax first for top-1."""
    logits = _skewed_logits()
    eids, w = topk_gate(logits, 2)
    assert np.allclose(w.sum(axis=1), 1.0)
    eids1, _ = topk_gate(logits, 1)
    assert np.array_equal(eids1[:, 0], np.argmax(logits, axis=1))


def test_parallel_math():
    """all-to-all formula and device sharding sanity."""
    assert alltoall_bytes(4096, 2, 4096, 2) == 4096 * 2 * 4096 * 2 * 2
    assign = shard_experts(16, 4)
    assert len(assign) == 16
    assert sorted(set(assign.tolist())) == [0, 1, 2, 3]
    assert (np.bincount(assign) == 4).all()
    load = device_load(4096, 2, 16, 8, d_model=4096, bytes_per_elem=2)
    assert load["experts_per_device"] == 2
    assert load["token_slots_per_device"] == 4096 * 2 / 8
    assert load["comm_bytes_per_device"] == alltoall_bytes(4096, 2, 4096, 2) / 8


def test_balanced_reduces_drops():
    """Idealized aux-loss balancing must not increase drops at CF=1.0, top-1."""
    rng = np.random.default_rng(51)
    logits = _skewed_logits()
    eids, w = topk_gate(logits, 1)
    skewed = route(eids, w, 8, 1.0)
    eids_b, w_b = balanced_gate(logits, 1, balance_strength=0.99,
                                noise_scale=0.1, rng=rng)
    balanced = route(eids_b, w_b, 8, 1.0)
    assert balanced["drop_rate"] <= skewed["drop_rate"]
    assert balanced["imbalance_ratio"] <= skewed["imbalance_ratio"]


def main():
    tests = [
        test_conservation,
        test_capacity_respected,
        test_determinism,
        test_gating_weights_renormalized,
        test_parallel_math,
        test_balanced_reduces_drops,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"OK: {len(tests)}/{len(tests)} smoke tests passed")


if __name__ == "__main__":
    main()
