#!/usr/bin/env python3
"""Smoke tests for 04-disagg-serve.

Run:  python3 test_smoke.py   (from this directory; no pytest required)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))

import numpy as np

from llm_inference_engine.disagg_serve import (
    ColocatedConfig,
    DisaggConfig,
    Request,
    kv_cache_bytes,
    make_workload,
    run_colocated,
    run_disaggregated,
    transfer_delay_s,
)


def test_kv_bytes_formula():
    # 2 * n_layers * d_model * prompt_len * bytes_per_elem (fp16)
    assert kv_cache_bytes(128) == 2 * 32 * 4096 * 128 * 2 == 67108864
    assert kv_cache_bytes(2048) == 2 * 32 * 4096 * 2048 * 2 == 1073741824
    # linear in prompt length
    assert kv_cache_bytes(256) == 2 * kv_cache_bytes(128)


def test_transfer_delay_sanity():
    d = transfer_delay_s(128, 100.0)
    assert d == 67108864 / 1e11, d
    # linear in prompt length, inverse in bandwidth
    assert transfer_delay_s(256, 100.0) == 2 * d
    assert abs(transfer_delay_s(128, 200.0) - d / 2) < 1e-15
    # fixed latency term adds exactly
    assert abs(transfer_delay_s(128, 100.0, latency_ms=1.0) - (d + 0.001)) < 1e-15
    # 2048-token prompt at 5 GB/s is ~215 ms: the regime where the
    # transfer overhead dominates TTFT
    assert 0.20 < transfer_delay_s(2048, 5.0) < 0.23


def test_single_request_colocated_ttft():
    # One idle worker: TTFT must equal exactly one prefill iteration.
    cfg = ColocatedConfig(workers=1)
    res = run_colocated([Request(128, 1, 0.0)], cfg)
    expected_ms = cfg.t0_ms + cfg.k_ms_per_token * 128
    assert abs(res.ttft_ms[0] - expected_ms) < 1e-9, (res.ttft_ms[0], expected_ms)
    assert res.completed == 1


def test_disagg_ttft_covers_transfer():
    # Disaggregated TTFT must at least cover prefill + KV transfer.
    cfg = DisaggConfig()
    r = Request(2048, 8, 0.0)
    res = run_disaggregated([r], cfg)
    floor_s = (2048 / cfg.prefill_tok_per_s
               + transfer_delay_s(2048, cfg.interconnect_gbps,
                                  latency_ms=cfg.transfer_latency_ms))
    assert res.ttft_ms[0] / 1000.0 >= floor_s - 1e-9


def test_determinism():
    for run in (run_colocated, run_disaggregated):
        rng = np.random.default_rng(31)
        w1 = make_workload(rng, 6.0, 120, short_frac=0.5)
        rng = np.random.default_rng(31)
        w2 = make_workload(rng, 6.0, 120, short_frac=0.5)
        a = run(w1)
        b = run(w2)
        assert np.array_equal(a.ttft_ms, b.ttft_ms)
        assert np.array_equal(a.tbt_gaps_ms, b.tbt_gaps_ms)
        assert a.throughput_tok_s == b.throughput_tok_s


def test_conservation():
    # Every request completes with exactly gen_len token times, finite TTFT.
    for run in (run_colocated, run_disaggregated):
        rng = np.random.default_rng(99)
        w = make_workload(rng, 8.0, 60, short_frac=0.3)
        n = len(w)
        res = run(w)
        assert res.completed == n, (res.completed, n)
        assert len(res.ttft_ms) == n
        assert np.all(np.isfinite(res.ttft_ms))
        assert np.all(res.ttft_ms > 0)
        for r in w:
            assert len(r.token_times) == r.gen_len, (len(r.token_times), r.gen_len)
            assert np.isfinite(r.completed_at)
        # token count conservation: throughput * makespan == generated tokens
        assert abs(res.throughput_tok_s * res.makespan_s
                   - sum(r.gen_len for r in w)) < 1e-6


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
