"""Smoke tests for sched_lab: conservation, KV cap, determinism, preemption."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from llm_inference_engine.sched_lab import (  # noqa: E402
    POLICY_NAMES,
    SimConfig,
    add_burst,
    make_workload,
    run_policy,
    summarize,
)

CAP = SimConfig().kv_cap_tokens


def _small_specs(seed=7, n=80, rate=4.0, long_frac=0.2):
    rng = np.random.default_rng(seed)
    return make_workload(rng, n, rate, long_frac=long_frac)


def test_conservation_all_requests_complete():
    """Every request finishes; generated tokens are conserved."""
    for policy in POLICY_NAMES:
        res = run_policy(policy, _small_specs())
        assert res.n == 80
        assert np.all(res.done_t >= 0.0), policy
        assert np.all(res.first_token_t >= 0.0), policy
        # arrival <= first token <= done, per request
        assert np.all(res.first_token_t >= res.arrival * 1000.0), policy
        assert np.all(res.done_t >= res.first_token_t), policy
        assert np.all(res.ttft_ms >= 0.0), policy
        assert np.all(res.tpot_ms > 0.0), policy
        assert res.total_gen_tokens == int(res.gen_len.sum())


def test_kv_cap_never_exceeded():
    """Running KV reservations never exceed the token budget, incl. bursts."""
    rng = np.random.default_rng(11)
    specs = add_burst(
        make_workload(rng, 120, 5.0, long_frac=0.25), t_burst=10.0, count=10
    )
    for policy in POLICY_NAMES:
        res = run_policy(policy, specs)
        assert res.max_kv_used <= CAP, (policy, res.max_kv_used)
        assert res.max_kv_used > 0, policy


def test_determinism_same_seed_same_result():
    """Two runs on identical specs give identical per-request metrics."""
    specs = _small_specs(seed=61, n=150, rate=6.0)
    for policy in POLICY_NAMES:
        r1 = run_policy(policy, specs)
        r2 = run_policy(policy, specs)
        assert np.array_equal(r1.ttft_ms, r2.ttft_ms), policy
        assert np.array_equal(r1.tpot_ms, r2.tpot_ms), policy
        assert r1.makespan_s == r2.makespan_s, policy
        assert r1.preemptions == r2.preemptions, policy
        assert np.array_equal(r1.iter_batch_tokens, r2.iter_batch_tokens), policy


def test_preemption_accounting():
    """priority-preempt preempts under contention; recompute is accounted."""
    rng = np.random.default_rng(61)
    specs = add_burst(
        make_workload(rng, 300, 6.0, long_frac=0.30), t_burst=20.0, count=10
    )
    res = run_policy("priority-preempt", specs)
    assert res.preemptions > 0, "expected preemptions under KV contention"
    assert res.recompute_tokens > 0
    # per-request charges sum to the reported total
    assert res.recompute_tokens == int(res.recompute_charged.sum())
    # preempted requests still complete with all their tokens
    assert np.all(res.done_t >= 0.0)
    assert res.total_gen_tokens == int(res.gen_len.sum())
    # continuous on the same workload never preempts
    res_c = run_policy("continuous", specs)
    assert res_c.preemptions == 0
    assert res_c.recompute_tokens == 0


def test_chunked_prefill_bounded_chunks():
    """chunked-prefill never processes more than chunk_tokens per request."""
    from llm_inference_engine.sched_lab import ChunkedPrefillScheduler, Simulator

    rng = np.random.default_rng(3)
    specs = add_burst(
        make_workload(rng, 60, 4.0, long_frac=0.3), t_burst=5.0, count=6
    )
    sched = ChunkedPrefillScheduler()
    sim = Simulator(
        specs["arrival"], specs["prompt_len"], specs["gen_len"], sched
    )
    cfg_chunk = sim.config.chunk_tokens
    sched.reset(sim)
    # drive a few iterations manually and inspect planned prefill work
    n = len(sim.reqs)
    ai = 0
    for _ in range(400):
        while ai < n and sim._arrivals[ai] <= sim.t:
            sim.waiting.append(sim.reqs[ai])
            ai += 1
        sched.admit(sim)
        work = sched.plan(sim)
        if not work:
            if ai < n:
                sim.t = sim._arrivals[ai]
                continue
            break
        for _req, tok, is_decode in work:
            if not is_decode:
                assert tok <= cfg_chunk, (tok, cfg_chunk)
                assert tok > 0
        bt = sum(t for _, t, _ in work)
        sim.t += sim.config.t0_ms + sim.config.k_ms_per_token * bt
        for req, tok, is_decode in work:
            if is_decode:
                req.generated += 1
                if req.generated >= req.gen_len:
                    sim._finish(req)
            else:
                req.remaining_prompt -= tok
                if req.remaining_prompt <= 0:
                    req.remaining_prompt = 0
                    req.state = "decode"
    assert sim.done > 0


def test_metrics_sane():
    """summarize() produces finite, non-negative headline metrics."""
    for policy in POLICY_NAMES:
        m = summarize(run_policy(policy, _small_specs()))
        for key in (
            "ttft_mean_ms", "ttft_p99_ms", "tpot_mean_ms", "tpot_p99_ms",
            "throughput_tok_s", "gpu_util", "makespan_s",
        ):
            assert np.isfinite(m[key]) and m[key] >= 0.0, (policy, key, m[key])
        assert m["ttft_p99_ms"] >= m["ttft_mean_ms"] - 1e-9, policy


if __name__ == "__main__":
    test_conservation_all_requests_complete()
    print("conservation ok")
    test_kv_cap_never_exceeded()
    print("kv cap ok")
    test_determinism_same_seed_same_result()
    print("determinism ok")
    test_preemption_accounting()
    print("preemption accounting ok")
    test_chunked_prefill_bounded_chunks()
    print("chunk bound ok")
    test_metrics_sane()
    print("metrics sane ok")
    print("ALL SMOKE TESTS PASSED")
