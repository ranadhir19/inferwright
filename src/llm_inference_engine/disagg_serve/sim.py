"""Discrete-event simulator for colocated vs disaggregated LLM serving.

Model
-----
A *request* is ``(prompt_len, gen_len, arrival_time)``.  Workers run
iteration-based continuous batching.  One iteration on a worker that
processes ``batch_tokens`` takes

    t = T0 + K * batch_tokens        (seconds; T0/K given in ms)

and a worker may hold at most ``kv_cap_tokens`` of KV cache
(``prompt_len + gen_len`` tokens per resident request).  Prefill is not
chunked: a request's whole prompt is processed in a single iteration.

Colocated worker: each iteration admits as many queued requests' prefills as
fit in the KV cap, then produces one decode token for every resident decode
request.  A long prefill therefore inflates the iteration time for every
request currently decoding on that worker (interference), and a new request
waits until its worker picks it up (head-of-line queueing).

Disaggregated: prefill workers are compute-optimized (each handles one
request at a time at ``prefill_tok_per_s`` tokens/s).  After prefill, the
prompt KV cache of

    bytes = 2 * n_layers * d_model * prompt_len * bytes_per_elem

is transferred over an interconnect of ``interconnect_gbps`` GB/s, adding a
transfer delay before the request joins a decode worker's queue.  Decode
workers run decode-only iterations ``t = T0d + Kd * batch_size``.

All randomness comes from the caller's ``numpy.random.Generator`` (workload
generation only); the simulation itself is fully deterministic, so identical
inputs give bit-identical outputs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Model constants (defaults used when the caller does not override them)
# --------------------------------------------------------------------------
N_LAYERS = 32
D_MODEL = 4096
BYTES_PER_ELEM = 2  # fp16


def kv_cache_bytes(prompt_len, n_layers=N_LAYERS, d_model=D_MODEL,
                   bytes_per_elem=BYTES_PER_ELEM):
    """Bytes of the prompt KV cache: 2 * n_layers * d_model * prompt_len * bpe.

    Factor of 2 accounts for the K and V tensors.
    """
    return 2 * int(n_layers) * int(d_model) * int(prompt_len) * int(bytes_per_elem)


def transfer_delay_s(prompt_len, bandwidth_gbps, n_layers=N_LAYERS,
                     d_model=D_MODEL, bytes_per_elem=BYTES_PER_ELEM,
                     latency_ms=0.0):
    """Seconds to move the prompt KV cache over an interconnect.

    ``latency_ms`` is a fixed per-transfer protocol/synchronization overhead;
    the variable part is ``kv_bytes / bandwidth``.
    """
    bw_bytes_per_s = float(bandwidth_gbps) * 1e9
    return (kv_cache_bytes(prompt_len, n_layers, d_model, bytes_per_elem)
            / bw_bytes_per_s + float(latency_ms) / 1000.0)


# --------------------------------------------------------------------------
# Configs / data structures
# --------------------------------------------------------------------------
@dataclass
class ColocatedConfig:
    workers: int = 4
    t0_ms: float = 2.0            # fixed per-iteration overhead
    k_ms_per_token: float = 0.009  # per-token cost in an iteration
    kv_cap_tokens: int = 8192      # KV-token capacity per worker


@dataclass
class DisaggConfig:
    prefill_workers: int = 1
    prefill_tok_per_s: float = 800_000.0  # compute-optimized prefill pool
    decode_workers: int = 3
    decode_t0_ms: float = 1.5
    decode_k_ms_per_token: float = 0.002
    decode_kv_cap_tokens: int = 8192
    interconnect_gbps: float = 200.0
    transfer_latency_ms: float = 1.0  # fixed per-transfer sync overhead
    n_layers: int = N_LAYERS
    d_model: int = D_MODEL
    bytes_per_elem: int = BYTES_PER_ELEM


class Request:
    """One serving request.  Mutable simulation scratch fields are private."""

    __slots__ = ("prompt_len", "gen_len", "arrival", "ttft_s",
                 "token_times", "completed_at", "_rem", "_kv", "_prefill_done")

    def __init__(self, prompt_len, gen_len, arrival):
        self.prompt_len = int(prompt_len)
        self.gen_len = int(gen_len)
        self.arrival = float(arrival)
        self.ttft_s = float("nan")
        self.token_times = []
        self.completed_at = float("nan")
        self._rem = 0
        self._kv = self.prompt_len + self.gen_len


@dataclass
class RunResult:
    ttft_ms: np.ndarray          # per-request time-to-first-token
    tbt_gaps_ms: np.ndarray      # all inter-token gaps across all requests
    throughput_tok_s: float      # generated tokens per second of makespan
    utilization: float           # mean worker busy fraction over makespan
    completed: int
    makespan_s: float
    per_req_tbt_mean_ms: np.ndarray = field(default=None)


# --------------------------------------------------------------------------
# Workload
# --------------------------------------------------------------------------
def make_workload(rng, rate_per_s, n_requests, short_frac=0.5,
                  short_len=128, long_len=2048,
                  gen_mean=256.0, gen_std=64.0):
    """Poisson arrivals; prompt mix short/long; gen_len ~ Normal(gen_mean, gen_std).

    Returns requests sorted by arrival time.
    """
    inter = rng.exponential(1.0 / float(rate_per_s), size=n_requests)
    arrivals = np.cumsum(inter)
    is_short = rng.random(n_requests) < short_frac
    prompts = np.where(is_short, short_len, long_len).astype(int)
    gens = np.clip(np.rint(rng.normal(gen_mean, gen_std, n_requests)).astype(int), 1, None)
    return [Request(int(p), int(g), float(a))
            for p, g, a in zip(prompts, gens, arrivals)]


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _assign_jsq(items, n_workers):
    """Join-the-shortest-queue: deterministic assignment by item order."""
    loads = [0] * n_workers
    buckets = [[] for _ in range(n_workers)]
    for it in items:
        w = 0
        for j in range(1, n_workers):
            if loads[j] < loads[w]:
                w = j
        buckets[w].append(it)
        loads[w] += 1
    return buckets


def _gather(requests):
    """Collect metrics from completed requests."""
    ttfts, gaps, per_req = [], [], []
    gen_tokens = 0
    t_end = 0.0
    for r in requests:
        ttfts.append(r.ttft_s * 1000.0)
        ts = r.token_times
        gen_tokens += len(ts)
        if len(ts) > 1:
            g = np.diff(np.asarray(ts)) * 1000.0
            gaps.extend(g.tolist())
            per_req.append(float(np.mean(g)))
        else:
            per_req.append(float("nan"))
        t_end = max(t_end, r.completed_at)
    return ttfts, gaps, per_req, gen_tokens, t_end


def _finish(ttfts, gaps, per_req, gen_tokens, t_end, t_start, busy_times, completed, n):
    assert completed == n, f"lost requests: {completed}/{n}"
    makespan = max(t_end - t_start, 1e-12)
    util = float(np.mean([b / makespan for b in busy_times]))
    return RunResult(
        ttft_ms=np.asarray(ttfts),
        tbt_gaps_ms=np.asarray(gaps),
        throughput_tok_s=gen_tokens / makespan,
        utilization=util,
        completed=completed,
        makespan_s=makespan,
        per_req_tbt_mean_ms=np.asarray(per_req),
    )


# --------------------------------------------------------------------------
# Colocated simulation
# --------------------------------------------------------------------------
def _simulate_colocated_worker(reqs, cfg):
    """Simulate one colocated worker; requests sorted by arrival."""
    t = 0.0
    i, n = 0, len(reqs)
    waiting = deque()
    active = []          # requests currently decoding
    busy = 0.0
    completed = 0
    while i < n or waiting or active:
        while i < n and reqs[i].arrival <= t + 1e-12:
            waiting.append(reqs[i])
            i += 1
        # Admit prefills while they fit in the KV cap.  The fallback
        # (`not active and not new`) guarantees progress: an empty worker
        # always admits the head of the queue.
        kv_used = sum(a._kv for a in active)
        new = []
        while waiting:
            r = waiting[0]
            if kv_used + r._kv <= cfg.kv_cap_tokens or (not active and not new):
                waiting.popleft()
                new.append(r)
                kv_used += r._kv
            else:
                break
        batch_tokens = sum(r.prompt_len for r in new) + len(active)
        if batch_tokens == 0:
            # Idle: jump to the next arrival.
            t = reqs[i].arrival
            continue
        dt = (cfg.t0_ms + cfg.k_ms_per_token * batch_tokens) / 1000.0
        t_end = t + dt
        for r in new:
            r.ttft_s = t_end - r.arrival
            r.token_times.append(t_end)
            r._rem = r.gen_len - 1
            if r._rem <= 0:
                r.completed_at = t_end
                completed += 1
        still = []
        for a in active:
            a.token_times.append(t_end)
            a._rem -= 1
            if a._rem <= 0:
                a.completed_at = t_end
                completed += 1
            else:
                still.append(a)
        active = still + [r for r in new if r._rem > 0]
        busy += dt
        t = t_end
    return busy, completed


def run_colocated(requests, cfg=None):
    """Run the colocated architecture over a request list (sorted by arrival)."""
    cfg = cfg or ColocatedConfig()
    reqs = sorted(requests, key=lambda r: r.arrival)
    buckets = _assign_jsq(reqs, cfg.workers)
    busy, completed = [], 0
    for b in buckets:
        bw, c = _simulate_colocated_worker(b, cfg)
        busy.append(bw)
        completed += c
    ttfts, gaps, per_req, gen_tokens, t_end = _gather(reqs)
    t_start = min(r.arrival for r in reqs)
    return _finish(ttfts, gaps, per_req, gen_tokens, t_end, t_start,
                   busy, completed, len(reqs))


# --------------------------------------------------------------------------
# Disaggregated simulation
# --------------------------------------------------------------------------
def _simulate_prefill_worker(reqs, tok_per_s):
    """One prefill worker: FIFO, one request at a time. Returns busy time."""
    t = 0.0
    busy = 0.0
    for r in reqs:
        t = max(t, r.arrival)
        dt = r.prompt_len / float(tok_per_s)
        t += dt
        busy += dt
        r.ttft_s = float("nan")  # set later; first token comes from decode
        r._prefill_done = t
    return busy


def _simulate_decode_worker(items, cfg):
    """One decode worker; items = [(ready_time, request), ...] sorted."""
    t = 0.0
    i, n = 0, len(items)
    waiting = deque()
    active = []
    busy = 0.0
    completed = 0
    while i < n or waiting or active:
        while i < n and items[i][0] <= t + 1e-12:
            waiting.append(items[i][1])
            i += 1
        kv_used = sum(a._kv for a in active)
        new = []
        while waiting:
            r = waiting[0]
            if kv_used + r._kv <= cfg.decode_kv_cap_tokens or not active:
                waiting.popleft()
                new.append(r)
                kv_used += r._kv
            else:
                break
        if not active and not new:
            t = items[i][0]
            continue
        dt = (cfg.decode_t0_ms + cfg.decode_k_ms_per_token * (len(active) + len(new))) / 1000.0
        t_end = t + dt
        for r in new:
            r.ttft_s = t_end - r.arrival
            r.token_times.append(t_end)
            r._rem = r.gen_len - 1
            if r._rem <= 0:
                r.completed_at = t_end
                completed += 1
        still = []
        for a in active:
            a.token_times.append(t_end)
            a._rem -= 1
            if a._rem <= 0:
                a.completed_at = t_end
                completed += 1
            else:
                still.append(a)
        active = still + [r for r in new if r._rem > 0]
        busy += dt
        t = t_end
    return busy, completed


def run_disaggregated(requests, cfg=None):
    """Run the disaggregated (prefill pool + decode pool) architecture."""
    cfg = cfg or DisaggConfig()
    reqs = sorted(requests, key=lambda r: r.arrival)
    # 1) prefill pool, JSQ by arrival order
    prefill_buckets = _assign_jsq(reqs, cfg.prefill_workers)
    prefill_busy = [_simulate_prefill_worker(b, cfg.prefill_tok_per_s)
                    for b in prefill_buckets]
    # 2) KV transfer over the interconnect
    ready = []
    for r in reqs:
        delay = transfer_delay_s(r.prompt_len, cfg.interconnect_gbps,
                                 cfg.n_layers, cfg.d_model, cfg.bytes_per_elem,
                                 cfg.transfer_latency_ms)
        ready.append((r._prefill_done + delay, r.arrival, r))
    ready.sort(key=lambda x: (x[0], x[1]))
    # 3) decode pool, JSQ by assigned KV-token load, in transfer-completion order
    loads = [0] * cfg.decode_workers
    decode_buckets = [[] for _ in range(cfg.decode_workers)]
    for rt, _, r in ready:
        w = 0
        for j in range(1, cfg.decode_workers):
            if loads[j] < loads[w]:
                w = j
        decode_buckets[w].append((rt, r))
        loads[w] += r._kv
    decode_busy, completed = [], 0
    for b in decode_buckets:
        b.sort(key=lambda x: x[0])
        bw, c = _simulate_decode_worker(b, cfg)
        decode_busy.append(bw)
        completed += c
    ttfts, gaps, per_req, gen_tokens, t_end = _gather(reqs)
    t_start = min(r.arrival for r in reqs)
    return _finish(ttfts, gaps, per_req, gen_tokens, t_end, t_start,
                   prefill_busy + decode_busy, completed, len(reqs))
