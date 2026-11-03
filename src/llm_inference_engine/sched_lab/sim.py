"""Discrete-event simulator for LLM inference request scheduling.

Compares four serving policies on a shared, seeded workload:

- ``static-fcfs``      -- fixed FCFS batches, each run to completion
  (head-of-line blocking baseline).
- ``continuous``       -- iteration-level batching: newly arrived requests
  join freed slots every iteration (vLLM style).
- ``chunked-prefill``  -- continuous batching with long prompts split into
  fixed-size chunks interleaved with decode steps (Sarathi style).
- ``priority-preempt`` -- continuous batching where short requests preempt
  long ones; a preempted request loses its KV cache and must recompute
  ``prompt_len + generated_so_far`` tokens when rescheduled (honest
  recompute cost).

Analytical GPU model
--------------------
Every iteration costs ``t = T0 + K * batch_tokens`` milliseconds, where
``batch_tokens`` is the number of tokens processed that iteration (prefill
tokens plus one token per decoding request). The KV cache is a hard token
budget (default 16384); each admitted request reserves
``prompt_len + gen_len`` tokens up front, and the running set never exceeds
the budget.

This model is analytical, calibrated to the *shape* of real serving curves
(fixed launch overhead plus roughly linear token scaling). Absolute
millisecond values are illustrative; the relative policy differences are
the finding.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Defaults for the analytical GPU / serving model
# ---------------------------------------------------------------------------

T0_MS = 2.0              # fixed per-iteration launch + scheduling overhead (ms)
K_MS_PER_TOKEN = 0.012   # incremental cost per token processed in an iteration
KV_CAP_TOKENS = 16384    # total KV-cache token budget
CHUNK_TOKENS = 512       # prefill chunk size for chunked-prefill
MAX_BATCH = 64           # max concurrent requests admitted per iteration
SHORT_TOTAL_TOKENS = 768  # prompt_len + gen_len <= this  =>  "short" request
AGE_S = 45.0             # waiting longer than this promotes a long request


@dataclass(frozen=True)
class SimConfig:
    """Analytical GPU / serving configuration."""

    t0_ms: float = T0_MS
    k_ms_per_token: float = K_MS_PER_TOKEN
    kv_cap_tokens: int = KV_CAP_TOKENS
    chunk_tokens: int = CHUNK_TOKENS
    max_batch: int = MAX_BATCH
    short_total_tokens: int = SHORT_TOTAL_TOKENS
    age_s: float = AGE_S


class Req:
    """Mutable per-request runtime state (built from a workload spec)."""

    __slots__ = (
        "rid", "arrival", "prompt_len", "gen_len", "is_short",
        "remaining_prompt", "generated", "state", "kv_reserved",
        "first_token_t", "done_t", "preemptions", "recompute_charged",
    )

    def __init__(self, rid: int, arrival: float, prompt_len: int,
                 gen_len: int, is_short: bool):
        self.rid = rid
        self.arrival = arrival
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.is_short = is_short
        self.remaining_prompt = prompt_len
        self.generated = 0
        self.state = "waiting"          # waiting | prefill | decode | done
        self.kv_reserved = prompt_len + gen_len
        self.first_token_t = -1.0       # ms sim-time of first generated token
        self.done_t = -1.0              # ms sim-time of completion
        self.preemptions = 0
        self.recompute_charged = 0      # extra tokens this request had to redo


# A unit of work for one iteration: (request, tokens_to_process, is_decode).
# Decode work is always exactly 1 token; prefill work is a token count.
Work = Tuple[Req, int, bool]


class Scheduler(abc.ABC):
    """Common interface implemented by all four serving policies."""

    name: str = "base"

    def reset(self, sim: "Simulator") -> None:
        """Clear per-run scheduler state."""

    @abc.abstractmethod
    def admit(self, sim: "Simulator") -> None:
        """Move requests from sim.waiting into the running set (may preempt)."""

    @abc.abstractmethod
    def plan(self, sim: "Simulator") -> List[Work]:
        """Choose this iteration's work given the current running set."""

    def on_finish(self, sim: "Simulator", req: Req) -> None:
        """Hook called when a request completes."""


class StaticFCFSScheduler(Scheduler):
    """Fixed FCFS batches, each run to completion.

    Forms a batch from the queue head (bounded by the KV budget and
    MAX_BATCH) and runs it to completion before admitting anything else.
    If the head request does not fit the KV budget, nothing behind it is
    admitted: genuine head-of-line blocking.
    """

    name = "static-fcfs"

    def reset(self, sim: "Simulator") -> None:
        self._batch: List[Req] = []

    def admit(self, sim: "Simulator") -> None:
        self._batch = [r for r in self._batch if r.state != "done"]
        if self._batch:
            return
        cfg = sim.config
        while sim.waiting and len(self._batch) < cfg.max_batch:
            head = sim.waiting[0]
            if sim.kv_used + head.kv_reserved > cfg.kv_cap_tokens:
                break  # head-of-line blocking: do not skip the head
            sim.waiting.pop(0)
            sim._admit_req(head)
            self._batch.append(head)

    def plan(self, sim: "Simulator") -> List[Work]:
        work: List[Work] = []
        for req in self._batch:
            if req.state == "prefill":
                # Batched prefill: every member's whole prompt in one iteration.
                work.append((req, req.remaining_prompt, False))
            elif req.state == "decode":
                work.append((req, 1, True))
        return work

    def on_finish(self, sim: "Simulator", req: Req) -> None:
        if req in self._batch:
            self._batch.remove(req)


class ContinuousScheduler(Scheduler):
    """Iteration-level batching (vLLM style).

    Every iteration, waiting requests are admitted FCFS into freed KV slots.
    A request's full prompt is prefetched in a single iteration; each
    decoding request then emits one token per iteration until done. No
    preemption: a running request keeps its slot to completion.
    """

    name = "continuous"

    def admit(self, sim: "Simulator") -> None:
        cfg = sim.config
        i = 0
        while i < len(sim.waiting) and len(sim.running) < cfg.max_batch:
            req = sim.waiting[i]
            if sim.kv_used + req.kv_reserved <= cfg.kv_cap_tokens:
                sim.waiting.pop(i)
                sim._admit_req(req)
            else:
                i += 1

    def plan(self, sim: "Simulator") -> List[Work]:
        work: List[Work] = []
        for req in sim.running:
            if req.state == "prefill":
                work.append((req, req.remaining_prompt, False))
            elif req.state == "decode":
                work.append((req, 1, True))
        return work


class ChunkedPrefillScheduler(ContinuousScheduler):
    """Continuous batching with chunked prefill (Sarathi style).

    Same admission as continuous, but each request's prompt is processed in
    chunks of at most ``chunk_tokens`` per iteration, interleaved with the
    decode steps of all running requests. A 4096-token prompt therefore
    streams over 8 iterations instead of stalling a single giant one, so
    every iteration's latency stays bounded and short requests (whose whole
    prompt fits in one chunk) never queue behind a burst of long prefills.
    """

    name = "chunked-prefill"

    def plan(self, sim: "Simulator") -> List[Work]:
        cfg = sim.config
        work: List[Work] = []
        for req in sim.running:
            if req.state == "prefill":
                take = min(req.remaining_prompt, cfg.chunk_tokens)
                work.append((req, take, False))
        for req in sim.running:
            if req.state == "decode":
                work.append((req, 1, True))
        return work


class PriorityPreemptScheduler(ContinuousScheduler):
    """Continuous batching plus short-request priority with preemption.

    Waiting short requests are admitted ahead of long ones. When a short
    request cannot be admitted because the KV budget is full, the running
    long request with the least decode progress is preempted: its KV is
    dropped and, on rescheduling, it must recompute
    ``prompt_len + generated_so_far`` tokens (charged honestly as extra
    batch tokens in later iterations). Long requests waiting longer than
    ``age_s`` are promoted so they cannot starve forever.
    """

    name = "priority-preempt"

    def admit(self, sim: "Simulator") -> None:
        cfg = sim.config
        t = sim.t

        def eff_short(req: Req) -> bool:
            # sim.t and req.arrival are both in ms; age_s is in seconds.
            return req.is_short or (t - req.arrival) / 1000.0 > cfg.age_s

        order = sorted(
            sim.waiting,
            key=lambda r: (0 if eff_short(r) else 1, r.arrival, r.rid),
        )
        for req in order:
            if len(sim.running) >= cfg.max_batch:
                break
            # Make room for short requests by preempting long runners.
            while sim.kv_used + req.kv_reserved > cfg.kv_cap_tokens:
                if not eff_short(req):
                    break
                victim: Optional[Req] = None
                best: Optional[Tuple[int, int]] = None
                for r in sim.running:
                    if r.is_short:
                        continue
                    key = (r.generated, r.rid)
                    if best is None or key < best:
                        best = key
                        victim = r
                if victim is None:
                    break
                sim._preempt(victim)
            if sim.kv_used + req.kv_reserved <= cfg.kv_cap_tokens:
                sim.waiting.remove(req)
                sim._admit_req(req)
            # else: cannot fit and nothing preemptible -- leave it queued.


POLICY_NAMES = [
    "static-fcfs",
    "continuous",
    "chunked-prefill",
    "priority-preempt",
]


def make_scheduler(name: str) -> Scheduler:
    """Instantiate a scheduler policy by name."""
    if name == "static-fcfs":
        return StaticFCFSScheduler()
    if name == "continuous":
        return ContinuousScheduler()
    if name == "chunked-prefill":
        return ChunkedPrefillScheduler()
    if name == "priority-preempt":
        return PriorityPreemptScheduler()
    raise ValueError(f"unknown scheduler policy: {name!r}")


@dataclass
class SimResult:
    """Per-request outcomes plus per-iteration traces of one simulation run."""

    policy: str
    config: SimConfig
    n: int
    makespan_s: float
    arrival: np.ndarray          # s
    prompt_len: np.ndarray
    gen_len: np.ndarray
    is_short: np.ndarray         # bool
    ttft_ms: np.ndarray
    tpot_ms: np.ndarray          # per-request mean time per output token
    first_token_t: np.ndarray    # ms sim time
    done_t: np.ndarray           # ms sim time
    recompute_charged: np.ndarray
    preemptions: int
    recompute_tokens: int
    max_kv_used: int
    total_gen_tokens: int
    total_prompt_tokens: int
    iter_batch_tokens: np.ndarray
    iter_prefill_tokens: np.ndarray
    iter_dt_ms: np.ndarray


def summarize(result: SimResult, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Aggregate headline metrics; optionally restricted to a request subset."""
    if mask is None:
        mask = np.ones(result.n, dtype=bool)
    ttft = result.ttft_ms[mask]
    tpot = result.tpot_ms[mask]
    bt = result.iter_batch_tokens.astype(float)
    dt = result.iter_dt_ms
    time_wtd_batch = float(np.sum(bt * dt) / np.sum(dt)) if np.sum(dt) > 0 else 0.0
    return {
        "n": int(np.sum(mask)),
        "ttft_mean_ms": float(np.mean(ttft)),
        "ttft_p99_ms": float(np.percentile(ttft, 99)),
        "tpot_mean_ms": float(np.mean(tpot)),
        "tpot_p99_ms": float(np.percentile(tpot, 99)),
        # Generated-token throughput over the whole run makespan.
        "throughput_tok_s": float(result.total_gen_tokens / result.makespan_s)
        if result.makespan_s > 0 else 0.0,
        # Time-weighted mean batch tokens per iteration, as a fraction of the
        # KV token budget.
        "gpu_util": time_wtd_batch / result.config.kv_cap_tokens,
        "makespan_s": float(result.makespan_s),
        "preemptions": int(result.preemptions),
        "recompute_tokens": int(result.recompute_tokens),
        "max_kv_used": int(result.max_kv_used),
    }


class Simulator:
    """Iteration-stepped discrete-event simulator for one workload."""

    def __init__(
        self,
        arrivals: np.ndarray,
        prompt_lens: np.ndarray,
        gen_lens: np.ndarray,
        scheduler: Scheduler,
        config: Optional[SimConfig] = None,
    ):
        self.config = config or SimConfig()
        self.scheduler = scheduler
        order = np.argsort(arrivals, kind="stable")
        # Arrivals stored in ms (same unit as sim time) so the idle-jump
        # `self.t = arrivals[ai]` compares exactly on the next pass --
        # (x*1000)/1000 is not always == x in floating point.
        self._arrivals = np.asarray(arrivals, dtype=float)[order] * 1000.0
        prompt_lens = np.asarray(prompt_lens, dtype=np.int64)[order]
        gen_lens = np.asarray(gen_lens, dtype=np.int64)[order]
        self.reqs: List[Req] = [
            Req(
                rid=i,
                arrival=float(self._arrivals[i]),  # ms
                prompt_len=int(prompt_lens[i]),
                gen_len=int(gen_lens[i]),
                is_short=bool(
                    prompt_lens[i] + gen_lens[i] <= self.config.short_total_tokens
                ),
            )
            for i in range(len(order))
        ]
        self.waiting: List[Req] = []
        self.running: List[Req] = []
        self.t = 0.0  # sim time, ms
        self.kv_used = 0
        self.done = 0
        self.max_kv_used = 0
        self.preemptions = 0
        self.recompute_tokens = 0
        self.iter_batch_tokens: List[int] = []
        self.iter_prefill_tokens: List[int] = []
        self.iter_dt_ms: List[float] = []
        self._admit_needed = True

    # -- internal helpers used by schedulers --------------------------------

    def _admit_req(self, req: Req) -> None:
        self.kv_used += req.kv_reserved
        if self.kv_used > self.max_kv_used:
            self.max_kv_used = self.kv_used
        self.running.append(req)
        req.state = "prefill" if req.remaining_prompt > 0 else "decode"

    def _preempt(self, req: Req) -> None:
        """Drop a running request's KV; it must recompute on reschedule."""
        self.running.remove(req)
        self.kv_used -= req.kv_reserved
        req.preemptions += 1
        charge = req.prompt_len + req.generated
        req.recompute_charged += charge
        self.recompute_tokens += charge
        # Logical decode progress is kept, but the KV for the prompt plus
        # every generated token must be rebuilt before decoding resumes.
        req.remaining_prompt = req.prompt_len + req.generated
        req.state = "waiting"
        self.waiting.append(req)
        self.preemptions += 1
        self._admit_needed = True

    def _finish(self, req: Req) -> None:
        req.state = "done"
        req.done_t = self.t
        self.kv_used -= req.kv_reserved
        self.running.remove(req)
        self.done += 1
        self.scheduler.on_finish(self, req)

    # -- main loop -----------------------------------------------------------

    def run(self) -> SimResult:
        n = len(self.reqs)
        ai = 0  # next arrival index
        arrivals = self._arrivals
        sched = self.scheduler
        sched.reset(self)

        while self.done < n:
            arrived = False
            while ai < n and arrivals[ai] <= self.t:
                self.waiting.append(self.reqs[ai])
                ai += 1
                arrived = True
            if arrived:
                self._admit_needed = True
            if self._admit_needed:
                sched.admit(self)
                self._admit_needed = False

            work = sched.plan(self)
            if not work:
                if ai < n:
                    self.t = arrivals[ai]  # idle: jump to next arrival (exact)
                    continue
                if self.waiting:
                    raise RuntimeError(
                        "deadlock: waiting requests cannot be admitted"
                    )
                break

            batch_tokens = 0
            prefill_tokens = 0
            for _req, tok, is_decode in work:
                batch_tokens += tok
                if not is_decode:
                    prefill_tokens += tok
            dt = self.config.t0_ms + self.config.k_ms_per_token * batch_tokens
            self.t += dt

            freed = False
            for req, tok, is_decode in work:
                if is_decode:
                    req.generated += 1
                    if req.first_token_t < 0.0:
                        req.first_token_t = self.t
                    if req.generated >= req.gen_len:
                        self._finish(req)
                        freed = True
                else:
                    req.remaining_prompt -= tok
                    if req.remaining_prompt <= 0:
                        req.remaining_prompt = 0
                        req.state = "decode"
            if freed:
                self._admit_needed = True

            self.iter_batch_tokens.append(batch_tokens)
            self.iter_prefill_tokens.append(prefill_tokens)
            self.iter_dt_ms.append(dt)

        return self._result()

    def _result(self) -> SimResult:
        n = len(self.reqs)
        arrival_ms = np.array([r.arrival for r in self.reqs])
        arrival = arrival_ms / 1000.0  # public API keeps seconds
        prompt_len = np.array([r.prompt_len for r in self.reqs])
        gen_len = np.array([r.gen_len for r in self.reqs])
        is_short = np.array([r.is_short for r in self.reqs])
        first_token_t = np.array([r.first_token_t for r in self.reqs])
        done_t = np.array([r.done_t for r in self.reqs])
        ttft_ms = first_token_t - arrival_ms
        tpot_ms = (done_t - first_token_t) / np.maximum(gen_len, 1)
        return SimResult(
            policy=self.scheduler.name,
            config=self.config,
            n=n,
            makespan_s=self.t / 1000.0,
            arrival=arrival,
            prompt_len=prompt_len,
            gen_len=gen_len,
            is_short=is_short,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            first_token_t=first_token_t,
            done_t=done_t,
            recompute_charged=np.array([r.recompute_charged for r in self.reqs]),
            preemptions=self.preemptions,
            recompute_tokens=self.recompute_tokens,
            max_kv_used=self.max_kv_used,
            total_gen_tokens=int(np.sum(gen_len)),
            total_prompt_tokens=int(np.sum(prompt_len)),
            iter_batch_tokens=np.array(self.iter_batch_tokens, dtype=np.int64),
            iter_prefill_tokens=np.array(self.iter_prefill_tokens, dtype=np.int64),
            iter_dt_ms=np.array(self.iter_dt_ms),
        )


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------

def make_workload(
    rng: np.random.Generator,
    n: int,
    arrival_rate: float,
    long_frac: float = 0.15,
) -> Dict[str, np.ndarray]:
    """Poisson arrivals with a mixed short-chat / long-document shape mix.

    - 1 - long_frac: chat requests, prompt 32-512 tokens, gen 32-512 tokens.
    - long_frac: document requests, prompt 1024-4096 tokens, gen 128-1024.
    """
    inter = rng.exponential(1.0 / arrival_rate, size=n)
    arrival = np.cumsum(inter)
    is_long = rng.random(n) < long_frac
    prompt_len = np.where(
        is_long,
        rng.integers(1024, 4097, size=n),
        rng.integers(32, 513, size=n),
    ).astype(np.int64)
    gen_len = np.where(
        is_long,
        rng.integers(128, 1025, size=n),
        rng.integers(32, 513, size=n),
    ).astype(np.int64)
    return {"arrival": arrival, "prompt_len": prompt_len, "gen_len": gen_len}


def add_burst(
    specs: Dict[str, np.ndarray],
    t_burst: float,
    prompt_len: int = 4096,
    gen_len: int = 256,
    count: int = 10,
) -> Dict[str, np.ndarray]:
    """Append ``count`` long-prompt requests all arriving at ``t_burst``."""
    burst = {
        "arrival": np.full(count, t_burst),
        "prompt_len": np.full(count, prompt_len, dtype=np.int64),
        "gen_len": np.full(count, gen_len, dtype=np.int64),
    }
    merged = {
        k: np.concatenate([specs[k], burst[k]]) for k in ("arrival", "prompt_len", "gen_len")
    }
    order = np.argsort(merged["arrival"], kind="stable")
    return {k: v[order] for k, v in merged.items()}


def run_policy(
    policy_name: str,
    specs: Dict[str, np.ndarray],
    config: Optional[SimConfig] = None,
) -> SimResult:
    """Run one policy on a workload spec; returns per-request results."""
    sim = Simulator(
        specs["arrival"],
        specs["prompt_len"],
        specs["gen_len"],
        make_scheduler(policy_name),
        config or SimConfig(),
    )
    return sim.run()
