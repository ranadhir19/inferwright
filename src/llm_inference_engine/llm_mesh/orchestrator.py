"""llm_mesh.orchestrator: task decomposition, dispatch, aggregation.

An ``Orchestrator`` takes a list of ``Subtask``s (each typed by task type),
routes each to a model under a routing policy, dispatches them
sequentially or in parallel, and aggregates the results. Tool agents
(e.g. a database check) are simulated as fixed-latency, fixed-cost calls.
With ``failover=True``, a ``ModelDown`` on the routed model triggers an
automatic re-route to the next-best available model.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import TASK_TYPES, ModelDown, ModelRegistry
from .observability import Record, Trace
from .router import POLICIES, route

# Simulated tool agent: a database / API check inside a mesh workflow.
TOOL_LATENCY_MS = 30.0
TOOL_COST_USD = 0.0001
TOOL_QUALITY = 0.97

# Fixed aggregation overhead for parallel dispatch (result collection,
# prompt stitching for the aggregator step).
AGG_OVERHEAD_MS = 50.0


@dataclass
class Subtask:
    name: str
    task_type: str  # one of TASK_TYPES, or "tool" for a tool agent
    in_tokens: int
    out_tokens: int
    prompt: str = ""
    tool: str = ""  # tool name when task_type == "tool"


@dataclass
class MeshResult:
    request_id: str
    mode: str
    records: List[Record]
    wall_latency_ms: float
    total_cost_usd: float
    mean_quality: float
    n_failovers: int
    failed: bool = False

    def as_trace(self, trace: Optional[Trace] = None) -> Trace:
        # NB: do not use `trace or Trace()` here -- an empty Trace is
        # falsy (it defines __len__), which would discard the caller's
        # trace object.
        if trace is None:
            trace = Trace()
        for r in self.records:
            trace.log(r)
        return trace


class Orchestrator:
    def __init__(self, registry: ModelRegistry, policy: str = "quality-first",
                 quality_floor: float = 0.60):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
        self.registry = registry
        self.policy = policy
        self.quality_floor = quality_floor

    def _run_subtask(self, rng, request_id: str, sub: Subtask,
                     failover: bool) -> Record:
        if sub.task_type == "tool":
            return Record(
                request_id=request_id, subtask=sub.name,
                task_type="tool", model=f"tool:{sub.tool or 'db'}",
                in_tokens=sub.in_tokens, out_tokens=sub.out_tokens,
                latency_ms=TOOL_LATENCY_MS, cost_usd=TOOL_COST_USD,
                quality=TOOL_QUALITY,
            )
        if sub.task_type not in TASK_TYPES:
            raise ValueError(f"unknown task type {sub.task_type!r}")
        model = route(self.registry, sub.task_type, sub.in_tokens,
                      sub.out_tokens, self.policy, self.quality_floor)
        try:
            res = self.registry.call(rng, model, sub.task_type,
                                     sub.in_tokens, sub.out_tokens)
        except ModelDown:
            if not failover:
                raise
            tried = (model,)
            model2 = route(self.registry, sub.task_type, sub.in_tokens,
                           sub.out_tokens, self.policy, self.quality_floor,
                           exclude=tried)
            res = self.registry.call(rng, model2, sub.task_type,
                                     sub.in_tokens, sub.out_tokens,
                                     failover_from=model)
        return Record(
            request_id=request_id, subtask=sub.name, task_type=sub.task_type,
            model=res.model, in_tokens=res.in_tokens,
            out_tokens=res.out_tokens, latency_ms=res.latency_ms,
            cost_usd=res.cost_usd, quality=res.quality,
            failover_from=res.failover_from,
        )

    def run(self, rng, request_id: str, subtasks: List[Subtask],
            mode: str = "sequential", failover: bool = True,
            trace: Optional[Trace] = None) -> MeshResult:
        """Dispatch subtasks and aggregate.

        Calls are executed serially in code (deterministic RNG order), but
        ``mode="parallel"`` models wall-clock latency as
        ``max(subtask latencies) + AGG_OVERHEAD_MS``, i.e. the mesh fans
        the subtasks out concurrently and aggregates.
        """
        if mode not in ("sequential", "parallel"):
            raise ValueError(f"unknown mode {mode!r}")
        records = [self._run_subtask(rng, request_id, s, failover)
                   for s in subtasks]
        lats = [r.latency_ms for r in records]
        wall = (sum(lats) if mode == "sequential"
                else (max(lats) + AGG_OVERHEAD_MS if lats else 0.0))
        result = MeshResult(
            request_id=request_id, mode=mode, records=records,
            wall_latency_ms=float(wall),
            total_cost_usd=float(sum(r.cost_usd for r in records)),
            mean_quality=float(sum(r.quality for r in records) / len(records))
            if records else 0.0,
            n_failovers=sum(1 for r in records if r.failover_from),
        )
        if trace is not None:
            result.as_trace(trace)
        return result

    def routing_table(self, in_tokens: int = 400,
                      out_tokens: int = 200) -> Dict[str, str]:
        """Which model the current policy would pick per task type."""
        return {t: route(self.registry, t, in_tokens, out_tokens, self.policy,
                         self.quality_floor) for t in TASK_TYPES}
