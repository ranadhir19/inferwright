"""llm_mesh.observability: per-request traces for the mesh.

Every model invocation and tool call appends one record. ``Trace.summary``
reduces a run to headline metrics: totals, per-model routing counts,
mean quality, and failover counts.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Record:
    request_id: str
    subtask: str
    task_type: str
    model: str  # model name, or "tool:<name>" for tool agents
    in_tokens: int
    out_tokens: int
    latency_ms: float
    cost_usd: float
    quality: float
    failover_from: Optional[str] = None
    failed: bool = False


class Trace:
    """Append-only log of mesh invocations."""

    def __init__(self):
        self.records: List[Record] = []

    def log(self, rec: Record) -> None:
        self.records.append(rec)

    def __len__(self) -> int:
        return len(self.records)

    def rows(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self.records]

    def summary(self) -> Dict[str, Any]:
        ok = [r for r in self.records if not r.failed]
        lat = np.array([r.latency_ms for r in ok])
        per_model: Dict[str, int] = {}
        for r in ok:
            per_model[r.model] = per_model.get(r.model, 0) + 1
        return {
            "n_records": len(self.records),
            "n_ok": len(ok),
            "n_failed": len(self.records) - len(ok),
            "n_failovers": sum(1 for r in ok if r.failover_from),
            "total_latency_ms": float(lat.sum()) if len(lat) else 0.0,
            "mean_latency_ms": float(lat.mean()) if len(lat) else 0.0,
            "p95_latency_ms": float(np.percentile(lat, 95)) if len(lat) else 0.0,
            "total_cost_usd": float(sum(r.cost_usd for r in ok)),
            "mean_quality": float(np.mean([r.quality for r in ok]))
            if ok else 0.0,
            "per_model": per_model,
        }
