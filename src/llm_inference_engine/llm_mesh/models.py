"""llm_mesh.models: simulated specialized-model registry for the mesh.

Each model is an analytical profile: mean quality per task type, latency
per output token, and price per 1k tokens. ``ModelRegistry.call`` samples
one inference with seeded noise (latency jitter, quality noise) so the
harness runs deterministically on CPU. No real LLM is invoked.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

TASK_TYPES = ("reasoning", "coding", "vision", "chat")


@dataclass(frozen=True)
class ModelSpec:
    """Static profile of one model in the mesh registry."""

    name: str
    family: str  # e.g. "frontier", "reasoning", "coding", "vision", "chat"
    in_usd_per_1k: float
    out_usd_per_1k: float
    ms_per_out_token: float
    quality: Dict[str, float]  # task_type -> mean quality in [0, 1]


@dataclass
class CallResult:
    """One simulated model invocation."""

    model: str
    task_type: str
    in_tokens: int
    out_tokens: int
    latency_ms: float
    cost_usd: float
    quality: float
    failover_from: Optional[str] = None


class ModelDown(Exception):
    """Raised when a call is attempted on a model marked down."""


def default_models() -> List[ModelSpec]:
    """Five simulated specialists plus one expensive generalist.

    Specialists beat the frontier model inside their niche and lose
    outside it; the frontier model is strong everywhere but costly.
    """
    return [
        ModelSpec(
            "frontier-1", "frontier", 0.010, 0.030, 12.0,
            {"reasoning": 0.93, "coding": 0.90, "vision": 0.88, "chat": 0.95},
        ),
        ModelSpec(
            "reasoner-r1", "reasoning", 0.002, 0.008, 18.0,
            {"reasoning": 0.96, "coding": 0.72, "vision": 0.40, "chat": 0.80},
        ),
        ModelSpec(
            "coder-cl", "coding", 0.001, 0.003, 8.0,
            {"reasoning": 0.65, "coding": 0.94, "vision": 0.35, "chat": 0.78},
        ),
        ModelSpec(
            "vision-qw", "vision", 0.002, 0.006, 10.0,
            {"reasoning": 0.55, "coding": 0.50, "vision": 0.93, "chat": 0.75},
        ),
        ModelSpec(
            "general-8b", "chat", 0.0003, 0.0006, 4.0,
            {"reasoning": 0.62, "coding": 0.68, "vision": 0.30, "chat": 0.90},
        ),
    ]


class ModelRegistry:
    """Catalog of mesh models with availability and cost/latency models."""

    def __init__(self, models: Optional[List[ModelSpec]] = None):
        self.models: Dict[str, ModelSpec] = {
            m.name: m for m in (models or default_models())
        }
        self.down = set()

    # -- availability ----------------------------------------------------
    def mark_down(self, name: str) -> None:
        if name not in self.models:
            raise KeyError(name)
        self.down.add(name)

    def mark_up(self, name: str) -> None:
        self.down.discard(name)

    def available(self) -> List[str]:
        return [n for n in self.models if n not in self.down]

    def is_up(self, name: str) -> bool:
        return name in self.models and name not in self.down

    # -- analytical expectations (no randomness) -------------------------
    def quality_of(self, name: str, task_type: str) -> float:
        return self.models[name].quality[task_type]

    def expected_cost_usd(self, name: str, in_tokens: int,
                          out_tokens: int) -> float:
        m = self.models[name]
        return (m.in_usd_per_1k * in_tokens
                + m.out_usd_per_1k * out_tokens) / 1000.0

    def expected_latency_ms(self, name: str, out_tokens: int) -> float:
        return self.models[name].ms_per_out_token * out_tokens + 20.0

    # -- simulated invocation --------------------------------------------
    def call(self, rng, name: str, task_type: str, in_tokens: int,
             out_tokens: int,
             failover_from: Optional[str] = None) -> CallResult:
        if name in self.down:
            raise ModelDown(name)
        spec = self.models[name]
        latency = max(
            1.0,
            spec.ms_per_out_token * out_tokens * rng.normal(1.0, 0.12)
            + rng.normal(20.0, 5.0),
        )
        quality = float(
            min(1.0, max(0.0,
                         spec.quality[task_type] + rng.normal(0.0, 0.03)))
        )
        return CallResult(
            model=name,
            task_type=task_type,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            latency_ms=float(latency),
            cost_usd=self.expected_cost_usd(name, in_tokens, out_tokens),
            quality=quality,
            failover_from=failover_from,
        )
