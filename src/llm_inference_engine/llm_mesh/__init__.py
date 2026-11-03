"""llm_mesh: LLM mesh simulator (numpy, CPU only).

A service-mesh-style architecture for many specialized language models:
a model registry, a semantic router, an orchestrator with sequential and
parallel dispatch, a self-improving (UCB1 bandit) router, dynamic model
selection from observed history, observability traces, failover on model
outage, and two end-to-end workflows.

Public API:
    TASK_TYPES, ModelSpec, CallResult, ModelDown, ModelRegistry,
    default_models,
    SemanticRouter, POLICIES, route, route_query, train_router,
    TRAIN_CORPUS, TEST_CORPUS,
    Subtask, MeshResult, Orchestrator,
    TOOL_LATENCY_MS, TOOL_COST_USD, TOOL_QUALITY, AGG_OVERHEAD_MS,
    Record, Trace,
    BanditRouter, DynamicSelector,
    fraud_claim_pipeline, chart_report_pipeline
"""

from .bandit import BanditRouter, DynamicSelector
from .models import (
    TASK_TYPES,
    CallResult,
    ModelDown,
    ModelRegistry,
    ModelSpec,
    default_models,
)
from .observability import Record, Trace
from .orchestrator import (
    AGG_OVERHEAD_MS,
    TOOL_COST_USD,
    TOOL_LATENCY_MS,
    TOOL_QUALITY,
    MeshResult,
    Orchestrator,
    Subtask,
)
from .router import (
    POLICIES,
    TEST_CORPUS,
    TRAIN_CORPUS,
    SemanticRouter,
    route,
    route_query,
    test_texts_labels,
    train_router,
)
from .workflows import chart_report_pipeline, fraud_claim_pipeline

__all__ = [
    "TASK_TYPES",
    "ModelSpec",
    "CallResult",
    "ModelDown",
    "ModelRegistry",
    "default_models",
    "SemanticRouter",
    "POLICIES",
    "route",
    "route_query",
    "train_router",
    "TRAIN_CORPUS",
    "TEST_CORPUS",
    "test_texts_labels",
    "Subtask",
    "MeshResult",
    "Orchestrator",
    "TOOL_LATENCY_MS",
    "TOOL_COST_USD",
    "TOOL_QUALITY",
    "AGG_OVERHEAD_MS",
    "Record",
    "Trace",
    "BanditRouter",
    "DynamicSelector",
    "fraud_claim_pipeline",
    "chart_report_pipeline",
]
