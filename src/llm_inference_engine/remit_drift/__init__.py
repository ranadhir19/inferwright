"""remit-drift: divergence of remit in multi-agent systems.

Scope drift for autonomous agents, measured and governed.

- ``remit``     — Remit / Grant: the mandate as a first-class runtime object
- ``policy``    — PolicyEngine: allow/deny/escalate validation, capability
                  registry, remit versioning, continuous auditing
- ``metrics``   — D(t): per-type divergence measures (functional KL, authority
                  rate, goal gap, planning violations, tool/data rates,
                  delegation drift, expired-grant usage)
- ``sim``       — multi-agent drift simulator; drift emerges from request
                  pressure, privilege creep, and normalization of deviance
- ``scenarios`` — 8 drift specialists, the enterprise finance-agent scenario,
                  and the support-ticket goal-divergence demo

D(t) = Distance(A(t), R): the distance between actual behavior and remit,
which grows without governance and stays bounded with it.
"""

__all__ = [
    "metrics",
    "policy",
    "remit",
    "scenarios",
    "sim",
    "DriftAgent",
    "DriftConfig",
    "PolicyEngine",
    "Remit",
    "run",
]

from llm_inference_engine.remit_drift import metrics, policy, remit, scenarios, sim
from llm_inference_engine.remit_drift.policy import CONTROLS, PolicyEngine
from llm_inference_engine.remit_drift.remit import Grant, Remit
from llm_inference_engine.remit_drift.sim import DriftAgent, DriftConfig, run
