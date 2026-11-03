"""Multi-agent drift simulator.

Each DriftAgent runs over discrete time steps. Drift *emerges* from plausible
mechanisms rather than being scripted:

  - request pressure: the planner asks for out-of-remit work with growing
    probability p_out(t) = 1 - exp(-t/tau) ("nobody objected, so it became normal")
  - privilege creep: attempts above the granted authority level grow linearly
  - tool/data creep: out-of-allowlist, out-of-scope usage grows linearly
  - plan decay: constraint violations per plan grow linearly
  - delegation drift: the delegated fraction of work grows linearly
  - grant decay: a temporary grant expires, but usage continues
  - normalization of deviance: without remit versioning, a repeatedly-attempted
    out-of-remit task is silently absorbed into the agent's effective remit

When the policy engine denies an action, the agent falls back to an in-remit
action (the governed behavior). Denied attempts are logged for audit.
"""

from dataclasses import dataclass, field

import numpy as np

from .policy import Action, PolicyEngine
from .remit import Grant


@dataclass
class DriftConfig:
    tau: float = 80.0
    out_tasks: tuple = ()
    out_authority: dict = field(default_factory=dict)  # out task -> authority level
    out_tools: tuple = ()
    out_data: tuple = ()
    default_tool: str = ""
    default_data: str = ""
    task_tool: dict = field(default_factory=dict)  # task kind -> tool to use
    task_data: dict = field(default_factory=dict)  # task kind -> data to access
    p_auth0: float = 0.0
    g_auth: float = 0.0
    p_tool0: float = 0.0
    g_tool: float = 0.0
    p_data0: float = 0.0
    g_data: float = 0.0
    p_plan0: float = 0.0
    g_plan: float = 0.0
    plan_violation: str = "policy:resilience"
    d0: float = 0.05
    g_deleg: float = 0.0
    grant_at: int | None = None
    grant_ttl: int = 40
    grant_scope: str = "admin:all"
    p_temp0: float = 0.0
    g_temp: float = 0.0
    silent_expand_after: int = 25  # denied attempts before silent remit expansion


class DriftAgent:
    def __init__(self, name, remit, config: DriftConfig):
        self.name = name
        self.remit = remit
        self.cfg = config
        self.registry = set(remit.responsibilities)  # approved capabilities
        self._effective = set(remit.responsibilities)
        self._denied_counts = {}
        self.grants = []
        self.silent_expansions = []  # (kind, t)

    def effective_responsibilities(self):
        return self._effective

    def _maybe_silently_expand(self, kind, t, policy: PolicyEngine):
        if kind in self._effective:
            return
        self._denied_counts[kind] = self._denied_counts.get(kind, 0) + 1
        if not policy.controls["remit_versioning"]:
            if self._denied_counts[kind] >= self.cfg.silent_expand_after:
                # normalization of deviance: the agent's *self-view* of its
                # remit silently absorbs the repeatedly-requested task.
                # The central capability registry does NOT change.
                self._effective.add(kind)
                self.silent_expansions.append((kind, t))
        else:
            policy.request_remit_change(self.name, kind, t)

    def step(self, t, rng, policy: PolicyEngine):
        cfg = self.cfg
        remit = self.remit

        # --- propose a task under request pressure ---
        p_out = (1.0 - np.exp(-t / cfg.tau)) if cfg.out_tasks else 0.0
        if cfg.out_tasks and rng.random() < p_out:
            kind = cfg.out_tasks[rng.integers(len(cfg.out_tasks))]
        else:
            kind = rng.choice(remit.responsibilities)
        in_remit_task = kind in remit.responsibilities

        # --- authority level (privilege creep) ---
        base_level = remit.max_level(kind) if in_remit_task else cfg.out_authority.get(kind, 2)
        level = base_level
        if rng.random() < min(1.0, cfg.p_auth0 + cfg.g_auth * t):
            level = min(3, base_level + 1)

        # --- tool / data (boundary creep) ---
        tool = cfg.default_tool or (remit.tool_allowlist[0] if remit.tool_allowlist else "")
        data = cfg.default_data or (remit.data_allowlist[0] if remit.data_allowlist else "")
        if kind in cfg.task_tool:
            tool = cfg.task_tool[kind]
        elif cfg.out_tools and rng.random() < min(1.0, cfg.p_tool0 + cfg.g_tool * t):
            tool = cfg.out_tools[rng.integers(len(cfg.out_tools))]
        if kind in cfg.task_data:
            data = cfg.task_data[kind]
        elif cfg.out_data and rng.random() < min(1.0, cfg.p_data0 + cfg.g_data * t):
            data = cfg.out_data[rng.integers(len(cfg.out_data))]

        # --- plan violations ---
        violations = ()
        if rng.random() < min(1.0, cfg.p_plan0 + cfg.g_plan * t):
            violations = (cfg.plan_violation,)

        # --- delegation ---
        deleg_ratio = min(0.98, cfg.d0 + cfg.g_deleg * t)
        delegated = bool(rng.random() < deleg_ratio)

        # --- temporary grant lifecycle ---
        grant = None
        grant_expired_used = False
        if cfg.grant_at is not None:
            if t == cfg.grant_at:
                self.grants.append(Grant(cfg.grant_scope, 3, t, cfg.grant_ttl))
            live = [g for g in self.grants if g.valid_at(t)]
            expired = [g for g in self.grants if g.expired(t)]
            if live:
                grant = live[0]
                level = max(level, 3)
            elif expired and rng.random() < min(1.0, cfg.p_temp0 + cfg.g_temp * t):
                grant = expired[0]
                level = max(level, 3)
                grant_expired_used = True

        high_risk = (not in_remit_task) or level >= 3
        action = Action(kind=kind, authority=level, tool=tool, data=data,
                        plan_violations=violations, delegated=delegated,
                        grant=grant, high_risk=high_risk)
        decision = policy.validate(self, action, t, deleg_ratio)

        executed = None
        if decision.blocked:
            # normalization of deviance: repeated denied attempts silently expand remit
            if not in_remit_task:
                self._maybe_silently_expand(kind, t, policy)
            # governed fallback: do compliant in-remit work instead
            fallback = Action(
                kind=rng.choice(remit.responsibilities),
                authority=remit.max_level(rng.choice(remit.responsibilities)),
                tool=remit.tool_allowlist[0] if remit.tool_allowlist else "",
                data=remit.data_allowlist[0] if remit.data_allowlist else "",
                plan_violations=(), delegated=False, grant=None, high_risk=False)
            fb_decision = policy.validate(self, fallback, t, 0.0)
            executed = fallback if fb_decision.verdict == "allow" else None
        else:
            executed = action

        compliant = (
            executed is not None
            and executed.kind in remit.responsibilities
            and not executed.plan_violations
            and executed.authority <= remit.max_level(executed.kind)
            and remit.allows_tool(executed.tool)
            and remit.allows_data(executed.data)
            and not grant_expired_used
        )
        event = {
            "t": t,
            "kind": executed.kind if executed else kind,
            "authority": executed.authority if executed else level,
            "tool": executed.tool if executed else tool,
            "data": executed.data if executed else data,
            "plan_violations": executed.plan_violations if executed else violations,
            "delegated": executed.delegated if executed else delegated,
            "grant_expired_used": grant_expired_used and decision.verdict == "allow",
            "executed": executed is not None,
            "attempted_kind": kind,
            "decision": decision.verdict,
            "control": decision.control,
            "proxy_r": 1.0 if executed else 0.0,
            "true_r": 1.0 if (executed and compliant) else (0.2 if executed else 0.0),
        }
        return event


def run(T, seed, agents, policy, window=50, stride=5):
    """Run agents for T steps. Returns (events_by_agent, series) where series maps
    divergence type -> D(t) array evaluated every `stride` steps."""
    from . import metrics as M
    rng = np.random.default_rng(seed)
    events = {a.name: [] for a in agents}
    for t in range(T):
        for a in agents:
            events[a.name].append(a.step(t, rng, policy))
    ts = list(range(window, T + 1, stride))
    series = {}
    for dtype in M.DIVERGENCE_TYPES:
        fn = M.METRIC_FNS[dtype]
        vals = []
        for t in ts:
            per_agent = []
            for a in agents:
                ev = [e for e in M.trailing(events[a.name], t, window) if e["executed"]]
                per_agent.append(fn(ev, a.remit))
            vals.append(float(np.mean(per_agent)))
        series[dtype] = (np.array(ts), np.array(vals))
    return events, series
