"""Policy engine: validates every planned action against the declared remit.

Verdicts: "allow" | "deny" | "escalate". Denied/escalated actions never
execute, so the agent's *actual* behavior A(t) cannot silently diverge from
its remit R while the relevant controls are enabled.

Controls (each independently switchable for ablation):
  explicit_remit      - task kind must be in remit.responsibilities
  capability_registry - task kind must be in the approved capability registry
  policy_engine       - runtime validation switch + data-scope enforcement
  tool_allowlist      - tool must be in remit.tool_allowlist
  auth_checks         - authority level within remit; grants must be unexpired
  planner_constraints - plans must not violate named constraints
  auditing            - detect-only: raises alerts, never blocks
  human_approval      - high-risk actions escalate (blocked in simulation)
  remit_versioning    - silent remit expansion is rejected; expansion needs
                        approval (in simulation: never granted)
"""

from dataclasses import dataclass

from .remit import Grant

CONTROLS = (
    "explicit_remit",
    "capability_registry",
    "policy_engine",
    "tool_allowlist",
    "auth_checks",
    "planner_constraints",
    "auditing",
    "human_approval",
    "remit_versioning",
)

ALL_ON = {c: True for c in CONTROLS}
ALL_OFF = {c: False for c in CONTROLS}


@dataclass(frozen=True)
class Action:
    kind: str
    authority: int
    tool: str
    data: str
    plan_violations: tuple = ()
    delegated: bool = False
    grant: Grant | None = None
    high_risk: bool = False


@dataclass(frozen=True)
class Decision:
    verdict: str  # "allow" | "deny" | "escalate"
    control: str  # control that decided
    reason: str

    @property
    def blocked(self) -> bool:
        return self.verdict in ("deny", "escalate")


class PolicyEngine:
    def __init__(self, controls=None):
        self.controls = dict(ALL_ON if controls is None else controls)
        for c in CONTROLS:
            self.controls.setdefault(c, False)
        self.version_requests = []  # (agent_name, kind, t) log

    def validate(self, agent, action: Action, t: int, deleg_ratio: float = 0.0) -> Decision:
        c = self.controls
        remit = agent.remit
        allowed_kinds = agent.effective_responsibilities()

        # Each control gates its own check independently, so ablation is meaningful.
        # 1. remit scope
        if c["explicit_remit"] and action.kind not in allowed_kinds:
            return Decision("deny", "explicit_remit",
                            f"task '{action.kind}' outside remit '{remit.name}'")
        # 2. capability registry (approved capabilities only)
        if c["capability_registry"] and action.kind not in agent.registry:
            return Decision("deny", "capability_registry",
                            f"capability '{action.kind}' not registered")
        # 3. tool allow-list
        if c["tool_allowlist"] and not remit.allows_tool(action.tool):
            return Decision("deny", "tool_allowlist",
                            f"tool '{action.tool}' not in allow-list")
        # 4. data scope (part of the remit; enforced by the runtime validator)
        if c["policy_engine"] and not remit.allows_data(action.data):
            return Decision("deny", "policy_engine",
                            f"data '{action.data}' outside remit scope")
        # 5. authorization: level + grant expiry
        if c["auth_checks"]:
            if action.grant is not None and action.grant.expired(t):
                return Decision("deny", "auth_checks",
                                f"grant '{action.grant.scope}' expired at step {t}")
            if action.authority > remit.max_level(action.kind):
                return Decision("deny", "auth_checks",
                                f"authority {action.authority} exceeds remit max "
                                f"{remit.max_level(action.kind)} for '{action.kind}'")
        # 6. planner constraints
        if c["planner_constraints"] and action.plan_violations:
            return Decision("deny", "planner_constraints",
                            f"plan violates {action.plan_violations}")
        # 7. delegation cap (enforced by the runtime validator)
        if c["policy_engine"] and action.delegated and deleg_ratio > remit.max_delegation:
            return Decision("deny", "policy_engine",
                            f"delegation ratio {deleg_ratio:.2f} exceeds remit cap "
                            f"{remit.max_delegation:.2f}")
        # 8. human approval for high-risk actions
        if c["human_approval"] and action.high_risk:
            return Decision("escalate", "human_approval",
                            f"high-risk action '{action.kind}' needs human approval")
        return Decision("allow", "policy_engine", "within remit")

    def audit(self, events, remit, t: int, window: int = 50):
        """Detect-only pass. Returns alert strings; never blocks."""
        from .metrics import goal_gap, trailing
        alerts = []
        if not self.controls["auditing"]:
            return alerts
        recent = trailing(events, t, window)
        if len(recent) < 10:
            return alerts
        gap = goal_gap(recent)
        if gap > 0.30:
            alerts.append(f"proxy/true objective gap {gap:.2f} — possible goal divergence")
        out_kind = sum(1 for e in recent if e["kind"] not in remit.responsibilities)
        if out_kind / len(recent) > 0.20:
            alerts.append(f"{out_kind/len(recent):.0%} of actions outside remit — "
                          f"possible functional divergence")
        expired = sum(1 for e in recent if e.get("grant_expired_used"))
        if expired:
            alerts.append(f"{expired} actions used expired grants — temporal divergence")
        return alerts

    def request_remit_change(self, agent_name: str, kind: str, t: int) -> bool:
        """Remit versioning: expansion needs approval. Always denied in simulation."""
        self.version_requests.append((agent_name, kind, t))
        return False  # no silent expansion, ever
