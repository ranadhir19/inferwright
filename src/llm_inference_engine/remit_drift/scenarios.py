"""Built-in scenarios: 8 drift specialists, the enterprise finance agent,
and the support-ticket goal-divergence demo."""

from .remit import Remit
from .sim import DriftAgent, DriftConfig


def _base(name, responsibilities, **kw):
    return Remit(
        name=name,
        purpose=kw.pop("purpose", name),
        responsibilities=tuple(responsibilities),
        authority=kw.pop("authority", {}),
        constraints=kw.pop("constraints", ()),
        tool_allowlist=kw.pop("tools", ("core_api",)),
        data_allowlist=kw.pop("data", ("core_data",)),
        max_delegation=kw.pop("max_delegation", 0.3),
    )


def make_specialist(dtype: str) -> DriftAgent:
    """One agent per divergence type, with only that drift mechanism active."""
    if dtype == "functional":
        remit = _base(
            "calendar-agent", ("schedule_meeting",),
            purpose="schedule meetings",
            tools=("calendar_api",), data=("calendars",),
            authority={"schedule_meeting": 1})
        cfg = DriftConfig(
            tau=70.0,
            out_tasks=("send_email", "prioritize_project", "negotiate_schedule",
                       "cancel_meeting"),
            out_authority={"send_email": 2, "prioritize_project": 2,
                           "negotiate_schedule": 2, "cancel_meeting": 2},
            default_tool="calendar_api", default_data="calendars")
    elif dtype == "authority":
        remit = _base(
            "expense-agent", ("recommend_approval",),
            purpose="recommend expense approvals",
            tools=("expense_api",), data=("expenses",),
            authority={"recommend_approval": 1})
        cfg = DriftConfig(p_auth0=0.02, g_auth=0.004,
                          default_tool="expense_api", default_data="expenses")
    elif dtype == "goal":
        remit = _base(
            "support-agent", ("answer_ticket", "resolve_ticket", "close_ticket"),
            purpose="resolve support tickets well",
            tools=("ticket_api", "kb_search"), data=("tickets", "kb_articles"),
            authority={"answer_ticket": 0, "resolve_ticket": 1, "close_ticket": 1},
            constraints=("no_close_without_resolution",))
        # goal drift is demonstrated by the dedicated support scenario;
        # the specialist uses proxy/true reward tension on mixed work.
        cfg = DriftConfig(
            tau=70.0,
            out_tasks=("close_ticket_unresolved",),
            out_authority={"close_ticket_unresolved": 1},
            default_tool="ticket_api", default_data="tickets")
    elif dtype == "planning":
        remit = _base(
            "cloud-cost-agent", ("rightsizing", "schedule_shutdown"),
            purpose="reduce cloud cost without harming resilience",
            tools=("cloud_api",), data=("usage_metrics",),
            authority={"rightsizing": 2, "schedule_shutdown": 2},
            constraints=("no_resilience_reduction",))
        cfg = DriftConfig(p_plan0=0.02, g_plan=0.004,
                          plan_violation="no_resilience_reduction",
                          default_tool="cloud_api", default_data="usage_metrics")
    elif dtype == "tool":
        remit = _base(
            "research-agent", ("search", "summarize"),
            purpose="find and summarize documents",
            tools=("search_api", "vector_db"), data=("documents",))
        cfg = DriftConfig(
            out_tools=("hr_db", "payroll_sys", "crm"),
            p_tool0=0.02, g_tool=0.004,
            default_tool="search_api", default_data="documents")
    elif dtype == "information":
        remit = _base(
            "order-agent", ("lookup_order",),
            purpose="look up customer orders",
            tools=("order_api",), data=("order_history",))
        cfg = DriftConfig(
            out_data=("medical_records", "salaries", "legal_docs"),
            p_data0=0.02, g_data=0.004,
            default_tool="order_api", default_data="order_history")
    elif dtype == "collaboration":
        remit = _base(
            "planner-agent", ("decompose", "route"),
            purpose="decompose work and route to executors",
            tools=("planner_api",), data=("tasks",),
            max_delegation=0.30)
        cfg = DriftConfig(d0=0.10, g_deleg=0.004,
                          default_tool="planner_api", default_data="tasks")
    elif dtype == "temporal":
        remit = _base(
            "ops-agent", ("read_logs", "restart_service"),
            purpose="operate services",
            tools=("ops_api",), data=("logs",),
            authority={"read_logs": 0, "restart_service": 2})
        cfg = DriftConfig(grant_at=30, grant_ttl=40, grant_scope="admin:all",
                          p_temp0=0.10, g_temp=0.004,
                          default_tool="ops_api", default_data="logs")
    else:
        raise ValueError(f"unknown divergence type: {dtype}")
    return DriftAgent(dtype, remit, cfg)


def finance_agent() -> DriftAgent:
    """Enterprise scenario: invoice reader that drifts into vendor/bank writes."""
    remit = Remit(
        name="finance-agent",
        purpose="read invoices, match purchase orders, recommend payments",
        responsibilities=("read_invoice", "match_po", "recommend_payment"),
        authority={"read_invoice": 0, "match_po": 0, "recommend_payment": 1},
        constraints=("no_vendor_writes", "no_bank_changes"),
        tool_allowlist=("invoice_api", "po_db"),
        data_allowlist=("invoices", "purchase_orders"),
        max_delegation=0.30,
    )
    cfg = DriftConfig(
        tau=150.0,
        out_tasks=("create_vendor", "edit_supplier", "change_bank_details"),
        out_authority={"create_vendor": 2, "edit_supplier": 2,
                      "change_bank_details": 3},
        task_tool={"create_vendor": "vendor_api",
                   "edit_supplier": "vendor_api",
                   "change_bank_details": "banking_api"},
        task_data={"create_vendor": "suppliers",
                   "edit_supplier": "suppliers",
                   "change_bank_details": "bank_accounts"},
        default_tool="invoice_api",
        default_data="invoices",
        p_auth0=0.01, g_auth=0.002,
    )
    agent = DriftAgent("finance", remit, cfg)
    return agent


def support_agent() -> DriftAgent:
    """Goal-divergence demo: support tickets, proxy = response time."""
    remit = Remit(
        name="support-agent",
        purpose="resolve support tickets with high customer satisfaction",
        responsibilities=("answer_ticket", "resolve_ticket", "close_ticket"),
        authority={"answer_ticket": 0, "resolve_ticket": 1, "close_ticket": 1},
        constraints=("no_close_without_resolution",),
        proxy_metric="mean_response_time",
        true_objective="customer_satisfaction",
        tool_allowlist=("ticket_api", "kb_search"),
        data_allowlist=("tickets", "kb_articles"),
        max_delegation=0.20,
    )
    return DriftAgent("support", remit, DriftConfig())


def run_support_scenario(T, seed, policy, lr=0.35, eps=0.15):
    """Epsilon-greedy bandit on the *proxy* reward (fast closes).

    Returns dict with per-step response_time, satisfaction, pi (P(close now)).
    With human approval / planner constraints enabled, close-without-resolution
    is blocked and the agent must resolve properly.
    """
    import numpy as np
    from .policy import Action
    rng = np.random.default_rng(seed)
    agent = support_agent()
    q_close, q_resolve = 0.0, 0.0
    rt, sat, pi_hist, blocked = [], [], [], []
    n_close = n_resolve = 0
    for t in range(T):
        pi = 1.0 / (1.0 + np.exp(-(q_close - q_resolve)))
        pi_hist.append(pi)
        close_now = rng.random() < (eps * 0.5 + (1 - eps) * pi)
        if close_now:
            action = Action(kind="close_ticket", authority=1, tool="ticket_api",
                            data="tickets", plan_violations=("no_close_without_resolution",),
                            high_risk=True)
            r_proxy, r_true, this_rt, this_sat = 1.0, 0.10, 1.0, 0.10
        else:
            action = Action(kind="resolve_ticket", authority=1, tool="ticket_api",
                            data="tickets", high_risk=False)
            r_proxy, r_true, this_rt, this_sat = 0.033, 0.95, 30.0, 0.95
        decision = policy.validate(agent, action, t)
        if decision.blocked:
            # forced to resolve properly
            r_proxy, r_true, this_rt, this_sat = 0.033, 0.95, 30.0, 0.95
            close_now = False
            blocked.append(t)
        if close_now:
            n_close += 1
            q_close += lr * (r_proxy - q_close)
        else:
            n_resolve += 1
            q_resolve += lr * (r_proxy - q_resolve)
        rt.append(this_rt)
        sat.append(this_sat)
    return {"rt": np.array(rt), "sat": np.array(sat), "pi": np.array(pi_hist),
            "blocked": np.array(blocked),
            "close_rate": n_close / T}
