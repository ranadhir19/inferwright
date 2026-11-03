"""Smoke tests for remit_drift: determinism, containment, ablation sanity."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from llm_inference_engine.remit_drift import metrics as M  # noqa: E402
from llm_inference_engine.remit_drift import scenarios, sim  # noqa: E402
from llm_inference_engine.remit_drift.policy import (  # noqa: E402
    ALL_OFF,
    ALL_ON,
    CONTROLS,
    PolicyEngine,
)


def _run(dtype, controls, seed=11, T=120):
    agent = scenarios.make_specialist(dtype)
    _, series = sim.run(T, seed, [agent], PolicyEngine(controls))
    return series[dtype][1][-1]


def test_determinism():
    """Two identical runs give identical D(t)."""
    a = _run("functional", ALL_OFF)
    b = _run("functional", ALL_OFF)
    assert a == b


def test_governance_bounds_all_types():
    """Full policy engine keeps every D(T) near zero."""
    for dtype in M.DIVERGENCE_TYPES:
        assert _run(dtype, ALL_ON) < 0.05, dtype


def test_drift_grows_without_governance():
    """Every type shows positive divergence without controls."""
    for dtype in M.DIVERGENCE_TYPES:
        assert _run(dtype, ALL_OFF) > 0.05, dtype


def test_auth_checks_catch_authority():
    d = dict(ALL_OFF)
    d["auth_checks"] = True
    assert _run("authority", d) < 0.25 * _run("authority", ALL_OFF)


def test_tool_allowlist_catches_tool_drift():
    d = dict(ALL_OFF)
    d["tool_allowlist"] = True
    assert _run("tool", d) < 0.25 * _run("tool", ALL_OFF)


def test_planner_constraints_catch_planning():
    d = dict(ALL_OFF)
    d["planner_constraints"] = True
    assert _run("planning", d) < 0.25 * _run("planning", ALL_OFF)


def test_auditing_detects_but_does_not_block():
    d = dict(ALL_OFF)
    d["auditing"] = True
    agent = scenarios.make_specialist("functional")
    events, _ = sim.run(120, 11, [agent], PolicyEngine(ALL_OFF))
    pol = PolicyEngine(d)
    alerts = pol.audit(events[agent.name], agent.remit, 119)
    assert len(alerts) > 0
    assert _run("functional", d) == _run("functional", ALL_OFF)


def test_support_scenario_collapse_and_containment():
    off = scenarios.run_support_scenario(150, 11, PolicyEngine(ALL_OFF))
    on = scenarios.run_support_scenario(150, 11, PolicyEngine(ALL_ON))
    assert off["sat"][-1] < 0.6  # satisfaction collapses ungoverned
    assert on["sat"][-1] > 0.9  # contained when governed
    assert on["close_rate"] == 0.0


def test_finance_timeline_ordering():
    agent = scenarios.finance_agent()
    events, _ = sim.run(300, 11, [agent], PolicyEngine(ALL_OFF))
    first = {}
    for e in events["finance"]:
        k = e["kind"]
        if k not in ("read_invoice", "match_po", "recommend_payment") and e["executed"]:
            first.setdefault(k, e["t"])
    assert set(first) == {"create_vendor", "edit_supplier", "change_bank_details"}
    # governed: nothing out-of-remit executes
    agent2 = scenarios.finance_agent()
    events2, _ = sim.run(300, 11, [agent2], PolicyEngine(ALL_ON))
    bad = [e for e in events2["finance"]
           if e["executed"] and e["kind"] not in ("read_invoice", "match_po", "recommend_payment")]
    assert bad == []


def test_all_controls_listed():
    assert len(CONTROLS) == 9
    assert len(M.DIVERGENCE_TYPES) == 8


if __name__ == "__main__":
    test_determinism()
    print("determinism ok")
    test_governance_bounds_all_types()
    print("governance bounds ok")
    test_drift_grows_without_governance()
    print("drift grows ok")
    test_auth_checks_catch_authority()
    print("auth checks ok")
    test_tool_allowlist_catches_tool_drift()
    print("tool allowlist ok")
    test_planner_constraints_catch_planning()
    print("planner constraints ok")
    test_auditing_detects_but_does_not_block()
    print("auditing ok")
    test_support_scenario_collapse_and_containment()
    print("support scenario ok")
    test_finance_timeline_ordering()
    print("finance timeline ok")
    test_all_controls_listed()
    print("controls listed ok")
    print("ALL SMOKE TESTS PASSED")
