"""Quickstart: divergence of remit, measured and governed.

An expense agent drifts (privilege creep); a finance agent drifts
(functional scope creep). Without governance D(t) grows; with the policy
engine every planned action is validated and divergence stays bounded.
"""
import numpy as np

from llm_inference_engine.remit_drift import scenarios, sim
from llm_inference_engine.remit_drift.policy import ALL_OFF, ALL_ON, PolicyEngine

T, SEED = 200, 11

print("== authority drift: expense agent (recommend -> approve) ==")
for label, controls in (("no governance", ALL_OFF), ("policy engine", ALL_ON)):
    agent = scenarios.make_specialist("authority")
    _, series = sim.run(T, SEED, [agent], PolicyEngine(controls))
    ts, vals = series["authority"]
    print(f"  {label:14s} D(50)={vals[0]:.2f}  D(200)={vals[-1]:.2f}")

print("== functional drift: finance agent (invoices -> vendors/bank) ==")
for label, controls in (("no governance", ALL_OFF), ("policy engine", ALL_ON)):
    agent = scenarios.finance_agent()
    events, _ = sim.run(T, SEED, [agent], PolicyEngine(controls))
    bad = [e for e in events["finance"]
           if e["executed"] and e["kind"] not in
           ("read_invoice", "match_po", "recommend_payment")]
    first = min((e["t"] for e in bad), default=None)
    print(f"  {label:14s} out-of-remit executed={len(bad):3d}  first at step={first}")

print("== goal drift: support tickets (proxy vs true objective) ==")
for label, controls in (("no governance", ALL_OFF), ("policy engine", ALL_ON)):
    r = scenarios.run_support_scenario(T, SEED, PolicyEngine(controls))
    sat_tail = r["sat"][-50:].mean()
    print(f"  {label:14s} close-now rate={r['close_rate']:.2f}  "
          f"mean satisfaction (last 50)={sat_tail:.2f}")

print("Remit as a runtime object: validate before execution, and D(t) stays bounded.")
