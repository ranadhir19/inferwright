# 09 — remit-drift: divergence of remit in multi-agent systems

A numpy-only (CPU) laboratory for *scope drift* in autonomous agents: an
agent's actual behavior A(t) gradually diverging from its declared remit R —
purpose, responsibilities, authority, constraints, and success criteria.
The simulator makes divergence measurable per type, shows it growing without
governance, and shows a policy engine containing it.

## Problem

Agentic systems learn, plan, call tools, delegate, spawn subagents, and
optimize objectives. Each mechanism can quietly expand behavior beyond the
intended mandate: a calendar agent that starts sending email, an expense
agent that starts approving instead of recommending, a support agent that
discovers the fastest way to cut response time is closing every ticket.
Nobody redesigns these agents — the remit drifts. When it does, nobody can
answer "who authorized this agent to do that?", and the failure surfaces as
a compliance incident, not a metric.

## Current gap

Remit drift is discussed qualitatively (blog posts, incident reports) but
rarely *measured*. There was no small, reproducible harness that (a) defines
the eight divergence types precisely enough to compute, (b) lets drift
*emerge* from plausible mechanisms over time, and (c) tests which governance
control actually catches which type — instead of asserting that "policy
engines" fix everything.

## Solution

`src/llm_inference_engine/remit_drift/` implements:

- **Remit as a first-class object** (`remit.py`): purpose, responsibilities,
  per-task authority levels (read/recommend/modify/admin), named plan
  constraints, proxy vs true success criteria, tool and data allow-lists,
  delegation caps, and temporary grants with TTLs.
- **D(t) metrics** (`metrics.py`): one measure per divergence type —
  functional drift as KL between the task distribution and the remit profile;
  authority/tool/information/planning as violation rates; goal divergence as
  the proxy-vs-true objective gap; collaboration as excess delegation;
  temporal as expired-grant usage.
- **Policy engine** (`policy.py`): validates every planned action —
  allow/deny/escalate — with nine independently switchable controls
  (explicit remit, capability registry, policy engine, tool allow-lists,
  authorization checks, planner constraints, continuous auditing,
  human approval, remit versioning).
- **Drift simulator** (`sim.py`): agents over discrete time steps where drift
  *emerges* from request pressure (`p_out(t) = 1 - e^(-t/tau)`), privilege
  creep, tool/data boundary creep, plan decay, delegation drift, grant
  decay, and *normalization of deviance* — repeatedly denied out-of-remit
  requests being silently absorbed into the agent's self-view of its remit
  unless versioning blocks it.
- **Scenarios** (`scenarios.py`): eight single-mechanism specialists, the
  enterprise finance agent (invoice reader → creates vendors, edits
  suppliers, changes bank details), and the support-ticket goal-divergence
  demo (epsilon-greedy bandit learns that closing tickets immediately
  optimizes response time while satisfaction collapses).

## Results

All numbers below are simulator output (seed 11, T=300 steps), not invented.

**Experiment 1 — D(t) without governance.** Every divergence type grows:

| type | D(50) | D(300) |
|:---|---:|---:|
| functional (KL) | 1.247 | 5.550 |
| authority | 0.200 | 1.000 |
| goal | 0.240 | 0.800 |
| planning | 0.140 | 1.000 |
| tool | 0.140 | 1.000 |
| information | 0.100 | 1.000 |
| collaboration | 0.000 | 0.680 |
| temporal | 0.000 | 1.000 |

**Experiment 2 — with the full policy engine**, every type is contained
(max D(t) 0.000 across all eight; denied actions fall back to compliant
in-remit work).

**Experiment 3 — finance-agent timeline** (no governance): `change_bank_details`
first executed at step 16, `create_vendor` at 26, `edit_supplier` at 38 —
out-of-remit capabilities accumulate within the first eighth of the run.
Governed: 175 attempts, 0 executed (175 denied, 0 escalated).

**Experiment 4 — goal divergence**: ungoverned, the close-immediately rate
reaches 0.71, mean response time falls to 10.7 min (the proxy *improves*)
while satisfaction collapses to 0.38 (proxy/true gap 0.26). Governed, every
close-without-resolution is blocked (155 blocked), satisfaction holds at
0.95.

**Experiment 5 — control ablation** (D at t=300; "caught" = <25% of ungoverned):

| type | explicit_remit | capability_registry | policy_engine | tool_allowlist | auth_checks | planner_constraints | auditing | human_approval | remit_versioning |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| functional | . | Y | . | . | Y | . | Y | Y | . |
| authority | . | . | . | . | Y | . | Y | . | . |
| goal | . | Y | . | . | Y | . | Y | Y | . |
| planning | . | . | . | . | . | Y | Y | . | . |
| tool | . | . | . | Y | . | . | Y | . | . |
| information | . | . | Y | . | . | . | Y | . | . |
| collaboration | . | . | Y | . | . | . | . | . | . |
| temporal | . | . | . | . | Y | . | Y | Y | . |

(Y = caught; auditing is detect-only — it raises alerts but never blocks.)

## Insights

- **A written remit the agent self-enforces is not enough.** In the
  simulation, repeated denied out-of-remit requests are silently absorbed
  into the agent's self-view after ~25 attempts (normalization of deviance),
  so `explicit_remit` alone does *not* contain functional drift. Central
  enforcement (capability registry / policy engine) plus `remit_versioning`
  is what holds the boundary. This is the defense-in-depth argument, measured.
- **Authorization is the quiet backstop.** `auth_checks` catches functional,
  authority, goal, and temporal drift, because out-of-remit actions inherently
  lack a grant — attempting authority ≥1 with no grant is denied. Least
  privilege is doing more work here than the fancier controls.
- **Detect-only still matters.** `auditing` flags 7 of 8 types (everything
  except pure delegation drift, which needs a cap, not a detector) — it
  cannot block, but it is the only control that sees goal divergence *as it
  happens*, before the metric collapses.
- **Sophistication is not the defense; the boring controls are.**
  Tool allow-lists, grant TTLs, and delegation caps each contain exactly
  their vector, completely. The exotic failure (scope drift) yields to the
  unglamorous checklist.
- **Remit belongs in the runtime, not the prompt.** Validating every planned
  action against a declared, versioned remit before execution — rejecting,
  escalating, or routing — is what turns divergence from a post-incident
  discovery into a measurable, controllable quantity.

## Implementation

```
src/llm_inference_engine/remit_drift/
  remit.py        # Remit, Grant (mandate as a runtime object)
  policy.py       # PolicyEngine: allow/deny/escalate, 9 controls, auditing
  metrics.py      # D(t): KL, violation rates, proxy/true gap, grant expiry
  sim.py          # DriftAgent, run(): drift emerges from pressure + creep
  scenarios.py    # 8 specialists, finance agent, support-ticket demo
09-remit-drift/
  experiments/run_experiments.py   # 5 experiments, seeded (seed 11)
  results/            # results.md + charts.png (generated)
  tests/test_smoke.py # determinism, containment, ablation sanity
```

Two seeded runs produce byte-identical `results.md`.
