# remit-drift results

Seed: 11 (single sequential RNG stream per run). T=300 steps, trailing window=50.

## Experiment 1 -- D(t) without governance (final values)

| divergence type | D(50) | D(300) |
|:---|---:|---:|
| functional | 1.247 | 5.550 |
| authority | 0.200 | 1.000 |
| goal | 0.240 | 0.800 |
| planning | 0.140 | 1.000 |
| tool | 0.140 | 1.000 |
| information | 0.100 | 1.000 |
| collaboration | 0.000 | 0.680 |
| temporal | 0.000 | 1.000 |

## Experiment 2 -- max D with vs without the policy engine

| divergence type | no governance | governed | contained |
|:---|---:|---:|:---:|
| functional | 5.550 | 0.000 | yes |
| authority | 1.000 | 0.000 | yes |
| goal | 0.800 | 0.000 | yes |
| planning | 1.000 | 0.000 | yes |
| tool | 1.000 | 0.000 | yes |
| information | 1.000 | 0.000 | yes |
| collaboration | 0.700 | 0.000 | yes |
| temporal | 1.000 | 0.000 | yes |

## Experiment 3 -- finance-agent timeline

First step each out-of-remit capability was *executed* (no governance):

- change_bank_details: step 16
- create_vendor: step 26
- edit_supplier: step 38

With governance: executed out-of-remit actions: 175 attempts, 0 executed (175 denied, 0 escalated).

## Experiment 4 -- goal divergence (support tickets)

| | no governance | governed |
|:---|---:|---:|
| close-immediately rate | 0.71 | 0.00 |
| final mean response time | 10.7 min | 30.0 min |
| final mean satisfaction | 0.38 | 0.95 |
| proxy/true gap | 0.26 | 0.00 |
| blocked close attempts | 0 | 155 |

## Experiment 5 -- control ablation (D at t=300)

| divergence type | none | all | explicit_remit | capability_registry | policy_engine | tool_allowlist | auth_checks | planner_constraints | auditing | human_approval | remit_versioning |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| functional | 5.550 | 0.000 | 5.550 | 0.000 | 5.550 | 5.550 | 0.000 | 5.550 | 5.550 | 0.000 | 5.550 |
| authority | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| goal | 0.800 | 0.000 | 0.800 | 0.114 | 0.800 | 0.800 | 0.000 | 0.800 | 0.800 | 0.114 | 0.800 |
| planning | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| tool | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| information | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| collaboration | 0.680 | 0.000 | 0.680 | 0.680 | 0.000 | 0.680 | 0.680 | 0.680 | 0.680 | 0.680 | 0.680 |
| temporal | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 |

### Coverage matrix (control catches type?)

| divergence type | explicit_remit | capability_registry | policy_engine | tool_allowlist | auth_checks | planner_constraints | auditing | human_approval | remit_versioning |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| functional | . | Y | . | . | Y | . | Y | Y | . |
| authority | . | . | . | . | Y | . | Y | . | . |
| goal | . | Y | . | . | Y | . | Y | Y | . |
| planning | . | . | . | . | . | Y | Y | . | . |
| tool | . | . | . | Y | . | . | Y | . | . |
| information | . | . | Y | . | . | . | Y | . | . |
| collaboration | . | . | Y | . | . | . | . | . | . |
| temporal | . | . | . | . | Y | . | Y | Y | . |
