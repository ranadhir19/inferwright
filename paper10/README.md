# Paper 10 — deposited artefacts

**Paper:** "When Paging Meets Speculation: A Simulation Study of KV-Cache Paging and Speculative Decoding Driven by a Real Production Trace"

This folder holds the paper-specific scripts and result files for the study, deposited alongside the simulator suites in this repository (`02-paged-kv` and `03-spec-decode`). The scripts drive the shipped allocators and speculative-decoding formulas from the repository's `src/` tree and run against the public BurstGPT production trace.

## Files

- `analysis.py` — strengthening analyses: real-trace paged run, unified-model validation, and the negative-regime checks. Writes `paper10_results.json`.
- `paper10_results.json` — results produced by `analysis.py`.
- `audit_probes.py` — independent audit probes: recomputes the headline trace numbers with a waste-definition reconciliation, runs the true joint simulation (speculative speed-up applied inside the allocator simulation), and checks trace-slice and budget sensitivity. Writes `audit_results.json`.
- `audit_results.json` — results produced by `audit_probes.py`.
- `strengthening_probes.py` — strengthening probes: representative trace sampling, cap-bias accounting, the prefill+decode residence model, consistent product-vs-joint comparison, and the waste-definition check on the low-load case. Writes `strengthening_results.json` and the scaled joint-study results in `scaled_joint_results.json`.
- `strengthening_results.json` — results produced by `strengthening_probes.py`.
- `scaled_joint_results.json` — scaled joint-study results (product-vs-joint comparison at the headline scale).
- `cycle2_probes.py` — Cycle-2 robustness harness: first re-derives the joint study from the trace and asserts equality with the deposited joint-study results, then runs the trace-arrival, incremental-allocator, and second-sample variants. Writes `cycle2_results.json`.
- `cycle2_results.json` — results produced by `cycle2_probes.py`.
- `hw_spec.py` — hardware-anchor script: a measured CPU speculative-decoding run (GPT-2 target, DistilGPT-2 draft, greedy decoding with exact-output losslessness check). Writes `hw_results.json`.
- `hw_results.json` — results produced by `hw_spec.py`.

Every table and figure in the paper can be regenerated from these artefacts together with the suites in this repository.
