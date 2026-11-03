# LLM Inference Engine

Ten advanced LLM inference systems, implemented from first principles in
numpy, benchmarked with seeded experiments, and reproducible on any laptop.

**Landing site:** https://ranadhir19.github.io/llm-inference-engine/
· **Install:** `pip install llm-inference-lab`

## Watch it built

The systems in this repo are built and measured on camera at **[The Architect's AI Workshop](https://www.youtube.com/@ArchitectsAIWorkshop)** — *first principles on the whiteboard, built live in the editor, measured honestly.* Browse the episodes: [youtube.com/@ArchitectsAIWorkshop/videos](https://www.youtube.com/@ArchitectsAIWorkshop/videos)

Episodes from this repo:

- [81.51% of This KV Cache Was Reserved for Nothing](https://youtu.be/hQoqXaM8MR8) — paged-kv

## The systems

| # | Project | What it answers |
|---|---|---|
| 01 | [Nano Transformer](01-nano-transformer/) | Can I build a LLaMA-style decoder from scratch and prove the KV cache is bit-exact? |
| 02 | [Paged KV-Cache](02-paged-kv/) | How much memory does contiguous allocation strand vs paging? |
| 03 | [Speculative Decoding](03-spec-decode/) | When does draft+verify beat vanilla autoregressive — and when does it backfire? |
| 04 | [Disaggregated Serving](04-disagg-serve/) | When does splitting prefill from decode pay for the KV transfer? |
| 05 | [Quantization Lab](05-quant-lab/) | What's the real perplexity cost of INT8/FP8/INT4? |
| 06 | [MoE Router](06-moe-router/) | How many tokens does expert capacity silently drop? |
| 07 | [Roofline Analysis](07-roofline/) | Where is the wall — memory or compute — and what does fusion move? |
| 08 | [Scheduling Lab](08-sched-lab/) | Continuous vs chunked vs preemptive scheduling: who feels what? |
| 09 | [Remit Drift](09-remit-drift/) | When agents drift from their remit — measured per type, governed by a policy engine |
| 10 | [LLM Mesh](10-llm-mesh/) | Many specialized models, one routing layer: route, orchestrate, aggregate |

Four categories: **From First Principles** (01, 07) · **Serving at Scale**
(02, 03, 04, 08, 10) · **Efficiency Frontiers** (05, 06) · **Agent Governance** (09).

## Results showcase

- **Nano Transformer:** KV-cache decode proven bit-exact (max diff 0.0);
  4.72 ms/token cached vs 645.68 recompute at 512 tokens — **136.8x**.
- **Paged KV:** **1.90%** waste vs 81.51% contiguous; 817 vs 488 concurrent;
  0 vs 603 rejected.
- **Speculative decoding:** **5.28x** best speedup; analytical model matches
  Monte Carlo within 0.018 tokens; aggressive drafting at low acceptance
  *slows down* (0.79x).
- **Disaggregated serving:** TTFT **17.1 ms vs 40.4 ms** on long prompts;
  wins above ~81-token prompts and ~74 GB/s interconnect.
- **Quant lab:** INT8 per-channel **lossless** (2.1131 vs 2.1130 perplexity)
  at ~8.2 bits/param; INT4-g32 at 2.1490 and 4.5 bits.
- **MoE router:** balanced gating drops **9.3%** vs 61.0% skewed at CF=1.0 —
  balancing beats raw capacity.
- **Roofline:** decode **7/7 memory-bound** at ~0.65% of A100 peak; fused
  INT8 dequant+GEMM gives **1.831x** decode step.
- **Scheduling:** continuous batching cuts high-load TTFT **56x mean /
  42x p99**; throughput identical — scheduling moves latency, not capacity.
- **Remit drift:** all 8 divergence types grow ungoverned (functional KL
  **5.55**); policy engine bounds every one to **0.000** — finance agent's
  vendor/bank writes denied **175/175**.
- **LLM Mesh:** quality-first mesh routing reaches **0.943** mean quality vs
  0.915 for a single frontier model at **$1.92 vs $5.00 (62% cheaper)**;
  UCB1 self-improving router cuts regret 0.1566 → 0.0296; failover keeps
  success at **1.000** when a model goes down.

Every number above comes from an actual seeded run in the repo's
`results/` folders.

## Quickstart

```bash
pip install llm-inference-lab
```

```python
from llm_inference_engine import transformer, paged_kv, spec_decode, sched_lab

# LLaMA-style decoder from scratch, KV cache bit-exact vs full forward
cfg = transformer.TransformerConfig(d_model=128, n_layers=4, n_heads=8,
                                    n_kv_heads=2, vocab_size=512, max_seq_len=1024)
model = transformer.TransformerModel(cfg, rng=__import__("numpy").random.default_rng(7))
cache = transformer.KVCache(cfg)
model.prefill(__import__("numpy").array([1, 2, 3, 4]), cache)
model.step(5, cache)   # one cached decode step
```

Per-project quickstarts in `examples/` (all runnable):

```bash
for f in examples/*.py; do python3 "$f"; done
```

## Reproducibility

Each project ships `experiments/run_experiments.py` (seeded, numpy-only) and
`tests/test_smoke.py`. Run twice, get byte-identical results — the headline
numbers are asserted before anything is written.

## Repo layout

```
src/llm_inference_engine/   # the pip package: transformer, paged_kv,
                            #   spec_decode, disagg_serve, quant_lab,
                            #   moe_router, roofline, sched_lab,
                            #   remit_drift, llm_mesh
01-nano-transformer/ ... 10-llm-mesh/
  experiments/  results/  tests/  README.md
examples/                   # one runnable quickstart per project
docs/                       # landing site (build.py) + user-guide.md
```

## License

MIT — Ranadhir Ghosh.
