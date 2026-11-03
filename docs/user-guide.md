# LLM Inference Engine — User Guide

Nine advanced inference systems, implemented from scratch in numpy, benchmarked
with seeded experiments, and reproducible on any laptop. This guide gets you
running in five minutes per project, explains *why* each technique matters, and
helps you pick the right one for your problem.

## Install

```bash
pip install llm-inference-lab
```

Requirements: Python 3.10+, numpy, matplotlib. No GPU, no torch, no frameworks.

```python
import llm_inference_engine as lie
lie.transformer, lie.paged_kv, lie.spec_decode, lie.disagg_serve
lie.quant_lab, lie.moe_router, lie.roofline, lie.sched_lab
lie.remit_drift
```

Every project also ships a runnable quickstart in `examples/`:

```bash
python3 examples/01_transformer_quickstart.py
```

## 1. Nano Transformer — the inference path from scratch

**Why.** Every serving stack (vLLM, TensorRT-LLM, llama.cpp) rests on the same
inference primitives: normalized residual blocks, RoPE, grouped-query
attention, SwiGLU, incremental decoding against a KV cache. This project makes
them visible and *verified* — each primitive checked against an independent
reference, and cached decode proven bit-exact against a full forward pass.

```python
from llm_inference_engine.transformer import (
    KVCache, TransformerConfig, TransformerModel, generate)
import numpy as np

cfg = TransformerConfig(d_model=128, n_layers=4, n_heads=8,
                        n_kv_heads=2, vocab_size=512, max_seq_len=1024)
model = TransformerModel(cfg, rng=np.random.default_rng(7))
cache = KVCache(cfg)
model.prefill(np.array([1, 2, 3, 4]), cache)      # process the prompt once
logits = model.step(5, cache)                     # decode one token, cached
out = generate(model, [1, 2, 3, 4], max_new_tokens=8, seed=7)
```

**Headline numbers.** KV-cache decode vs full forward: max diff **0.0**
(bit-exact). Decode at 512 tokens: **4.72 ms/token cached vs 645.68 ms/token
recompute — 136.8x**. KV memory (fp16): 16 KiB at 32 tokens → 256 KiB at 512.

**Use when** you want to understand what inference *computes* before optimizing
it, or need a reference implementation to validate kernels against.

## 2. Paged KV-Cache — stop stranding memory

**Why.** At admission time you don't know a sequence's final length. Reserving
`max_len` contiguous slots per sequence strands most of your GPU memory as
internal fragmentation. Paging (vLLM's idea) splits the pool into fixed blocks
and allocates `ceil(n/B)` per sequence — waste drops to the tail of one block.

```python
from llm_inference_engine.paged_kv import (
    BlockManager, ContiguousAllocator, gen_workload)
import numpy as np

reqs = gen_workload(np.random.default_rng(11), n=500)
paged = BlockManager(total_slots=200_000, block_size=16)
for r in reqs:
    paged.alloc(r.seq_id, r.length)
print(paged.fragmentation())   # {'internal_waste_pct': 1.94, ...}
```

**Headline numbers** (2000 requests, mixed lengths). Paged B=16: **1.90%**
waste, 817 concurrent, 0 rejected — vs contiguous **81.51%** waste, 488
concurrent, 603 rejected.

**Use when** your server's concurrency is limited by memory, not FLOPS — which
is nearly always in decode-heavy serving.

## 3. Speculative Decoding — bet on acceptance rate

**Why.** Vanilla decoding runs one expensive target forward pass per token. A
cheap draft model proposes `k` tokens; the target verifies them in one pass.
Speedup is `E/(1 + k·c)` where `E = (1−αᵏ⁺¹)/(1−α)` and `c` is the draft/target
cost ratio. The closed-form model here is validated against Monte Carlo, and
`optimal_k` picks the draft length that maximizes speedup — never a slowdown.

```python
from llm_inference_engine.spec_decode import expected_speedup, optimal_k

k, _ = optimal_k(alpha=0.9, c=0.1)          # -> 8
print(expected_speedup(k, 0.9, 0.1))        # -> ~3.4x
```

**Headline numbers.** Best measured: **5.28x** (k=8, α=0.95, c=0.05).
End-to-end sim: 49.5 vs 25.0 tok/s (**1.98x**). Aggressive drafting at low
acceptance *hurts*: 0.79x at α=0.3, k=8.

**Use when** you have a well-aligned draft model (small sibling, early-exit
head) and batch size is small enough that the target pass is bandwidth-bound.

## 4. Disaggregated Serving — split prefill from decode

**Why.** Colocated workers alternate compute-bound prefill and
bandwidth-bound decode, so one long prompt stalls everyone's first token. Two
specialized pools remove the interference — but you pay a KV-cache transfer
over the interconnect. This simulator brackets *exactly* when the win appears.

```python
from llm_inference_engine.disagg_serve import (
    make_workload, run_colocated, run_disaggregated)
import numpy as np

reqs = make_workload(np.random.default_rng(31), 8.0, 200, short_frac=0.2)
print(run_disaggregated(reqs).ttft_ms.mean(), run_colocated(reqs).ttft_ms.mean())
```

**Headline numbers.** Long prompts, high load: TTFT **17.1 ms** disaggregated
vs **40.4 ms** colocated. Crossover: disaggregation wins above **~81-token**
prompts and **~74 GB/s** interconnect.

**Use when** you serve long prompts at high load on fast interconnect —
the classic Splitwise regime. Don't bother for chat-length prompts.

## 5. Quantization Lab — bits vs quality, measured honestly

**Why.** Weight traffic dominates decode cost, so fewer bits per parameter
directly buys speed — but quality degrades. This workbench closes the loop on
a laptop: train a real (tiny) language model from scratch, quantize it with
five schemes, and plot the perplexity-vs-bits Pareto frontier.

```python
from llm_inference_engine.quant_lab.schemes import quantize_int8_per_channel
from llm_inference_engine.quant_lab.metrics import relative_l2_error

wq, meta = quantize_int8_per_channel(weights)
print(relative_l2_error(weights, wq))
```

**Headline numbers** (74k-param char LM, trained 600 Adam steps). INT8
per-channel: perplexity **2.1131 vs 2.1130 baseline — lossless at ~8.2
bits/param**. INT4 group-32: 2.1490 at 4.5 bits. On outlier-structured
weights, per-tensor INT8 is ~8x worse than per-channel — granularity tames
outliers.

**Use when** choosing a serving dtype (INT8 vs FP8 vs INT4), picking group
sizes, or explaining to stakeholders why "just use INT4" isn't free.

## 6. MoE Router — the dropping tax

**Why.** MoE promises big-model capacity at small-model cost, but the router
dispatches tokens into fixed-capacity experts and *drops the overflow*.
With realistic skewed popularity, the default capacity factor silently
discards 61% of top-1 assignments. This project quantifies the three-way
tradeoff: capacity factor, load balancing, and expert-parallel comm cost.

```python
from llm_inference_engine.moe_router import balanced_gate, route, alltoall_bytes

expert_ids, w = balanced_gate(logits, top_k=1, balance_strength=0.99,
                              noise_scale=0.1, rng=rng)
res = route(expert_ids, w, n_experts=8, capacity_factor=1.0)
print(res["drop_rate"], res["imbalance_ratio"])
```

**Headline numbers.** Top-1, CF=1.0: skewed gating drops **61.0%**;
balanced drops **9.3%**. Expert parallel (16 experts, 8 devices): 16.0
MiB/device per step, total comm independent of device count.

**Use when** tuning capacity factor or an auxiliary load-balancing loss, or
estimating the all-to-all cost of scaling experts across nodes.

## 7. Roofline Analysis — find the wall before you optimize

**Why.** "Make it faster" starts with knowing whether you're memory-bound or
compute-bound. This builds a per-op roofline for a full decoder layer on
A100/H100/CPU specs, then estimates what kernel fusion actually buys.

```python
from llm_inference_engine.roofline import (
    A100, LayerConfig, classify, layer_ops, step_summary)

cfg = LayerConfig(d_model=4096, n_heads=32, ffn_mult=2.0, batch=1, seq=2048)
s = step_summary(classify(layer_ops(cfg, mode="decode"), A100))
print(s["n_memory_bound"], "memory-bound,", s["step_time_s"] * 1e6, "µs/step")
```

**Headline numbers.** Decode B=1 on A100: **7/7 ops memory-bound**, ~0.65% of
peak FLOPS, 181.3 µs/step. Fused INT8 dequant+GEMM: **1.831x** decode step.
FlashAttention-style fusion: **1.139x** prefill (decode unchanged at 1.001x —
there's no redundant traffic to kill at B=1).

**Use when** deciding between quantization, batching, and kernel work —
the model tells you which wall you're hitting before you spend a sprint.

## 8. Scheduling Lab — latency is a scheduling decision

**Why.** The scheduler, not the model, decides what users feel. This
iteration-stepped simulator compares static FCFS batching, continuous
(iteration-level) batching, chunked prefill, and priority preemption on
identical workloads.

```python
from llm_inference_engine.sched_lab import make_workload, run_policy
import numpy as np

specs = make_workload(np.random.default_rng(61), n=1500, arrival_rate=6.0)
for pol in ("static-fcfs", "continuous", "chunked-prefill", "priority-preempt"):
    r = run_policy(pol, specs)
    print(pol, np.asarray(r.ttft_ms).mean(), np.percentile(r.ttft_ms, 99))
```

**Headline numbers.** High load: continuous batching TTFT **15.0 ms mean /
57.5 ms p99** vs static **844.6 ms / 2437.1 ms** — 56x/42x. Scheduling moves
*latency*, not capacity (throughput identical at ~1861 tok/s). Chunked
prefill bounds worst-case iteration latency to 25.7 ms (−82.8%).

**Use when** designing or tuning a serving scheduler, or explaining why
vLLM-style continuous batching was the industry's biggest free lunch.

## 9. Remit Drift — when agents outgrow their mandate

**Why.** Agents learn, plan, call tools, delegate, and optimize objectives —
and each mechanism can quietly expand behavior beyond the declared remit.
This lab makes scope drift *measurable*: the remit is a first-class runtime
object (purpose, authority levels, tool/data allow-lists, grant TTLs), D(t)
is computed per divergence type, and a policy engine validates every planned
action (allow/deny/escalate) with nine switchable controls.

```python
from llm_inference_engine.remit_drift import scenarios, sim
from llm_inference_engine.remit_drift.policy import ALL_OFF, ALL_ON, PolicyEngine

agent = scenarios.finance_agent()          # invoice reader -> vendor/bank writes
_, series = sim.run(300, 11, [agent], PolicyEngine(ALL_OFF))
ts, vals = series["functional"]
print("functional D(t):", round(vals[0], 2), "->", round(vals[-1], 2))
```

**Headline numbers.** All 8 divergence types grow ungoverned (functional KL
1.25 → 5.55); the policy engine bounds every one to 0.000. Finance agent:
first out-of-remit write at step 16; governed, 175/175 attempts denied.
Support tickets: close-now rate 0.71 and satisfaction 0.38 ungoverned vs
0.00 and 0.95 governed. Ablation: a written remit alone does *not* contain
drift — central enforcement plus versioning is required.

**Use when** designing agent guardrails, scoping agent authority, or
answering "who authorized this agent to do that?" with numbers.

## 10. LLM Mesh — many models, one routing layer

**Why.** No single model is best at everything: the frontier generalist is
expensive, the reasoning specialist is slow, the cheap chat model is weak
at math. An LLM mesh treats models like a service mesh treats services —
a registry of specialists, a semantic router, an orchestrator that fans
work out in parallel and aggregates, and a self-improving routing policy
that learns which model actually wins per task type.

```python
from llm_inference_engine.llm_mesh import models, router, orchestrator, bandit

reg = models.default_registry()
r = router.SemanticRouter(reg, policy="quality-first")
res = orchestrator.run("Explain this chart and summarize the report", r, reg)
print(res.quality, res.cost_usd, res.trace.model)
```

**Headline numbers.** On 400 mixed requests, quality-first mesh routing
reaches mean quality **0.9430** vs 0.9150 for the frontier model alone, at
**$1.92 vs $5.00 total — 62% cheaper**. Semantic router: **1.0000**
accuracy on held-out queries. Parallel fan-out: **1.64x** speedup. UCB1
router regret falls 0.1566 → 0.0296 over 3000 requests. Kill the vision
model mid-run and mesh success stays **1.000** vs 0.750 for static routing.
The honest tradeoff: quality-first p95 latency is higher (6395 vs 4956 ms)
because reasoning goes to the slowest specialist — the routing policy
chooses which axis you optimize.

**Use when** you serve heterogeneous workloads and want the cost/quality
frontier instead of one-model-fits-all, or when you need routing that
improves itself from observed outcomes.

## When to use which

| If your problem is… | Start with |
|---|---|
| "I want to understand inference mechanics" | 01 Nano Transformer → 07 Roofline |
| "My server runs out of memory" | 02 Paged KV → 08 Scheduling |
| "TTFT is too high on long prompts" | 04 Disagg Serve → 08 Scheduling |
| "Per-token latency too high at low batch" | 03 Spec Decode → 07 Roofline |
| "Model too big / memory bandwidth too high" | 05 Quant Lab → 07 Roofline |
| "MoE dropping tokens or imbalanced" | 06 MoE Router |
| "Agents doing things they shouldn't" | 09 Remit Drift |
| "One model does everything, badly and expensively" | 10 LLM Mesh |

## Reproducibility

Every experiment script is seeded and deterministic: run it twice, get
byte-identical results (the quant-lab training run takes ~4 minutes; the rest
finish in seconds). Results live in each project's `results/` (markdown +
charts), and each project page on the
[landing site](https://ranadhir19.github.io/llm-inference-engine/) walks
through Problem → Gap → Solution → Architecture → Results → Insights →
Implementation.
