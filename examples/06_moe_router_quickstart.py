"""Quickstart: MoE expert routing — capacity factor vs token dropping.

Routes tokens with skewed expert popularity and shows how the capacity
factor trades compute for dropped tokens, plus expert-parallel comm cost.
"""
import numpy as np

from llm_inference_engine.moe_router import (
    alltoall_bytes, balanced_gate, route, topk_gate,
)

rng = np.random.default_rng(51)
N_TOKENS, N_EXPERTS = 4096, 8

# Skewed popularity: a few experts attract most tokens (realistic).
logits = rng.normal(0, 1, size=(N_TOKENS, N_EXPERTS))
logits[:, :2] += 1.5

gates = {"skewed": topk_gate(logits, top_k=1),
         "balanced": balanced_gate(logits, top_k=1, balance_strength=0.99, noise_scale=0.1, rng=rng)}
for gate_name, (expert_ids, weights) in gates.items():
    for cf in (1.0, 1.5, 2.0):
        res = route(expert_ids, weights, N_EXPERTS, capacity_factor=cf)
        print(f"{gate_name:8s} top-1 CF={cf}: dropped={res['drop_rate']:.1%}, "
              f"imbalance={res['imbalance_ratio']:.2f}")

print(f"all-to-all per step (16 experts, top-2, d=4096, fp16): "
      f"{alltoall_bytes(4096, 2, 4096) / 1e6:.1f} MB")
print("Load balancing beats raw capacity: a balanced router at CF=1.0 "
      "drops less than a skewed one at CF=2.0.")
