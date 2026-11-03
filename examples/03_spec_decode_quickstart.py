"""Quickstart: speculative decoding speedup analysis.

A small draft model proposes k tokens; the big target model verifies them in
one forward pass. Speedup depends on the acceptance rate alpha and the
draft/target cost ratio c.
"""
from llm_inference_engine.spec_decode import expected_speedup, optimal_k, tokens_per_second

T_TARGET_MS = 40.0   # one target forward pass
C = 0.1              # draft token costs 10% of a target token

for alpha in (0.9, 0.7, 0.5, 0.3):
    k, _ = optimal_k(alpha, C)
    s = expected_speedup(k, alpha, C)
    tps, _ = tokens_per_second(T_TARGET_MS, C, k, alpha)
    print(f"alpha={alpha:.1f}: optimal k={k}, speedup={s:.2f}x, "
          f"{tps:.1f} tok/s vs {1000 / T_TARGET_MS:.1f} tok/s vanilla")

print("Low acceptance + long drafts can SLOW you down: speculation is a bet on alpha.")
