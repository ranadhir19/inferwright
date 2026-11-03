"""Quickstart: LLaMA-style decoder inference from scratch (numpy).

Builds a tiny transformer, verifies KV-cache incremental decoding matches a
full forward pass bit-exactly, and shows the latency win of the KV cache.
"""
import time

import numpy as np

from llm_inference_engine.transformer import (
    KVCache, TransformerConfig, TransformerModel, generate,
)

rng = np.random.default_rng(7)
cfg = TransformerConfig(d_model=128, n_layers=4, n_heads=8, n_kv_heads=2,
                        vocab_size=512, ffn_mult=2, max_seq_len=1024)
model = TransformerModel(cfg, rng=rng)

prompt = rng.integers(0, cfg.vocab_size, size=64)

# Full forward pass vs incremental KV-cache decode: must match exactly.
logits_full = model.forward(prompt)
cache = KVCache(cfg)
model.prefill(prompt[:-1], cache)
logits_step = model.step(int(prompt[-1]), cache)
diff = float(np.max(np.abs(logits_full[-1] - logits_step)))
print(f"KV-cache vs full forward max diff: {diff:.2e} (expect ~0)")

# Latency: decode one token at position 512 with and without the cache.
ctx = rng.integers(0, cfg.vocab_size, size=512)
t0 = time.perf_counter()
model.forward(ctx)
t_full = (time.perf_counter() - t0) * 1000
cache = KVCache(cfg)
model.prefill(ctx, cache)
t0 = time.perf_counter()
model.step(int(ctx[-1]), cache)
t_cached = (time.perf_counter() - t0) * 1000
print(f"decode @seq512: {t_cached:.2f} ms/token cached vs "
      f"{t_full:.2f} ms/token recompute ({t_full / t_cached:.0f}x)")

# Generate a few tokens (random weights: demonstrates mechanics, not quality).
out = generate(model, list(prompt[:8]), max_new_tokens=8, seed=7)
print("generated token ids:", out[8:])
print(f"KV cache size @512 tokens: {cache.bytes_for(cfg, 512) / 1024:.0f} KiB (fp16)")
