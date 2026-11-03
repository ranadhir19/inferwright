"""Smoke tests for 01-nano-transformer (fast: small config, CPU numpy).

Run from the repo root with:
    python3 -m pytest 01-nano-transformer/tests/test_smoke.py -q
or simply:
    python3 01-nano-transformer/tests/test_smoke.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_inference_engine.transformer import (
    KVCache,
    TransformerConfig,
    TransformerModel,
    apply_rope,
    build_rope_cache,
    generate,
)


def tiny_config() -> TransformerConfig:
    return TransformerConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=64,
        ffn_mult=2.0,
        max_seq_len=64,
    )


def test_forward_shapes():
    cfg = tiny_config()
    model = TransformerModel(cfg, np.random.default_rng(0))
    toks = np.array([1, 2, 3, 4, 5])
    logits = model.forward(toks)
    assert logits.shape == (5, cfg.vocab_size), logits.shape
    assert np.all(np.isfinite(logits))


def test_step_shape_and_cache_growth():
    cfg = tiny_config()
    model = TransformerModel(cfg, np.random.default_rng(0))
    cache = KVCache(cfg)
    for i, tok in enumerate([7, 8, 9]):
        logits = model.step(tok, cache)
        assert logits.shape == (cfg.vocab_size,), logits.shape
        assert cache.seq_len == i + 1
    k, v = cache.get(0)
    assert k.shape == (3, cfg.n_kv_heads, cfg.head_dim)
    assert v.shape == (3, cfg.n_kv_heads, cfg.head_dim)


def test_cache_matches_full_forward():
    cfg = tiny_config()
    model = TransformerModel(cfg, np.random.default_rng(1))
    rng = np.random.default_rng(2)
    toks = rng.integers(0, cfg.vocab_size, size=16)
    full = model.forward(toks)
    cache = KVCache(cfg)
    inc = np.stack([model.step(int(t), cache) for t in toks])
    diff = float(np.max(np.abs(full - inc)))
    assert diff < 1e-5, f"cache/full mismatch: {diff}"


def test_determinism_same_seed():
    cfg = tiny_config()
    m1 = TransformerModel(cfg, np.random.default_rng(42))
    m2 = TransformerModel(cfg, np.random.default_rng(42))
    toks = np.array([3, 1, 4, 1, 5])
    assert np.array_equal(m1.forward(toks), m2.forward(toks))
    g1 = generate(m1, toks, max_new_tokens=8, temperature=0.0)
    g2 = generate(m2, toks, max_new_tokens=8, temperature=0.0)
    assert g1 == g2


def test_generate_greedy_and_sampling():
    cfg = tiny_config()
    model = TransformerModel(cfg, np.random.default_rng(3))
    prompt = [1, 2, 3]
    greedy = generate(model, prompt, max_new_tokens=5, temperature=0.0)
    assert greedy[:3] == prompt and len(greedy) == 8
    assert all(0 <= t < cfg.vocab_size for t in greedy)
    sampled = generate(model, prompt, max_new_tokens=5, temperature=1.0, top_k=8, seed=0)
    assert len(sampled) == 8
    # same seed -> same sampled output
    sampled2 = generate(model, prompt, max_new_tokens=5, temperature=1.0, top_k=8, seed=0)
    assert sampled == sampled2


def test_rope_changes_with_position():
    cos, sin = build_rope_cache(16, 8)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 2, 8)).astype(np.float32)
    r0 = apply_rope(x, cos[0:1][:, None, :], sin[0:1][:, None, :])
    r5 = apply_rope(x, cos[5:6][:, None, :], sin[5:6][:, None, :])
    assert r0.shape == x.shape
    assert not np.allclose(r0, r5), "RoPE should depend on position"


def test_kv_cache_analytic_bytes():
    cfg = tiny_config()
    # 2 * n_layers * n_kv_heads * head_dim * seq_len * 2 bytes (fp16)
    expected = 2 * 2 * 2 * 8 * 16 * 2
    assert KVCache.bytes_for(cfg, 16, bytes_per_elem=2) == expected


if __name__ == "__main__":
    test_forward_shapes()
    test_step_shape_and_cache_growth()
    test_cache_matches_full_forward()
    test_determinism_same_seed()
    test_generate_greedy_and_sampling()
    test_rope_changes_with_position()
    test_kv_cache_analytic_bytes()
    print("all smoke tests passed")
