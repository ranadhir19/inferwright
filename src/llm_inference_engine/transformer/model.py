"""LLaMA-style decoder-only transformer inference from scratch (numpy, CPU).

Implements the core mechanics of modern LLM inference with no ML framework:

- :class:`TransformerConfig` -- model hyperparameters
- :class:`RMSNorm` -- root-mean-square layer normalization (no bias, no mean shift)
- RoPE rotary embeddings via :func:`build_rope_cache` / :func:`apply_rope`
- Grouped-query attention (GQA): ``n_kv_heads < n_heads``, KV heads repeated
- SwiGLU feed-forward network: ``down(silu(gate(x)) * up(x))``
- :class:`KVCache` -- per-layer key/value buffers with append, for incremental decode
- :class:`TransformerModel` -- ``forward(tokens)`` full pass, ``step(token, cache)``
  single-token incremental decode, and ``generate()`` with greedy or
  temperature/top-k sampling

All weights are random-initialized (seeded). This module demonstrates inference
mechanics and their numerical/benchmark properties, not trained model quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TransformerConfig:
    """Hyperparameters for the nano transformer."""

    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2
    vocab_size: int = 512
    ffn_mult: float = 2.0
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    dtype: type = np.float32

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_hidden(self) -> int:
        return int(self.d_model * self.ffn_mult)

    @property
    def n_rep(self) -> int:
        """How many query heads share one KV head (GQA repeat factor)."""
        return self.n_heads // self.n_kv_heads


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm:
    """Root-mean-square normalization: ``w * x / sqrt(mean(x^2) + eps)``."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        self.weight = np.ones(dim, dtype=np.float32)
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
        return (self.weight * x / np.sqrt(ms + self.eps)).astype(np.float32)


def rmsnorm_naive(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Reference RMSNorm computed row-by-row in a Python loop (for verification)."""
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        row = x[i].astype(np.float64)
        out[i] = weight * x[i] / np.sqrt(np.mean(row**2) + eps)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def build_rope_cache(
    max_seq_len: int, head_dim: int, theta: float = 10000.0
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute RoPE cos/sin tables.

    Returns ``(cos, sin)`` each of shape ``(max_seq_len, head_dim // 2)`` using
    the LLaMA (GPT-NeoX style, halves) pairing convention.
    """
    freqs = 1.0 / (
        theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    )
    positions = np.arange(max_seq_len, dtype=np.float64)
    angles = np.outer(positions, freqs)  # (T, head_dim/2)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def apply_rope(
    x: np.ndarray, cos: np.ndarray, sin: np.ndarray
) -> np.ndarray:
    """Apply rotary position embedding to ``x``.

    ``x`` has shape ``(..., head_dim)``; ``cos``/``sin`` broadcast to
    ``(..., head_dim // 2)``. Pairing is halves: first half pairs with the
    second half, matching LLaMA.
    """
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1
    ).astype(x.dtype)


def rope_complex_reference(
    x: np.ndarray, angles: np.ndarray
) -> np.ndarray:
    """Reference RoPE via direct complex-exponential rotation (for verification).

    ``angles`` has shape broadcastable to ``(..., head_dim // 2)``.
    """
    half = x.shape[-1] // 2
    z = x[..., :half].astype(np.float64) + 1j * x[..., half:].astype(np.float64)
    rotated = z * np.exp(1j * angles.astype(np.float64))
    return np.concatenate([rotated.real, rotated.imag], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------


def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    """Repeat each KV head ``n_rep`` times along the head axis (GQA -> MHA shape)."""
    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=-2)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def sdpa(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, n_rep: int = 1
) -> np.ndarray:
    """Scaled dot-product attention with GQA KV-head repetition.

    ``q``: ``(Tq, n_heads, head_dim)``; ``k``/``v``: ``(Tk, n_kv_heads, head_dim)``.
    Returns ``(Tq, n_heads, head_dim)``. Causal masking is applied when
    ``Tq == Tk`` (full-sequence forward); for incremental decode (``Tq == 1``)
    the single query attends to all cached keys.
    """
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)
    d = q.shape[-1]
    # BLAS batched matmul: (H, Tq, D) @ (H, D, Tk) -> (H, Tq, Tk).
    qh = np.ascontiguousarray(q.transpose(1, 0, 2))
    kh = np.ascontiguousarray(k.transpose(1, 2, 0))
    scores = np.matmul(qh, kh) / np.sqrt(d)
    Tq, Tk = q.shape[0], k.shape[0]
    if Tq == Tk:
        # Causal mask: query i may only attend to keys j <= i.
        mask = np.triu(np.ones((Tq, Tk), dtype=bool), k=1)
        scores = np.where(mask[None, :, :], -np.inf, scores)
    weights = softmax(scores.astype(np.float64), axis=-1).astype(np.float32)
    vh = np.ascontiguousarray(v.transpose(1, 0, 2))
    out = np.matmul(weights, vh)  # (H, Tq, D)
    return np.ascontiguousarray(out.transpose(1, 0, 2))


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


class KVCache:
    """Per-layer key/value cache with append for incremental decoding.

    Buffers are preallocated to ``max_seq_len``; :meth:`append` writes new
    key/value rows at the current position and :meth:`advance` moves the
    position forward. :meth:`get` returns the filled prefix for a layer.
    """

    def __init__(self, config: TransformerConfig) -> None:
        self.config = config
        dt = config.dtype
        shape = (config.max_seq_len, config.n_kv_heads, config.head_dim)
        self.k = [np.zeros(shape, dtype=dt) for _ in range(config.n_layers)]
        self.v = [np.zeros(shape, dtype=dt) for _ in range(config.n_layers)]
        self.seq_len = 0

    def append(self, layer: int, k_new: np.ndarray, v_new: np.ndarray) -> None:
        """Write ``k_new``/``v_new`` of shape ``(T, n_kv_heads, head_dim)``."""
        t = k_new.shape[0]
        if self.seq_len + t > self.config.max_seq_len:
            raise ValueError("KV cache overflow: exceeds max_seq_len")
        self.k[layer][self.seq_len : self.seq_len + t] = k_new
        self.v[layer][self.seq_len : self.seq_len + t] = v_new

    def advance(self, n: int) -> None:
        self.seq_len += n

    def get(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Return cached ``(k, v)`` prefix of shape ``(seq_len, n_kv_heads, head_dim)``."""
        return self.k[layer][: self.seq_len], self.v[layer][: self.seq_len]

    def nbytes(self) -> int:
        """Bytes currently occupied by the filled cache portion."""
        cfg = self.config
        per_layer = 2 * cfg.n_kv_heads * cfg.head_dim * self.seq_len
        return cfg.n_layers * per_layer * np.dtype(cfg.dtype).itemsize

    @staticmethod
    def bytes_for(config: TransformerConfig, seq_len: int, bytes_per_elem: int = 2) -> int:
        """Analytic KV-cache footprint: 2 * L * Hkv * D * T * bytes_per_elem."""
        return (
            2
            * config.n_layers
            * config.n_kv_heads
            * config.head_dim
            * seq_len
            * bytes_per_elem
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class _LayerWeights:
    def __init__(self, config: TransformerConfig, rng: np.random.Generator) -> None:
        dt = config.dtype
        d, h, hkv, hd, fh = (
            config.d_model,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.ffn_hidden,
        )
        s = 0.02
        self.wq = (rng.standard_normal((d, h * hd)) * s).astype(dt)
        self.wk = (rng.standard_normal((d, hkv * hd)) * s).astype(dt)
        self.wv = (rng.standard_normal((d, hkv * hd)) * s).astype(dt)
        self.wo = (rng.standard_normal((h * hd, d)) * s).astype(dt)
        self.w_gate = (rng.standard_normal((d, fh)) * s).astype(dt)
        self.w_up = (rng.standard_normal((d, fh)) * s).astype(dt)
        self.w_down = (rng.standard_normal((fh, d)) * s).astype(dt)
        self.attn_norm = RMSNorm(d)
        self.ffn_norm = RMSNorm(d)


class TransformerModel:
    """LLaMA-style decoder-only transformer (numpy, CPU)."""

    def __init__(
        self, config: TransformerConfig, rng: np.random.Generator | None = None
    ) -> None:
        self.config = config
        rng = rng if rng is not None else np.random.default_rng()
        dt = config.dtype
        self.tok_emb = (rng.standard_normal((config.vocab_size, config.d_model)) * 0.02).astype(dt)
        self.layers = [_LayerWeights(config, rng) for _ in range(config.n_layers)]
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = (rng.standard_normal((config.d_model, config.vocab_size)) * 0.02).astype(dt)
        self.rope_cos, self.rope_sin = build_rope_cache(
            config.max_seq_len, config.head_dim, config.rope_theta
        )

    # -- internal block ----------------------------------------------------

    def _rope(self, x: np.ndarray, start: int) -> np.ndarray:
        t = x.shape[-3] if x.ndim == 3 else x.shape[0]
        # x: (T, H, D) with T at axis 0
        cos = self.rope_cos[start : start + t][:, None, :]
        sin = self.rope_sin[start : start + t][:, None, :]
        return apply_rope(x, cos, sin)

    def _block_full(self, x: np.ndarray, layer: _LayerWeights) -> np.ndarray:
        """One transformer block over a full sequence (positions 0..T-1)."""
        cfg = self.config
        h = layer.attn_norm(x)
        t = x.shape[0]
        q = (h @ layer.wq).reshape(t, cfg.n_heads, cfg.head_dim)
        k = (h @ layer.wk).reshape(t, cfg.n_kv_heads, cfg.head_dim)
        v = (h @ layer.wv).reshape(t, cfg.n_kv_heads, cfg.head_dim)
        q = self._rope(q, 0)
        k = self._rope(k, 0)
        attn = sdpa(q, k, v, n_rep=cfg.n_rep)
        x = x + (attn.reshape(t, cfg.d_model) @ layer.wo)
        h = layer.ffn_norm(x)
        ffn = silu(h @ layer.w_gate) * (h @ layer.w_up)
        x = x + (ffn @ layer.w_down)
        return x

    def _block_step(
        self, x: np.ndarray, layer: _LayerWeights, li: int, cache: KVCache, pos: int
    ) -> np.ndarray:
        """One transformer block for a single new token at position ``pos``.

        Appends the token's key/value to the cache, then attends over the full
        cached prefix (positions 0..pos).
        """
        cfg = self.config
        h = layer.attn_norm(x)
        q = (h @ layer.wq).reshape(1, cfg.n_heads, cfg.head_dim)
        k_new = (h @ layer.wk).reshape(1, cfg.n_kv_heads, cfg.head_dim)
        v_new = (h @ layer.wv).reshape(1, cfg.n_kv_heads, cfg.head_dim)
        q = self._rope(q, pos)
        k_new = self._rope(k_new, pos)
        cache.append(li, k_new, v_new)
        k_all = cache.k[li][: pos + 1]
        v_all = cache.v[li][: pos + 1]
        attn = sdpa(q, k_all, v_all, n_rep=cfg.n_rep)
        x = x + (attn.reshape(1, cfg.d_model) @ layer.wo)
        h = layer.ffn_norm(x)
        ffn = silu(h @ layer.w_gate) * (h @ layer.w_up)
        x = x + (ffn @ layer.w_down)
        return x

    # -- public API --------------------------------------------------------

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """Full forward pass. ``tokens``: ``(T,)`` int array. Returns ``(T, vocab)`` logits."""
        tokens = np.asarray(tokens, dtype=np.int64)
        x = self.tok_emb[tokens]
        for layer in self.layers:
            x = self._block_full(x, layer)
        return self.final_norm(x) @ self.lm_head

    def step(self, token: int, cache: KVCache) -> np.ndarray:
        """Single-token incremental decode. Returns ``(vocab,)`` logits."""
        x = self.tok_emb[np.array([int(token)], dtype=np.int64)]
        pos = cache.seq_len
        for li, layer in enumerate(self.layers):
            x = self._block_step(x, layer, li, cache, pos)
        cache.advance(1)
        return (self.final_norm(x) @ self.lm_head)[0]

    def prefill(self, tokens: np.ndarray, cache: KVCache) -> np.ndarray:
        """Fill ``cache`` with a prompt via incremental steps. Returns last logits."""
        logits = None
        for tok in np.asarray(tokens, dtype=np.int64):
            logits = self.step(int(tok), cache)
        return logits

    def generate(
        self,
        prompt_tokens: np.ndarray | list[int],
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 0,
        seed: int | None = None,
    ) -> list[int]:
        """Generate token ids. ``temperature=0`` (default) is greedy decoding."""
        rng = np.random.default_rng(seed)
        cache = KVCache(self.config)
        out = [int(t) for t in prompt_tokens]
        logits = self.prefill(np.asarray(out, dtype=np.int64), cache)
        for _ in range(max_new_tokens):
            if temperature <= 0.0:
                nxt = int(np.argmax(logits))
            else:
                l = logits.astype(np.float64) / temperature
                if top_k > 0:
                    cutoff = np.sort(l)[-top_k]
                    l = np.where(l >= cutoff, l, -np.inf)
                l = l - np.max(l)
                probs = np.exp(l)
                probs = probs / probs.sum()
                nxt = int(rng.choice(self.config.vocab_size, p=probs))
            out.append(nxt)
            logits = self.step(nxt, cache)
        return out
