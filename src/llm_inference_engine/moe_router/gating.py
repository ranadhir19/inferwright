"""Token -> expert gating for mixture-of-experts (MoE) layers.

Implements the standard softmax top-k router used by models such as
GShard / Switch Transformer, plus two simulation-oriented variants:

- :func:`noisy_gate` models imperfect gating (e.g. low-precision or
  approximated router scores) by perturbing the logits before top-k.
- :func:`balanced_gate` models the *effect* of an auxiliary
  load-balancing loss (as in Switch Transformer) as reduced logit skew:
  the load-balancing loss pushes router logits toward uniform, which we
  approximate by shrinking each logit toward the per-token mean.

Conventions
-----------
``logits`` has shape ``(n_tokens, n_experts)``. All gating functions
return ``(expert_ids, weights)`` where ``expert_ids`` is an integer
array of shape ``(n_tokens, top_k)`` (ranked, best first) and ``weights``
has the same shape with per-token weights that sum to 1.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def topk_gate(logits: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Softmax top-k gating with weight renormalization.

    Parameters
    ----------
    logits:
        Token -> expert scores, shape ``(n_tokens, n_experts)``.
    top_k:
        Number of experts each token is routed to (1 or 2 typical).

    Returns
    -------
    expert_ids:
        Integer array, shape ``(n_tokens, top_k)``, best expert first.
    weights:
        Float array, same shape; renormalized softmax probabilities of
        the selected experts (each row sums to 1).
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape {logits.shape}")
    n_tokens, n_experts = logits.shape
    if not 1 <= top_k <= n_experts:
        raise ValueError(f"top_k must be in [1, n_experts], got {top_k}")

    probs = softmax(logits)
    # argpartition is O(n) per token; then sort only the selected k.
    part = np.argpartition(probs, -top_k, axis=-1)[:, -top_k:]
    rows = np.arange(n_tokens)[:, None]
    order = np.argsort(-probs[rows, part], axis=-1)
    expert_ids = np.take_along_axis(part, order, axis=-1)
    w = probs[rows, expert_ids]
    weights = w / np.sum(w, axis=-1, keepdims=True)
    return expert_ids, weights


def noisy_gate(
    logits: np.ndarray,
    top_k: int,
    noise_scale: float = 1.0,
    rng: np.random.Generator | None = None,
    noise: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k gating with noise added to the logits before selection.

    Simulates imperfect routing (quantized/approximate router scores,
    training noise). Weights are renormalized softmax probabilities of
    the *noisy* logits for the selected experts, matching noisy top-k
    gating (Shazeer et al., 2017).

    Parameters
    ----------
    noise_scale:
        Standard deviation (gaussian) or scale (gumbel) of the noise.
    noise:
        ``"gaussian"`` or ``"gumbel"``.
    rng:
        Seeded generator for reproducibility; created fresh if None.
    """
    if rng is None:
        rng = np.random.default_rng()
    logits = np.asarray(logits, dtype=np.float64)
    if noise == "gumbel":
        perturb = rng.gumbel(0.0, noise_scale, size=logits.shape)
    elif noise == "gaussian":
        perturb = rng.normal(0.0, noise_scale, size=logits.shape)
    else:
        raise ValueError(f"unknown noise type: {noise!r}")
    return topk_gate(logits + perturb, top_k)


def balanced_gate(
    logits: np.ndarray,
    top_k: int,
    balance_strength: float = 0.9,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k gating with the skew reduced, as an aux load-balancing loss would.

    A load-balancing auxiliary loss (Switch Transformer) encourages the
    router to use every expert equally. We model its *steady-state effect*
    by shrinking each token's logits toward their per-token mean::

        adjusted = mean + (1 - balance_strength) * (logits - mean)

    ``balance_strength=0`` recovers :func:`topk_gate`;
    ``balance_strength=1`` makes logits uniform (perfectly balanced).

    A small ``noise_scale`` can be added on top to break exact ties and
    produce near-uniform (rather than degenerate) routing.
    """
    if not 0.0 <= balance_strength <= 1.0:
        raise ValueError(f"balance_strength must be in [0, 1], got {balance_strength}")
    logits = np.asarray(logits, dtype=np.float64)
    mean = logits.mean(axis=-1, keepdims=True)
    shrunk = mean + (1.0 - balance_strength) * (logits - mean)
    if noise_scale > 0.0:
        return noisy_gate(shrunk, top_k, noise_scale=noise_scale, rng=rng)
    return topk_gate(shrunk, top_k)
