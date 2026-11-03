"""LLaMA-style decoder-only transformer inference from scratch (numpy, CPU).

Public API:

- :class:`TransformerConfig` -- model hyperparameters
- :class:`TransformerModel` -- ``forward`` / ``step`` / ``prefill`` / ``generate``
- :class:`KVCache` -- per-layer key/value cache for incremental decoding
- :class:`RMSNorm` -- root-mean-square normalization
- :func:`apply_rope` -- rotary position embedding application
- :func:`generate` -- module-level generation helper
"""

from .model import (
    KVCache,
    RMSNorm,
    TransformerConfig,
    TransformerModel,
    apply_rope,
    build_rope_cache,
    rmsnorm_naive,
    rope_complex_reference,
)


def generate(
    model: TransformerModel,
    prompt_tokens,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int | None = None,
):
    """Generate token ids from ``prompt_tokens`` using ``model``.

    ``temperature=0`` (default) selects greedy decoding; otherwise samples with
    temperature scaling and optional top-k filtering.
    """
    return model.generate(
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )


__all__ = [
    "TransformerConfig",
    "TransformerModel",
    "KVCache",
    "RMSNorm",
    "apply_rope",
    "build_rope_cache",
    "rmsnorm_naive",
    "rope_complex_reference",
    "generate",
]
