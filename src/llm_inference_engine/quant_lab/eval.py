"""Evaluation of quantized models: perplexity under weight quantization.

Quantization scope
------------------
Only the Linear *weight matrices* (``wq/wk/wv/wo/w1/w2/head_w``) are
quantized. Token/positional embeddings, layer-norm parameters and biases stay
fp32. Rationale, mirroring industry practice (e.g. LLM.int8()):

* embeddings are lookup tables, not matmul operands — quantizing them saves
  little compute and disproportionately hurts rare tokens;
* norms/biases are <1% of parameters; keeping them fp32 costs nothing and
  avoids destabilizing the residual stream.
"""

import numpy as np

from . import tiny_lm
from .schemes import effective_bits_per_param
from .tiny_lm import is_linear_weight


def quantize_model(params, quant_fn, scope="all", **kwargs):
    """Quantize a subset of a model's Linear weights.

    ``quant_fn``: ``(w) -> (wq, meta)`` from :mod:`quant_lab.schemes`.
    ``scope``: ``"all"`` (every Linear weight), ``"l0"``/``"l1"``/...
    (one transformer block), or ``"head"`` (the output head).

    Returns ``(new_params, metas)`` where ``metas`` maps quantized tensor
    names to their metadata dicts.
    """
    new_params = {kk: vv.copy() for kk, vv in params.items()}
    metas = {}
    for name, w in params.items():
        if not is_linear_weight(name):
            continue
        if scope == "all":
            do_it = True
        elif scope == "head":
            do_it = name.startswith("head")
        else:  # e.g. "l0", "l1"
            do_it = name.startswith(scope + ".")
        if do_it:
            wq, meta = quant_fn(w, **kwargs)
            new_params[name] = wq
            metas[name] = meta
    return new_params, metas


def quantized_bits_per_param(params, metas):
    """Mean effective bits/param over the *quantized* weight tensors."""
    tot_bits, tot_n = 0.0, 0
    for name, meta in metas.items():
        n = int(params[name].size)
        tot_bits += effective_bits_per_param(meta, n) * n
        tot_n += n
    return tot_bits / tot_n if tot_n else 32.0


def model_bits_per_param(params, metas):
    """Mean effective bits/param over the *whole* model (incl. fp32 parts)."""
    tot_bits, tot_n = 0.0, 0
    for name, w in params.items():
        n = int(w.size)
        if name in metas:
            tot_bits += effective_bits_per_param(metas[name], n) * n
        else:
            tot_bits += 32.0 * n
        tot_n += n
    return tot_bits / tot_n


def evaluate(params, cfg, eval_ids):
    """Return ``{"perplexity": ..., "loss": ...}`` on the eval split."""
    ppl = tiny_lm.evaluate_perplexity(params, cfg, eval_ids)
    return {"perplexity": float(ppl), "loss": float(np.log(ppl))}
