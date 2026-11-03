"""quant_lab: weight-quantization workbench (INT8 / FP8 / INT4) in numpy.

Trains a tiny character-level transformer from scratch on CPU, then measures
real quality degradation (eval perplexity) when its Linear weights are
quantized with different schemes and granularities.

Public API
----------
* :mod:`quant_lab.schemes` — quantization schemes. Each scheme is a function
  ``(w) -> (dequantized_w, metadata)``: ``quantize_int8_per_channel``,
  ``quantize_int8_per_tensor``, ``quantize_int4_per_group``,
  ``quantize_fp8_e4m3``, plus ``effective_bits_per_param`` and
  ``fp8_e4m3_grid_values``.
* :mod:`quant_lab.metrics` — reconstruction metrics: ``relative_l2_error``,
  ``snr_db``, ``max_abs_error``, ``mean_abs_error``,
  ``reconstruction_report``.
* :mod:`quant_lab.tiny_lm` — tiny char-level transformer (``TinyLMConfig``,
  ``init_params``, ``forward``/``backward``, ``Adam``, ``build_corpus``,
  ``encode_corpus``, ``train``, ``evaluate_perplexity``).
* :mod:`quant_lab.eval` — quantized-model evaluation: ``quantize_model``,
  ``quantized_bits_per_param``, ``model_bits_per_param``, ``evaluate``.

Experiments live in ``05-quant-lab/experiments/run_experiments.py``.
"""

from . import eval, metrics, schemes, tiny_lm

__all__ = ["eval", "metrics", "schemes", "tiny_lm"]
