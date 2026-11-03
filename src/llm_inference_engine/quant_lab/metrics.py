"""Reconstruction-quality metrics for quantized tensors (numpy, CPU)."""

import numpy as np


def relative_l2_error(w, wq):
    """||w - wq||_2 / ||w||_2. Scale-invariant; the headline error metric."""
    w = np.asarray(w, dtype=np.float64)
    wq = np.asarray(wq, dtype=np.float64)
    denom = np.linalg.norm(w)
    if denom == 0.0:
        return 0.0 if np.linalg.norm(wq) == 0.0 else float("inf")
    return float(np.linalg.norm(w - wq) / denom)


def snr_db(w, wq):
    """Signal-to-noise ratio of the quantization, in dB."""
    w = np.asarray(w, dtype=np.float64)
    wq = np.asarray(wq, dtype=np.float64)
    signal = float(np.sum(w ** 2))
    noise = float(np.sum((w - wq) ** 2))
    if noise == 0.0:
        return float("inf")
    if signal == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(signal / noise))


def max_abs_error(w, wq):
    """max |w - wq|. Sensitive to outliers a single bad channel can cause."""
    return float(np.max(np.abs(np.asarray(w, dtype=np.float64) -
                               np.asarray(wq, dtype=np.float64))))


def mean_abs_error(w, wq):
    """mean |w - wq|."""
    return float(np.mean(np.abs(np.asarray(w, dtype=np.float64) -
                                np.asarray(wq, dtype=np.float64))))


def reconstruction_report(w, wq):
    """All metrics in one dict."""
    return {
        "relative_l2": relative_l2_error(w, wq),
        "snr_db": snr_db(w, wq),
        "max_abs": max_abs_error(w, wq),
        "mean_abs": mean_abs_error(w, wq),
    }
