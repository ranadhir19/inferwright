"""Quickstart: disaggregated prefill/decode vs colocated serving.

Compares time-to-first-token under both architectures on a long-prompt
workload, and reports the KV-cache transfer cost.
"""
import numpy as np

from llm_inference_engine.disagg_serve import (
    kv_cache_bytes, make_workload, run_colocated, run_disaggregated,
)

rng = np.random.default_rng(31)
prompt_len = 2048
print(f"KV cache for a {prompt_len}-token prompt: "
      f"{kv_cache_bytes(prompt_len) / 1e9:.2f} GB (fp16, 7B-ish model)")


def summarize(name, res):
    ttft = np.asarray(res.ttft_ms)
    print(f"{name:14s} mean TTFT={ttft.mean():6.1f} ms  "
          f"p99 TTFT={np.percentile(ttft, 99):6.1f} ms  "
          f"throughput={res.throughput_tok_s:.0f} tok/s")


reqs = make_workload(rng, 8.0, 200, short_frac=0.2)  # long-prompt heavy
summarize("colocated", run_colocated(reqs))
summarize("disaggregated", run_disaggregated(reqs))
print("Disaggregation wins when prefill stops blocking decodes; "
      "the KV transfer is the tax.")
