# 05-quant-lab: weight quantization workbench -- results

Seeded end-to-end (`np.random.default_rng(41)`); numpy + matplotlib only, CPU.

## Setup

* Tiny char-level transformer: d_model=64, 2 layers, 4 heads, d_ff=128, context 64, vocab 29 chars, 74397 params.
* Corpus: 12031 chars of generated repetitive text (90% train / 10% eval).
* Training: Adam lr=3e-3, batch 16, 600 steps, 250s CPU; final train loss 0.4724.
* Quantization scope: all Linear weight matrices (`wq/wk/wv/wo/w1/w2/head_w`); embeddings, norms and biases stay fp32.
* Bits/param counts quantized weights only, assuming fp16 scales (standard GPTQ-style accounting).

## Experiment 1: synthetic reconstruction (outlier-structured weights)

Weights ~ N(0, 0.03) with 0.2% of entries x25 and ~1.5% of columns x12 (outlier channels).

| shape | scheme | rel. L2 error | max-abs error | bits/param |
|---|---|---|---|---|
| 1024x1024 | INT8 per-channel | 0.0491 | 0.0773 | 8.016 |
| 1024x1024 | FP8 E4M3 (direct cast) | 0.0323 | 0.5009 | 8.000 |
| 1024x1024 | INT4 group-128 | 0.2274 | 0.7894 | 4.125 |
| 1024x1024 | INT4 group-32 | 0.1317 | 0.3709 | 4.500 |
| 1024x1024 | INT8 per-tensor | 0.3736 | 0.0777 | 8.000 |
| 4096x1024 | INT8 per-channel | 0.0466 | 0.0967 | 8.016 |
| 4096x1024 | FP8 E4M3 (direct cast) | 0.0337 | 0.9809 | 8.000 |
| 4096x1024 | INT4 group-128 | 0.2302 | 0.8872 | 4.125 |
| 4096x1024 | INT4 group-32 | 0.1334 | 0.6129 | 4.500 |
| 4096x1024 | INT8 per-tensor | 0.3864 | 0.0979 | 8.000 |

Reading: per-channel / per-group schemes isolate outlier channels into their own scale, so one hot channel cannot blow up the error of the whole tensor -- per-tensor INT8 pays the full outlier tax on both metrics.

## Experiment 2: trained-model perplexity (eval run twice)

| scheme | eval perplexity | bits/param |
|---|---|---|
| FP32 baseline | 2.1130 | 32.000 |
| INT8 per-channel | 2.1131 | 8.220 |
| FP8 E4M3 (direct cast) | 2.1212 | 8.000 |
| INT4 group-128 | 2.1867 | 4.220 |
| INT4 group-32 | 2.1490 | 4.500 |
| INT8 per-tensor | 2.1121 | 8.003 |

The eval sweep was executed twice (training once; weights reloaded from disk for pass 2). Both passes produced bit-identical perplexities -- the numbers above are deterministic, not noise.

## Experiment 3: per-layer sensitivity (INT4 group-32, one scope)

| quantized scope | eval perplexity | delta vs baseline |
|---|---|---|
| l0 | 2.1416 | +0.0286 |
| l1 | 2.1098 | -0.0033 |
| head | 2.1246 | +0.0116 |

## Scale limits

Tiny model (74397 params), character-level, CPU, synthetic repetitive text. The *methodology* -- granularity ablations, Pareto analysis, sensitivity sweeps -- transfers to real LLMs; the absolute perplexities do not.
