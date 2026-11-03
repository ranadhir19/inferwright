# 06-moe-router — results

Seeded `np.random.default_rng(51)`; numpy CPU only. Two consecutive runs produce identical headlines.

## Experiment 1 — routing quality vs capacity factor

n_tokens=4096, n_experts=8, skewed expert popularity [0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05].

### top-1, skewed gating

| CF | expert capacity | drop rate % | imbalance max/mean | processed tokens |
|----|---------------|-------------|--------------------|------------------|
| 0.50 | 256 | 77.51 | 2.22 | 921 |
| 0.75 | 384 | 68.14 | 2.35 | 1305 |
| 1.00 | 512 | 60.99 | 2.56 | 1598 |
| 1.25 | 640 | 54.74 | 2.76 | 1854 |
| 1.50 | 768 | 48.49 | 2.91 | 2110 |
| 2.00 | 1024 | 38.09 | 3.23 | 2536 |

### top-2, skewed gating

| CF | expert capacity | drop rate % | imbalance max/mean | processed tokens |
|----|---------------|-------------|--------------------|------------------|
| 0.50 | 256 | 83.18 | 1.49 | 1044 |
| 0.75 | 384 | 76.93 | 1.63 | 1402 |
| 1.00 | 512 | 71.75 | 1.77 | 1665 |
| 1.25 | 640 | 67.07 | 1.90 | 1882 |
| 1.50 | 768 | 62.38 | 1.99 | 2102 |
| 2.00 | 1024 | 53.00 | 2.13 | 2556 |

## Experiment 2 — load-balancing loss effect at CF=1.0

Balanced gating = skew shrunk 99% (idealized aux load-balancing loss) + tiny tie-break noise.

| top-k | gating | drop rate % | imbalance max/mean | processed tokens |
|-------|--------|-------------|--------------------|------------------|
| 1 | skewed | 60.96 | 2.56 | 1599 |
| 1 | balanced | 9.33 | 1.10 | 3714 |
| 2 | skewed | 71.51 | 1.75 | 1693 |
| 2 | balanced | 50.00 | 1.00 | 2251 |

## Experiment 3 — expert parallelism

E=16 experts, n_tokens=4096, top_k=2, d_model=4096, fp16.

| devices D | experts/device | all-to-all total | comm/device | token slots/device |
|-----------|----------------|------------------|-------------|--------------------|
| 1 | 16 | 128.0 MiB | 128.0 MiB | 8192 |
| 2 | 8 | 128.0 MiB | 64.0 MiB | 4096 |
| 4 | 4 | 128.0 MiB | 32.0 MiB | 2048 |
| 8 | 2 | 128.0 MiB | 16.0 MiB | 1024 |

Total all-to-all volume is independent of D (dispatch + combine of every token->expert vector); per-device compute shrinks as 1/D, so the comm/compute ratio worsens with more devices.

## Headline

top-1 skewed CF=1.0: drop 61.0%, imbalance 2.56; balanced: drop 9.3%; all-to-all at D=8: 16.0 MiB/device (128.0 MiB total)
