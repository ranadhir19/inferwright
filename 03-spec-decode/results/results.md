# 03-spec-decode — results

Seeded, reproducible (numpy, CPU only). Monte Carlo: 100,000 trials/cell over 120 cells (k=1..8 x alpha x c), seed 21.

## 1. Formula vs Monte Carlo

Max absolute deviation between the analytical expected accepted tokens and the Monte Carlo mean: **0.01786** tokens (worst cell: k=7, alpha=0.85, c=0.1, MC mean=4.83220, formula=4.85006). Speedup max deviation: 0.01051. The geometric acceptance model is validated.

## 2. Speedup sweep (expected speedup vs vanilla AR)

Rows: draft length k. Columns: acceptance probability alpha.


### c = 0.05

| k | alpha=0.3 | alpha=0.5 | alpha=0.7 | alpha=0.85 | alpha=0.95 |
|---|---|---|---|---|---|
| 1 | 1.238 | 1.429 | 1.619 | 1.762 | 1.857 |
| 2 | 1.264 | 1.591 | 1.991 | 2.339 | 2.593 |
| 3 | 1.232 | 1.630 | 2.203 | 2.771 | 3.226 |
| 4 | 1.188 | 1.615 | 2.311 | 3.091 | 3.770 |
| 5 | 1.142 | 1.575 | 2.353 | 3.322 | 4.239 |
| 6 | 1.099 | 1.526 | 2.353 | 3.484 | 4.641 |
| 7 | 1.058 | 1.476 | 2.327 | 3.593 | 4.986 |
| 8 | 1.020 | 1.426 | 2.285 | 3.659 | 5.282 |

### c = 0.1

| k | alpha=0.3 | alpha=0.5 | alpha=0.7 | alpha=0.85 | alpha=0.95 |
|---|---|---|---|---|---|
| 1 | 1.182 | 1.364 | 1.545 | 1.682 | 1.773 |
| 2 | 1.158 | 1.458 | 1.825 | 2.144 | 2.377 |
| 3 | 1.090 | 1.442 | 1.948 | 2.451 | 2.854 |
| 4 | 1.018 | 1.384 | 1.981 | 2.649 | 3.232 |
| 5 | 0.952 | 1.312 | 1.961 | 2.768 | 3.532 |
| 6 | 0.893 | 1.240 | 1.912 | 2.831 | 3.771 |
| 7 | 0.840 | 1.172 | 1.848 | 2.853 | 3.960 |
| 8 | 0.794 | 1.109 | 1.777 | 2.846 | 4.108 |

### c = 0.2

| k | alpha=0.3 | alpha=0.5 | alpha=0.7 | alpha=0.85 | alpha=0.95 |
|---|---|---|---|---|---|
| 1 | 1.083 | 1.250 | 1.417 | 1.542 | 1.625 |
| 2 | 0.993 | 1.250 | 1.564 | 1.838 | 2.037 |
| 3 | 0.886 | 1.172 | 1.583 | 1.992 | 2.319 |
| 4 | 0.792 | 1.076 | 1.541 | 2.060 | 2.514 |
| 5 | 0.714 | 0.984 | 1.471 | 2.076 | 2.649 |
| 6 | 0.649 | 0.902 | 1.390 | 2.059 | 2.742 |
| 7 | 0.595 | 0.830 | 1.309 | 2.021 | 2.805 |
| 8 | 0.549 | 0.768 | 1.230 | 1.970 | 2.844 |

### Optimal k per (alpha, c)

| alpha | c=0.05 | c=0.1 | c=0.2 |
|---|---|---|---|
| 0.3 | k=2 (1.26x) | k=1 (1.18x) | k=1 (1.08x) |
| 0.5 | k=3 (1.63x) | k=2 (1.46x) | k=1 (1.25x) |
| 0.7 | k=6 (2.35x) | k=4 (1.98x) | k=3 (1.58x) |
| 0.85 | k=8 (3.66x) | k=7 (2.85x) | k=5 (2.08x) |
| 0.95 | k=8 (5.28x) | k=8 (4.11x) | k=8 (2.84x) |

Best cell in sweep: **5.282x speedup at k=8, alpha=0.95, c=0.05**.

## 3. End-to-end wall-clock scenario (target forward 40 ms, c = 0.1)

- alpha=0.7 at optimal k=4: speculative **49.5 tok/s** vs vanilla **25.0 tok/s** (speedup 1.979x, analytical 1.981x).

- Failure case alpha=0.3, k=8: speculative **19.8 tok/s** vs vanilla **25.0 tok/s** (speedup 0.791x, analytical 0.794x) — a slowdown: draft compute is wasted because most drafts are rejected.

## 4. Charts

`results/charts.png`: (a) speedup heatmap vs (k, alpha) at c=0.1; (b) optimal-k vs alpha curves for the three c values.
