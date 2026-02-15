# Publishable Report: exp4_matrix_20260215_151709_cifar10_resnet50_fbase_k50_m4_publishable

Generated at: 2026-02-15T15:47:12
Seeds: 1

## Aggregate Performance

| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |
|---|---:|---:|---:|---:|
| Reference | 1.00 | 8.5743 +/- 0.0000 | 0.0000 | 0.00 +/- 0.00 |
| Failure + Blocking | 1.00 | 6.7222 +/- 0.0000 | 0.0000 | 2.00 +/- 0.00 |
| Failure + Overlapped | 1.00 | 6.8576 +/- 0.0000 | 0.0000 | 2.00 +/- 0.00 |

## Divergence (Across Seeds)

| Variant | Mean Abs Loss Diff (mean +/- std) |
|---|---:|
| Failure + Blocking | 0.037772 +/- 0.000000 |
| Failure + Overlapped | 0.037772 +/- 0.000000 |

### Parameter L2 by Step

| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |
|---:|---:|---:|
| 200 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 400 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 800 | 463567.993342 +/- 0.000000 | 463567.993342 +/- 0.000000 |
| 1200 | 352190.498324 +/- 0.000000 | 352190.498324 +/- 0.000000 |
| 1600 | 309525.978356 +/- 0.000000 | 309525.978356 +/- 0.000000 |

