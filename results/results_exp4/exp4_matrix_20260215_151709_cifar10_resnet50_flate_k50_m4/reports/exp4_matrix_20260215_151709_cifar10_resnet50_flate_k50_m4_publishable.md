# Publishable Report: exp4_matrix_20260215_151709_cifar10_resnet50_flate_k50_m4_publishable

Generated at: 2026-02-15T16:00:51
Seeds: 1

## Aggregate Performance

| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |
|---|---:|---:|---:|---:|
| Reference | 1.00 | 7.2259 +/- 0.0000 | 0.0000 | 0.00 +/- 0.00 |
| Failure + Blocking | 1.00 | 6.6058 +/- 0.0000 | 0.0000 | 2.00 +/- 0.00 |
| Failure + Overlapped | 1.00 | 6.8030 +/- 0.0000 | 0.0000 | 2.00 +/- 0.00 |

## Divergence (Across Seeds)

| Variant | Mean Abs Loss Diff (mean +/- std) |
|---|---:|
| Failure + Blocking | 0.024610 +/- 0.000000 |
| Failure + Overlapped | 0.024610 +/- 0.000000 |

### Parameter L2 by Step

| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |
|---:|---:|---:|
| 200 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 400 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 800 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 1200 | 63812.196604 +/- 0.000000 | 63812.196604 +/- 0.000000 |
| 1600 | 54133.490772 +/- 0.000000 | 54133.490772 +/- 0.000000 |

