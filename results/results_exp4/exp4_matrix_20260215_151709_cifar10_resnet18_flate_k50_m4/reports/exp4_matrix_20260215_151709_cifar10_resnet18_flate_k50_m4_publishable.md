# Publishable Report: exp4_matrix_20260215_151709_cifar10_resnet18_flate_k50_m4_publishable

Generated at: 2026-02-15T15:34:17
Seeds: 1

## Aggregate Performance

| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |
|---|---:|---:|---:|---:|
| Reference | 1.00 | 13.3230 +/- 0.0000 | 0.0000 | 0.00 +/- 0.00 |
| Failure + Blocking | 1.00 | 11.2143 +/- 0.0000 | 0.0000 | 2.00 +/- 0.00 |
| Failure + Overlapped | 1.00 | 11.2408 +/- 0.0000 | 0.0000 | 2.00 +/- 0.00 |

## Divergence (Across Seeds)

| Variant | Mean Abs Loss Diff (mean +/- std) |
|---|---:|
| Failure + Blocking | 0.103883 +/- 0.000000 |
| Failure + Overlapped | 0.106400 +/- 0.000000 |

### Parameter L2 by Step

| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |
|---:|---:|---:|
| 200 | 2784.577236 +/- 0.000000 | 3239.996427 +/- 0.000000 |
| 400 | 1876.853055 +/- 0.000000 | 2101.934931 +/- 0.000000 |
| 800 | 996.181639 +/- 0.000000 | 1063.554416 +/- 0.000000 |
| 1200 | 801.174867 +/- 0.000000 | 831.109846 +/- 0.000000 |
| 1600 | 785.859397 +/- 0.000000 | 807.069277 +/- 0.000000 |

