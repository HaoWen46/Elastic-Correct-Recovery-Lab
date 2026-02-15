# Publishable Report: exp4_matrix_20260215_151709_cifar100_resnet18_fbase_k50_m4_publishable

Generated at: 2026-02-16T00:22:04
Seeds: 1

## Aggregate Performance

| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |
|---|---:|---:|---:|---:|
| Reference | 1.00 | 14.9770 +/- 0.0000 | 0.0000 | 0.00 +/- 0.00 |
| Failure + Blocking | 1.00 | 12.5790 +/- 0.0000 | 0.0000 | 2.00 +/- 0.00 |
| Failure + Overlapped | 1.00 | 12.7443 +/- 0.0000 | 0.0000 | 2.00 +/- 0.00 |

## Divergence (Across Seeds)

| Variant | Mean Abs Loss Diff (mean +/- std) |
|---|---:|
| Failure + Blocking | 0.090599 +/- 0.000000 |
| Failure + Overlapped | 0.094519 +/- 0.000000 |

### Parameter L2 by Step

| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |
|---:|---:|---:|
| 200 | 290.659330 +/- 0.000000 | 275.151575 +/- 0.000000 |
| 400 | 197.606963 +/- 0.000000 | 170.596368 +/- 0.000000 |
| 800 | 132.608454 +/- 0.000000 | 115.918968 +/- 0.000000 |
| 1200 | 116.642827 +/- 0.000000 | 102.926683 +/- 0.000000 |
| 1600 | 112.124153 +/- 0.000000 | 99.523554 +/- 0.000000 |

