# Publishable Report: exp4_matrix_20260215_151709_cifar100_resnet50_flate_k50_m4_publishable

Generated at: 2026-02-16T00:54:25
Seeds: 1

## Aggregate Performance

| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |
|---|---:|---:|---:|---:|
| Reference | 1.00 | 8.3879 +/- 0.0000 | 0.0000 | 0.00 +/- 0.00 |
| Failure + Blocking | 1.00 | 7.2988 +/- 0.0000 | 0.0000 | 2.00 +/- 0.00 |
| Failure + Overlapped | 1.00 | 7.3179 +/- 0.0000 | 0.0000 | 2.00 +/- 0.00 |

## Divergence (Across Seeds)

| Variant | Mean Abs Loss Diff (mean +/- std) |
|---|---:|
| Failure + Blocking | 0.026191 +/- 0.000000 |
| Failure + Overlapped | 0.026191 +/- 0.000000 |

### Parameter L2 by Step

| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |
|---:|---:|---:|
| 200 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 400 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 800 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 1200 | 723.682436 +/- 0.000000 | 723.682436 +/- 0.000000 |
| 1600 | 1360.538301 +/- 0.000000 | 1360.538301 +/- 0.000000 |

