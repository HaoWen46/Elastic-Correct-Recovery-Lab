# Publishable Report: exp4_matrix_20260215_151709_cifar100_resnet18_flate_k50_m4_publishable

Generated at: 2026-02-16T00:29:54
Seeds: 1

## Aggregate Performance

| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |
|---|---:|---:|---:|---:|
| Reference | 1.00 | 14.5657 +/- 0.0000 | 0.0000 | 0.00 +/- 0.00 |
| Failure + Blocking | 1.00 | 12.2979 +/- 0.0000 | 0.0000 | 2.00 +/- 0.00 |
| Failure + Overlapped | 1.00 | 12.7471 +/- 0.0000 | 0.0000 | 2.00 +/- 0.00 |

## Divergence (Across Seeds)

| Variant | Mean Abs Loss Diff (mean +/- std) |
|---|---:|
| Failure + Blocking | 0.097281 +/- 0.000000 |
| Failure + Overlapped | 0.096965 +/- 0.000000 |

### Parameter L2 by Step

| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |
|---:|---:|---:|
| 200 | 278.898296 +/- 0.000000 | 310.333397 +/- 0.000000 |
| 400 | 183.162057 +/- 0.000000 | 184.854129 +/- 0.000000 |
| 800 | 123.001826 +/- 0.000000 | 120.203170 +/- 0.000000 |
| 1200 | 108.878529 +/- 0.000000 | 107.726017 +/- 0.000000 |
| 1600 | 105.039287 +/- 0.000000 | 104.232217 +/- 0.000000 |

