# Publishable Report: exp4_matrix_20260216_221349_cifar10_resnet50_flate_k50_m4_publishable

Generated at: 2026-02-17T01:23:58
Seeds: 3

## Aggregate Performance

| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |
|---|---:|---:|---:|---:|
| Reference | 1.00 | 8.4124 +/- 0.0165 | 0.0187 | 0.00 +/- 0.00 |
| Failure + Blocking | 1.00 | 7.5360 +/- 0.0212 | 0.0240 | 2.00 +/- 0.00 |
| Failure + Overlapped | 1.00 | 7.8680 +/- 0.0266 | 0.0301 | 2.00 +/- 0.00 |

## Divergence (Across Seeds)

| Variant | Mean Abs Loss Diff (mean +/- std) |
|---|---:|
| Failure + Blocking | 0.026413 +/- 0.001564 |
| Failure + Overlapped | 0.026413 +/- 0.001564 |

### Parameter L2 by Step

| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |
|---:|---:|---:|
| 200 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 400 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 800 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 1200 | 24726.589668 +/- 33871.526832 | 24726.589668 +/- 33871.526832 |
| 1600 | 21369.419189 +/- 28426.856573 | 21369.419189 +/- 28426.856573 |

