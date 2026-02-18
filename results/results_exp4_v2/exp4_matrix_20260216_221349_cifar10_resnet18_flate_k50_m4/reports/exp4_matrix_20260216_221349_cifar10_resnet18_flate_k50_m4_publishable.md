# Publishable Report: exp4_matrix_20260216_221349_cifar10_resnet18_flate_k50_m4_publishable

Generated at: 2026-02-16T23:01:04
Seeds: 3

## Aggregate Performance

| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |
|---|---:|---:|---:|---:|
| Reference | 1.00 | 14.5432 +/- 0.7380 | 0.8352 | 0.00 +/- 0.00 |
| Failure + Blocking | 1.00 | 12.3472 +/- 0.7112 | 0.8048 | 2.00 +/- 0.00 |
| Failure + Overlapped | 1.00 | 12.6464 +/- 0.7680 | 0.8691 | 2.00 +/- 0.00 |

## Divergence (Across Seeds)

| Variant | Mean Abs Loss Diff (mean +/- std) |
|---|---:|
| Failure + Blocking | 0.121309 +/- 0.014165 |
| Failure + Overlapped | 0.103685 +/- 0.008923 |

### Parameter L2 by Step

| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |
|---:|---:|---:|
| 200 | 1758.840647 +/- 615.206414 | 1685.359713 +/- 694.414989 |
| 400 | 1027.080634 +/- 280.070492 | 959.664163 +/- 374.133145 |
| 800 | 489.220197 +/- 85.418467 | 414.295594 +/- 148.616855 |
| 1200 | 345.246812 +/- 53.441675 | 275.696167 +/- 94.434746 |
| 1600 | 317.864364 +/- 48.044071 | 247.338594 +/- 81.203290 |

