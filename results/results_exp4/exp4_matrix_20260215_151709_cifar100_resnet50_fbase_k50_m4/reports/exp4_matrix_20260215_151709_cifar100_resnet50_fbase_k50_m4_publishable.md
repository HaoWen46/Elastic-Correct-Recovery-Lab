# Publishable Report: exp4_matrix_20260215_151709_cifar100_resnet50_fbase_k50_m4_publishable

Generated at: 2026-02-16T00:41:55
Seeds: 1

## Aggregate Performance

| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |
|---|---:|---:|---:|---:|
| Reference | 1.00 | 8.6444 +/- 0.0000 | 0.0000 | 0.00 +/- 0.00 |
| Failure + Blocking | 1.00 | 7.6277 +/- 0.0000 | 0.0000 | 2.00 +/- 0.00 |
| Failure + Overlapped | 1.00 | 7.8637 +/- 0.0000 | 0.0000 | 2.00 +/- 0.00 |

## Divergence (Across Seeds)

| Variant | Mean Abs Loss Diff (mean +/- std) |
|---|---:|
| Failure + Blocking | 0.037167 +/- 0.000000 |
| Failure + Overlapped | 0.037167 +/- 0.000000 |

### Parameter L2 by Step

| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |
|---:|---:|---:|
| 200 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 400 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 800 | 1132.729619 +/- 0.000000 | 1132.729619 +/- 0.000000 |
| 1200 | 907.419525 +/- 0.000000 | 907.419525 +/- 0.000000 |
| 1600 | 1334.049099 +/- 0.000000 | 1334.049099 +/- 0.000000 |

