# Publishable Report: exp4_matrix_20260216_221349_cifar100_resnet50_flate_k50_m4_publishable

Generated at: 2026-02-18T05:53:10
Seeds: 3

## Aggregate Performance

| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |
|---|---:|---:|---:|---:|
| Reference | 1.00 | 9.4388 +/- 0.0858 | 0.0971 | 0.00 +/- 0.00 |
| Failure + Blocking | 1.00 | 8.4278 +/- 0.0544 | 0.0615 | 2.00 +/- 0.00 |
| Failure + Overlapped | 1.00 | 8.7625 +/- 0.0472 | 0.0534 | 2.00 +/- 0.00 |

## Divergence (Across Seeds)

| Variant | Mean Abs Loss Diff (mean +/- std) |
|---|---:|
| Failure + Blocking | 0.032901 +/- 0.005943 |
| Failure + Overlapped | 0.032901 +/- 0.005943 |

### Parameter L2 by Step

| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |
|---:|---:|---:|
| 200 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 400 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 800 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 1200 | 11184.513418 +/- 18031.166693 | 11184.513418 +/- 18031.166693 |
| 1600 | 10651.660559 +/- 16145.296023 | 10651.660559 +/- 16145.296023 |

