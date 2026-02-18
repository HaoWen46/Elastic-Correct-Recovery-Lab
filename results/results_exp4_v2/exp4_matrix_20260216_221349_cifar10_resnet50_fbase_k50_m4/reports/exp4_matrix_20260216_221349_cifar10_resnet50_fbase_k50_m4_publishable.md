# Publishable Report: exp4_matrix_20260216_221349_cifar10_resnet50_fbase_k50_m4_publishable

Generated at: 2026-02-17T00:48:04
Seeds: 3

## Aggregate Performance

| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |
|---|---:|---:|---:|---:|
| Reference | 1.00 | 8.4189 +/- 0.0666 | 0.0753 | 0.00 +/- 0.00 |
| Failure + Blocking | 1.00 | 7.5410 +/- 0.0507 | 0.0574 | 2.00 +/- 0.00 |
| Failure + Overlapped | 1.00 | 7.8045 +/- 0.0552 | 0.0625 | 2.00 +/- 0.00 |

## Divergence (Across Seeds)

| Variant | Mean Abs Loss Diff (mean +/- std) |
|---|---:|
| Failure + Blocking | 0.044937 +/- 0.010055 |
| Failure + Overlapped | 0.044937 +/- 0.010055 |

### Parameter L2 by Step

| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |
|---:|---:|---:|
| 200 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 400 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 800 | 161869.044204 +/- 261279.247834 | 161869.044204 +/- 261279.247834 |
| 1200 | 122232.400645 +/- 199158.885474 | 122232.400645 +/- 199158.885474 |
| 1600 | 108241.238956 +/- 174342.641638 | 108241.238956 +/- 174342.641638 |

