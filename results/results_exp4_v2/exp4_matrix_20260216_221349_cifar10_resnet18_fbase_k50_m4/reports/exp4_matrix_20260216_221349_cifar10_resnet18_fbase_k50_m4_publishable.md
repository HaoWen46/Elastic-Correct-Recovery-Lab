# Publishable Report: exp4_matrix_20260216_221349_cifar10_resnet18_fbase_k50_m4_publishable

Generated at: 2026-02-16T22:37:38
Seeds: 3

## Aggregate Performance

| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |
|---|---:|---:|---:|---:|
| Reference | 1.00 | 14.3302 +/- 0.5228 | 0.5916 | 0.00 +/- 0.00 |
| Failure + Blocking | 1.00 | 12.0825 +/- 1.3404 | 1.5168 | 2.00 +/- 0.00 |
| Failure + Overlapped | 1.00 | 12.5711 +/- 0.9858 | 1.1155 | 2.00 +/- 0.00 |

## Divergence (Across Seeds)

| Variant | Mean Abs Loss Diff (mean +/- std) |
|---|---:|
| Failure + Blocking | 0.109720 +/- 0.006283 |
| Failure + Overlapped | 0.105832 +/- 0.004737 |

### Parameter L2 by Step

| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |
|---:|---:|---:|
| 200 | 1747.136213 +/- 238.252451 | 1710.480747 +/- 316.808408 |
| 400 | 994.513035 +/- 166.561895 | 987.122614 +/- 119.400252 |
| 800 | 438.075815 +/- 82.943480 | 454.192200 +/- 79.702002 |
| 1200 | 281.754617 +/- 62.102010 | 319.074267 +/- 63.478066 |
| 1600 | 252.165769 +/- 54.914749 | 290.619391 +/- 56.604483 |

