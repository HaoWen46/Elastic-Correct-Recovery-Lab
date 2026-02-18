# Publishable Report: exp4_matrix_20260216_221349_cifar100_resnet18_flate_k50_m4_publishable

Generated at: 2026-02-18T04:47:57
Seeds: 3

## Aggregate Performance

| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |
|---|---:|---:|---:|---:|
| Reference | 1.00 | 16.3530 +/- 0.1942 | 0.2198 | 0.00 +/- 0.00 |
| Failure + Blocking | 1.00 | 13.6529 +/- 0.1341 | 0.1517 | 2.00 +/- 0.00 |
| Failure + Overlapped | 1.00 | 13.8934 +/- 0.1037 | 0.1174 | 2.00 +/- 0.00 |

## Divergence (Across Seeds)

| Variant | Mean Abs Loss Diff (mean +/- std) |
|---|---:|
| Failure + Blocking | 0.088250 +/- 0.003534 |
| Failure + Overlapped | 0.089588 +/- 0.002773 |

### Parameter L2 by Step

| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |
|---:|---:|---:|
| 200 | 303.056965 +/- 56.092399 | 288.479901 +/- 70.207329 |
| 400 | 178.004742 +/- 14.793825 | 166.885634 +/- 23.916120 |
| 800 | 122.497514 +/- 15.179240 | 113.952164 +/- 16.091627 |
| 1200 | 106.944381 +/- 13.572363 | 100.852951 +/- 16.165246 |
| 1600 | 103.661101 +/- 13.102218 | 97.844160 +/- 15.141419 |

