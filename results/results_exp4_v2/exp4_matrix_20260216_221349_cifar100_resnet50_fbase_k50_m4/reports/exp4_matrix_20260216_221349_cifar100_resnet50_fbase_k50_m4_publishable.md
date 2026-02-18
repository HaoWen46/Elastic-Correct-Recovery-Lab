# Publishable Report: exp4_matrix_20260216_221349_cifar100_resnet50_fbase_k50_m4_publishable

Generated at: 2026-02-18T05:20:41
Seeds: 3

## Aggregate Performance

| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |
|---|---:|---:|---:|---:|
| Reference | 1.00 | 9.3980 +/- 0.0849 | 0.0961 | 0.00 +/- 0.00 |
| Failure + Blocking | 1.00 | 8.3523 +/- 0.0780 | 0.0883 | 2.00 +/- 0.00 |
| Failure + Overlapped | 1.00 | 8.6963 +/- 0.0975 | 0.1103 | 2.00 +/- 0.00 |

## Divergence (Across Seeds)

| Variant | Mean Abs Loss Diff (mean +/- std) |
|---|---:|
| Failure + Blocking | 0.044097 +/- 0.006535 |
| Failure + Overlapped | 0.044097 +/- 0.006535 |

### Parameter L2 by Step

| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |
|---:|---:|---:|
| 200 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 400 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| 800 | 2725.631070 +/- 1794.575423 | 2725.631070 +/- 1794.575423 |
| 1200 | 4734.604936 +/- 6706.022593 | 4734.604936 +/- 6706.022593 |
| 1600 | 5767.222622 +/- 7655.443808 | 5767.222622 +/- 7655.443808 |

