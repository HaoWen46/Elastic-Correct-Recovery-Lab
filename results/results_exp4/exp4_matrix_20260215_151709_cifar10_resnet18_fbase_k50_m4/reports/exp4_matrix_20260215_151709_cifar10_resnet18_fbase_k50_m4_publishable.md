# Publishable Report: exp4_matrix_20260215_151709_cifar10_resnet18_fbase_k50_m4_publishable

Generated at: 2026-02-15T15:25:47
Seeds: 1

## Aggregate Performance

| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |
|---|---:|---:|---:|---:|
| Reference | 1.00 | 11.7289 +/- 0.0000 | 0.0000 | 1.00 +/- 0.00 |
| Failure + Blocking | 1.00 | 11.3023 +/- 0.0000 | 0.0000 | 2.00 +/- 0.00 |
| Failure + Overlapped | 1.00 | 11.5721 +/- 0.0000 | 0.0000 | 2.00 +/- 0.00 |

## Divergence (Across Seeds)

| Variant | Mean Abs Loss Diff (mean +/- std) |
|---|---:|
| Failure + Blocking | 0.110506 +/- 0.000000 |
| Failure + Overlapped | 0.111508 +/- 0.000000 |

### Parameter L2 by Step

| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |
|---:|---:|---:|
| 200 | 1762.028606 +/- 0.000000 | 2769.167203 +/- 0.000000 |
| 400 | 901.227306 +/- 0.000000 | 1729.389898 +/- 0.000000 |
| 800 | 387.100771 +/- 0.000000 | 891.305276 +/- 0.000000 |
| 1200 | 266.687405 +/- 0.000000 | 699.941650 +/- 0.000000 |
| 1600 | 240.547377 +/- 0.000000 | 689.262147 +/- 0.000000 |

