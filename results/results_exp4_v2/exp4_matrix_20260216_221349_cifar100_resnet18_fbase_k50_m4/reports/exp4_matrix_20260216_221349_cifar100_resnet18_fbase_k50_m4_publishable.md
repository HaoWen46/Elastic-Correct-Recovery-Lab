# Publishable Report: exp4_matrix_20260216_221349_cifar100_resnet18_fbase_k50_m4_publishable

Generated at: 2026-02-17T01:48:31
Seeds: 3

## Aggregate Performance

| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |
|---|---:|---:|---:|---:|
| Reference | 1.00 | 14.1061 +/- 0.3321 | 0.3759 | 0.00 +/- 0.00 |
| Failure + Blocking | 1.00 | 11.8239 +/- 0.0279 | 0.0316 | 2.00 +/- 0.00 |
| Failure + Overlapped | 1.00 | 12.1600 +/- 0.0935 | 0.1058 | 2.00 +/- 0.00 |

## Divergence (Across Seeds)

| Variant | Mean Abs Loss Diff (mean +/- std) |
|---|---:|
| Failure + Blocking | 0.095023 +/- 0.002004 |
| Failure + Overlapped | 0.096039 +/- 0.004218 |

### Parameter L2 by Step

| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |
|---:|---:|---:|
| 200 | 276.547686 +/- 60.827754 | 268.320465 +/- 45.206894 |
| 400 | 174.791602 +/- 35.114420 | 164.064269 +/- 12.124839 |
| 800 | 124.016612 +/- 21.524462 | 117.030932 +/- 3.206721 |
| 1200 | 109.874698 +/- 20.541794 | 102.659168 +/- 1.201724 |
| 1600 | 106.352826 +/- 18.984651 | 99.408650 +/- 1.047499 |

