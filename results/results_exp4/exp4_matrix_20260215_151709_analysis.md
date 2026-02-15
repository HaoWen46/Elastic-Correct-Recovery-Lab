# Exp4 Matrix Analysis (Pre-Report)

- Matrix prefix: `exp4_matrix_20260215_151709`
- Summary source: `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_summary.csv`
- Manifest source: `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_manifest.csv`
- Suites: 8
- Manifest statuses: {'skipped_exists': 4, 'completed': 4}

## Integrity Check

- `num_seeds` values: ['1']
- All correctness pass rates are 1.0: True
- Restart means (blocking, overlapped): [('2.0', '2.0')]

## Topline Quantitative Findings

- Average goodput drop vs reference: blocking `-13.25%`, overlapped `-11.50%`.
- Average overlapped gain vs blocking: `1.99%`.
- Divergence (mean abs loss diff) is close between blocking and overlapped across all suites; largest gap is small (<0.004).

## Ranked By Overlapped Gain vs Blocking

| dataset | model | failure | ref | blk | ovl | ovl_vs_blk | blk_loss | ovl_loss |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cifar100 | resnet18 | late:800,1400 | 14.566 | 12.298 | 12.747 | +3.65% | 0.097281 | 0.096965 |
| cifar100 | resnet50 | base:400,1200 | 8.644 | 7.628 | 7.864 | +3.09% | 0.037167 | 0.037167 |
| cifar10 | resnet50 | late:800,1400 | 7.226 | 6.606 | 6.803 | +2.99% | 0.024610 | 0.024610 |
| cifar10 | resnet18 | base:400,1200 | 11.729 | 11.302 | 11.572 | +2.39% | 0.110506 | 0.111508 |
| cifar10 | resnet50 | base:400,1200 | 8.574 | 6.722 | 6.858 | +2.01% | 0.037772 | 0.037772 |
| cifar100 | resnet18 | base:400,1200 | 14.977 | 12.579 | 12.744 | +1.31% | 0.090599 | 0.094519 |
| cifar100 | resnet50 | late:800,1400 | 8.388 | 7.299 | 7.318 | +0.26% | 0.026191 | 0.026191 |
| cifar10 | resnet18 | late:800,1400 | 13.323 | 11.214 | 11.241 | +0.24% | 0.103883 | 0.106400 |

## Breakdown

### By Model (mean overlapped gain vs blocking)

- `resnet18`: +1.90% (n=4)
- `resnet50`: +2.09% (n=4)

### By Dataset (mean overlapped gain vs blocking)

- `cifar10`: +1.91% (n=4)
- `cifar100`: +2.08% (n=4)

## Report Draft Plan

1. Experimental Setup: hardware, model/dataset matrix, fail schedules, checkpoint setup.
2. Recovery Correctness: pass-rate table, restart behavior.
3. Performance Under Failure: reference vs blocking vs overlapped goodput; percentage deltas.
4. Stability/Divergence: loss-diff summary and step-wise L2 from publishable JSONs.
5. Practical Takeaways: when overlapped helps most (by model/dataset/failure timing).

## Gaps Before "Publishable" Claims

- Current matrix is `num_seeds=1` for each suite, so confidence intervals are not meaningful.
- Recommended minimum: rerun with `PROFILE=balanced` (2 seeds) or custom `SEEDS_CSV=1337,2027,4242` for 3 seeds.
- If quota-limited, prioritize the 4 CIFAR-100 suites first (currently strongest signal).
