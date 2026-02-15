# Exp4 Matrix Report Draft (v1)

Generated: 2026-02-16T01:17:20

## 1. Scope and Setup

- Matrix prefix: `exp4_matrix_20260215_151709`
- Suites: 8 total (`2 datasets x 2 models x 2 failure schedules`)
- Datasets: `cifar10`, `cifar100`
- Models: `resnet18`, `resnet50`
- Failure schedules: `base=400,1200`, `late=800,1400`
- Checkpoint setup: `every=50`, `max_inflight=4`
- Distributed setup: `nproc=4`
- Seeds per suite: `1` (`PROFILE=small`)
- Manifest statuses: `{'skipped_exists': 4, 'completed': 4}`

## 2. Integrity and Recovery Correctness

- Correctness pass rate is `1.0` for reference, blocking, and overlapped in all 8 suites.
- Mean restarts are consistent at `2.0` for both failure variants, matching two injected failures.
- Replayed steps are `0` in all suites (from aggregate metrics).

## 3. Main Quantitative Results

- Mean goodput delta vs reference: blocking `-13.25%`, overlapped `-11.50%`.
- Mean overlapped gain vs blocking: `1.99%`.
- Mean wall-clock delta (overlapped - blocking): `-3.56s` (negative is better).
- Mean stall-time delta (overlapped - blocking): `-10.22s`.
- Mean divergence delta (overlapped - blocking, loss abs diff): `0.000890`.

### 3.1 Per-Suite Ranking (Overlapped vs Blocking Goodput)

| dataset | model | failure | blk_goodput | ovl_goodput | ovl_vs_blk | blk_stall_s | ovl_stall_s | ovl_minus_blk_loss_diff |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cifar100 | resnet18 | late:800,1400 | 12.298 | 12.747 | +3.65% | 11.796 | 5.102 | -0.000316 |
| cifar100 | resnet50 | base:400,1200 | 7.628 | 7.864 | +3.09% | 24.210 | 10.273 | +0.000000 |
| cifar10 | resnet50 | late:800,1400 | 6.606 | 6.803 | +2.99% | 24.719 | 10.108 | +0.000000 |
| cifar10 | resnet18 | base:400,1200 | 11.302 | 11.572 | +2.39% | 11.070 | 6.127 | +0.001002 |
| cifar10 | resnet50 | base:400,1200 | 6.722 | 6.858 | +2.01% | 24.283 | 9.752 | +0.000000 |
| cifar100 | resnet18 | base:400,1200 | 12.579 | 12.744 | +1.31% | 11.175 | 5.196 | +0.003920 |
| cifar100 | resnet50 | late:800,1400 | 7.299 | 7.318 | +0.26% | 26.613 | 10.484 | +0.000000 |
| cifar10 | resnet18 | late:800,1400 | 11.214 | 11.241 | +0.24% | 10.822 | 5.882 | +0.002517 |

### 3.2 Breakdown by Dataset

| dataset | n | mean ovl_vs_blk | mean ovl_minus_blk_stall_s | mean ovl_minus_blk_loss_diff |
|---|---:|---:|---:|---:|
| cifar10 | 4 | +1.91% | -9.756 | +0.000880 |
| cifar100 | 4 | +2.08% | -10.685 | +0.000901 |

### 3.3 Breakdown by Model

| model | n | mean ovl_vs_blk | mean ovl_minus_blk_stall_s | mean ovl_minus_blk_loss_diff |
|---|---:|---:|---:|---:|
| resnet18 | 4 | +1.90% | -5.639 | +0.001781 |
| resnet50 | 4 | +2.09% | -14.802 | +0.000000 |

### 3.4 Breakdown by Failure Schedule

| failure_label | n | mean ovl_vs_blk | mean ovl_minus_blk_stall_s | mean ovl_minus_blk_loss_diff |
|---|---:|---:|---:|---:|
| base | 4 | +2.20% | -9.848 | +0.001231 |
| late | 4 | +1.78% | -10.594 | +0.000550 |

## 4. Interpretation

- Overlapped checkpointing consistently improves goodput vs blocking in this matrix (8/8 suites), with gains between `0.24%` and `3.65%`.
- Goodput gains track lower checkpoint-induced stall time; overlapped reduces stall time by about 10 seconds on average per suite.
- Divergence differences are very small in absolute terms, and correctness remains perfect in all suites.

## 5. Threats to Validity

- `num_seeds=1` per suite means no robust confidence intervals; this is directional evidence only.
- Single checkpoint interval (`k=50`) and single inflight setting (`m=4`) in this run limit checkpoint-policy conclusions.
- GPU server background load and run interruptions (SIGHUP events in nohup log) may add variance to wall-clock metrics.

## 6. Minimal Next Runs for Stronger Claims

1. Rerun same matrix with `SEEDS_CSV=1337,2027` (or 3 seeds if quota allows).
2. Add one checkpoint sensitivity axis (`CHECKPOINT_EVERY_CSV=50,100`) for the 4 CIFAR-100 suites first.
3. Keep `MATRIX_PREFIX` fixed per campaign and use resume mode to avoid wasting GPU quota.

## 7. Artifact Index

- Manifest: `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_manifest.csv`
- Matrix summary: `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_summary.csv`
- Detailed table (generated): `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_detailed_table.csv`
- Pre-report analysis: `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_analysis.md`
