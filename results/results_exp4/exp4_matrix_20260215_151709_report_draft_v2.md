# Exp4 Matrix Report Draft (v2)

Generated: 2026-02-16T01:18:26

## Executive Summary

- Across 8 suites, overlapped checkpointing improves goodput over blocking by `1.99%` on average.
- Relative to reference runs, average goodput drops are `13.25%` (blocking) and `11.50%` (overlapped).
- Overlapped reduces checkpoint-induced stall time by `10.22s` on average per suite.
- Correctness remains perfect (`pass_rate=1.0`) for all variants in all suites; mean restarts are exactly `2.0` for failure variants.

## 1. Experiment Inventory

- Matrix prefix: `exp4_matrix_20260215_151709`
- Suites: 8 (`2 datasets x 2 models x 2 failure schedules`)
- Datasets: `cifar10`, `cifar100`
- Models: `resnet18`, `resnet50`
- Failure schedules: `base(400,1200)`, `late(800,1400)`
- Checkpoint settings: `every=50`, `max_inflight=4`
- Seeds per suite: `1`
- Manifest status counts: `{'skipped_exists': 4, 'completed': 4}`
- Run completion check: `24/24` run supervisors report `status=completed`.

## 2. Core Results

| Metric | Value |
|---|---:|
| Mean blocking vs reference goodput delta | -13.25% |
| Mean overlapped vs reference goodput delta | -11.50% |
| Mean overlapped vs blocking goodput delta | +1.99% |
| Mean overlapped minus blocking wall-clock | -3.56s |
| Mean overlapped minus blocking stall time | -10.22s |
| Mean overlapped minus blocking divergence (loss abs diff) | +0.000890 |

### 2.1 Per-Suite Ranking (Overlapped vs Blocking)

| dataset | model | failure | ref_gp | blk_gp | ovl_gp | ovl_vs_blk | ovl_minus_blk_stall_s | ovl_minus_blk_loss_diff |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cifar100 | resnet18 | late:800,1400 | 14.566 | 12.298 | 12.747 | +3.65% | -6.694 | -0.000316 |
| cifar100 | resnet50 | base:400,1200 | 8.644 | 7.628 | 7.864 | +3.09% | -13.937 | +0.000000 |
| cifar10 | resnet50 | late:800,1400 | 7.226 | 6.606 | 6.803 | +2.99% | -14.612 | +0.000000 |
| cifar10 | resnet18 | base:400,1200 | 11.729 | 11.302 | 11.572 | +2.39% | -4.943 | +0.001002 |
| cifar10 | resnet50 | base:400,1200 | 8.574 | 6.722 | 6.858 | +2.01% | -14.531 | +0.000000 |
| cifar100 | resnet18 | base:400,1200 | 14.977 | 12.579 | 12.744 | +1.31% | -5.979 | +0.003920 |
| cifar100 | resnet50 | late:800,1400 | 8.388 | 7.299 | 7.318 | +0.26% | -16.128 | +0.000000 |
| cifar10 | resnet18 | late:800,1400 | 13.323 | 11.214 | 11.241 | +0.24% | -4.940 | +0.002517 |

Best suite by overlapped gain: `exp4_matrix_20260215_151709_cifar100_resnet18_flate_k50_m4` (+3.65%).
Worst suite by overlapped gain: `exp4_matrix_20260215_151709_cifar10_resnet18_flate_k50_m4` (+0.24%).

### 2.2 Sliced Means

#### By Dataset

| group | n | mean ovl_vs_blk | mean ovl_minus_blk_stall_s | mean ovl_minus_blk_loss_diff |
|---|---:|---:|---:|---:|
| cifar10 | 4 | +1.91% | -9.756 | +0.000880 |
| cifar100 | 4 | +2.08% | -10.685 | +0.000901 |

#### By Model

| group | n | mean ovl_vs_blk | mean ovl_minus_blk_stall_s | mean ovl_minus_blk_loss_diff |
|---|---:|---:|---:|---:|
| resnet18 | 4 | +1.90% | -5.639 | +0.001781 |
| resnet50 | 4 | +2.09% | -14.802 | +0.000000 |

#### By Failure schedule

| group | n | mean ovl_vs_blk | mean ovl_minus_blk_stall_s | mean ovl_minus_blk_loss_diff |
|---|---:|---:|---:|---:|
| base | 4 | +2.20% | -9.848 | +0.001231 |
| late | 4 | +1.78% | -10.594 | +0.000550 |

## 3. Interpretation

- Overlapped checkpointing is consistently beneficial in this matrix (`8/8` suites) for throughput under injected failures.
- Most of the throughput benefit aligns with reduced checkpoint stall time rather than reduced write time alone.
- Divergence differences are tiny in absolute value and do not show evidence of correctness degradation.

## 4. Figure Index (Existing Artifacts)

| suite | goodput plot | loss plot |
|---|---|---|
| `exp4_matrix_20260215_151709_cifar100_resnet18_fbase_k50_m4` | `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_cifar100_resnet18_fbase_k50_m4/plots/goodput_exp4_matrix_20260215_151709_cifar100_resnet18_fbase_k50_m4_paperlite.png` | `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_cifar100_resnet18_fbase_k50_m4/plots/loss_curves_exp4_matrix_20260215_151709_cifar100_resnet18_fbase_k50_m4_paperlite.png` |
| `exp4_matrix_20260215_151709_cifar100_resnet18_flate_k50_m4` | `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_cifar100_resnet18_flate_k50_m4/plots/goodput_exp4_matrix_20260215_151709_cifar100_resnet18_flate_k50_m4_paperlite.png` | `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_cifar100_resnet18_flate_k50_m4/plots/loss_curves_exp4_matrix_20260215_151709_cifar100_resnet18_flate_k50_m4_paperlite.png` |
| `exp4_matrix_20260215_151709_cifar100_resnet50_fbase_k50_m4` | `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_cifar100_resnet50_fbase_k50_m4/plots/goodput_exp4_matrix_20260215_151709_cifar100_resnet50_fbase_k50_m4_paperlite.png` | `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_cifar100_resnet50_fbase_k50_m4/plots/loss_curves_exp4_matrix_20260215_151709_cifar100_resnet50_fbase_k50_m4_paperlite.png` |
| `exp4_matrix_20260215_151709_cifar100_resnet50_flate_k50_m4` | `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_cifar100_resnet50_flate_k50_m4/plots/goodput_exp4_matrix_20260215_151709_cifar100_resnet50_flate_k50_m4_paperlite.png` | `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_cifar100_resnet50_flate_k50_m4/plots/loss_curves_exp4_matrix_20260215_151709_cifar100_resnet50_flate_k50_m4_paperlite.png` |
| `exp4_matrix_20260215_151709_cifar10_resnet18_fbase_k50_m4` | `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_cifar10_resnet18_fbase_k50_m4/plots/goodput_exp4_matrix_20260215_151709_cifar10_resnet18_fbase_k50_m4_paperlite.png` | `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_cifar10_resnet18_fbase_k50_m4/plots/loss_curves_exp4_matrix_20260215_151709_cifar10_resnet18_fbase_k50_m4_paperlite.png` |
| `exp4_matrix_20260215_151709_cifar10_resnet18_flate_k50_m4` | `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_cifar10_resnet18_flate_k50_m4/plots/goodput_exp4_matrix_20260215_151709_cifar10_resnet18_flate_k50_m4_paperlite.png` | `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_cifar10_resnet18_flate_k50_m4/plots/loss_curves_exp4_matrix_20260215_151709_cifar10_resnet18_flate_k50_m4_paperlite.png` |
| `exp4_matrix_20260215_151709_cifar10_resnet50_fbase_k50_m4` | `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_cifar10_resnet50_fbase_k50_m4/plots/goodput_exp4_matrix_20260215_151709_cifar10_resnet50_fbase_k50_m4_paperlite.png` | `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_cifar10_resnet50_fbase_k50_m4/plots/loss_curves_exp4_matrix_20260215_151709_cifar10_resnet50_fbase_k50_m4_paperlite.png` |
| `exp4_matrix_20260215_151709_cifar10_resnet50_flate_k50_m4` | `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_cifar10_resnet50_flate_k50_m4/plots/goodput_exp4_matrix_20260215_151709_cifar10_resnet50_flate_k50_m4_paperlite.png` | `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_cifar10_resnet50_flate_k50_m4/plots/loss_curves_exp4_matrix_20260215_151709_cifar10_resnet50_flate_k50_m4_paperlite.png` |

## 5. Limitations and Next Runs

- Current evidence is single-seed (`n=1`) per suite; this is not enough for strong statistical claims.
- Minimal upgrade path under quota constraints: rerun the same matrix with `SEEDS_CSV=1337,2027` and resume enabled.
- If budget allows one extra axis, add `CHECKPOINT_EVERY_CSV=50,100` for CIFAR-100 suites first.

## 6. Source Artifacts

- `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_manifest.csv`
- `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_summary.csv`
- `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_detailed_table.csv`
- `/Users/haowenchen/Files/projects/ecrl/results/results_exp4/exp4_matrix_20260215_151709_analysis.md`
