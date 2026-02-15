## Results (Draft)
### Main Outcomes
- Correctness pass rate is 1.0 for all methods in all suites.
- Mean restarts for failure variants are exactly 2.0, matching injected failures.
- Mean goodput delta vs reference:
  - Blocking: -13.25%
  - Overlapped: -11.50%
- Mean overlapped gain vs blocking: 1.99%
- Mean overlapped minus blocking stall time: -10.22s (negative = less stall).

### Per-Suite Ranking (Overlapped vs Blocking)
| dataset | model | failure | blk_gp | ovl_gp | ovl_vs_blk | ovl_minus_blk_stall_s | ovl_minus_blk_loss_diff |
|---|---|---|---:|---:|---:|---:|---:|
| cifar100 | resnet18 | late:800,1400 | 12.298 | 12.747 | +3.65% | -6.694 | -0.000316 |
| cifar100 | resnet50 | base:400,1200 | 7.628 | 7.864 | +3.09% | -13.937 | +0.000000 |
| cifar10 | resnet50 | late:800,1400 | 6.606 | 6.803 | +2.99% | -14.612 | +0.000000 |
| cifar10 | resnet18 | base:400,1200 | 11.302 | 11.572 | +2.39% | -4.943 | +0.001002 |
| cifar10 | resnet50 | base:400,1200 | 6.722 | 6.858 | +2.01% | -14.531 | +0.000000 |
| cifar100 | resnet18 | base:400,1200 | 12.579 | 12.744 | +1.31% | -5.979 | +0.003920 |
| cifar100 | resnet50 | late:800,1400 | 7.299 | 7.318 | +0.26% | -16.128 | +0.000000 |
| cifar10 | resnet18 | late:800,1400 | 11.214 | 11.241 | +0.24% | -4.940 | +0.002517 |
