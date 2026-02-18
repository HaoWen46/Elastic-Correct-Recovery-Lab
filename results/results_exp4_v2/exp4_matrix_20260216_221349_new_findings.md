# New Findings: exp4_matrix_20260216_221349

- Source summary: `results/results_exp4/exp4_matrix_20260216_221349_summary.csv`
- Suites: 8
- Status counts: {'completed_prev': 5, 'completed': 3}
- Seeds per suite: [3]

## Overall

- Mean reference goodput: 11.875087
- Mean blocking goodput: 10.220443 (-13.2700% vs ref)
- Mean overlapped goodput: 10.550279 (-10.3195% vs ref)
- Mean overlapped vs blocking gain: 3.3827%
- Gain range (ovl vs blk): [1.7612%, 4.4050%]
- Overlapped better suites: 8/8
- Mean loss-diff (blk): 0.070331
- Mean loss-diff (ovl): 0.067936

## Dataset Slice

| dataset | n | ref gp | blk gp | ovl gp | ovl vs blk (%) | blk drop vs ref (%) | ovl drop vs ref (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | 4 | 11.426186 | 9.876670 | 10.222504 | 3.5917 | -12.9077 | -9.7719 |
| cifar100 | 4 | 12.323989 | 10.564216 | 10.878053 | 3.1737 | -13.6322 | -10.8671 |

## Model Slice

| model | n | ref gp | blk gp | ovl gp | ovl vs blk (%) | blk drop vs ref (%) | ovl drop vs ref (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet18 | 4 | 14.833133 | 12.476603 | 12.817711 | 2.7679 | -15.8690 | -13.5388 |
| resnet50 | 4 | 8.917042 | 7.964283 | 8.282846 | 3.9976 | -10.6710 | -7.1002 |

## Failure Slice

| failure_label | n | ref gp | blk gp | ovl gp | ovl vs blk (%) | blk drop vs ref (%) | ovl drop vs ref (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 4 | 11.563303 | 9.949936 | 10.307997 | 3.6249 | -13.3545 | -10.2088 |
| late | 4 | 12.186872 | 10.490950 | 10.792560 | 3.1405 | -13.1854 | -10.4302 |

## Suite Ranking (ovl vs blk gain)

| rank | run_prefix | ovl vs blk (%) | blk drop vs ref (%) | ovl drop vs ref (%) |
|---:|---|---:|---:|---:|
| 1 | exp4_matrix_20260216_221349_cifar10_resnet50_flate_k50_m4 | 4.4050 | -10.4179 | -6.4718 |
| 2 | exp4_matrix_20260216_221349_cifar100_resnet50_fbase_k50_m4 | 4.1186 | -11.1265 | -7.4662 |
| 3 | exp4_matrix_20260216_221349_cifar10_resnet18_fbase_k50_m4 | 4.0441 | -15.6849 | -12.2751 |
| 4 | exp4_matrix_20260216_221349_cifar100_resnet50_flate_k50_m4 | 3.9724 | -10.7118 | -7.1649 |
| 5 | exp4_matrix_20260216_221349_cifar10_resnet50_fbase_k50_m4 | 3.4943 | -10.4277 | -7.2977 |
| 6 | exp4_matrix_20260216_221349_cifar100_resnet18_fbase_k50_m4 | 2.8426 | -16.1790 | -13.7963 |
| 7 | exp4_matrix_20260216_221349_cifar10_resnet18_flate_k50_m4 | 2.4235 | -15.1004 | -13.0428 |
| 8 | exp4_matrix_20260216_221349_cifar100_resnet18_flate_k50_m4 | 1.7612 | -16.5115 | -15.0411 |
