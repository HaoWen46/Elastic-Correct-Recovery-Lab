# ECRL: Elastic & Correct Recovery Lab (DDP)

ECRL is a reproducible evaluation framework for distributed training recovery focused on:

1. Correctness of restart semantics under failures and elastic world-size changes.
2. Goodput under failures (useful progress accounting for replay).
3. Elastic resume (`N -> M`) with constant global batch and preserved data-progress correctness.

This repository is an evaluation + semantics project for DDP Data Parallel training.

## Scope and Non-Negotiable Invariants

- Python `3.11.2` only.
- Virtual environment required for all Python work.
- DDP Data Parallel only (no FSDP, no GPU/OS checkpoint-restart).
- `GLOBAL_BATCH` is constant across all runs and resumes.
- Runtime assertion: `GLOBAL_BATCH % world_size == 0`.
- `LOCAL_BATCH = GLOBAL_BATCH // world_size`.
- DataLoader uses `num_workers=0`, `drop_last=True`.
- Epoch geometry is fixed by definition:
  - `D = len(dataset)`
  - `steps_per_epoch = floor(D / GLOBAL_BATCH)`
  - `epoch_samples = steps_per_epoch * GLOBAL_BATCH`
- Epoch progression is driven by explicit `for step_in_epoch in range(steps_per_epoch)` loops.
- Rank-0-only checkpoint writes.
- Atomic checkpoints: temp file -> `fsync/close` -> `rename` -> pointer update (`latest.json`).
- Checkpoint barriers at capture boundary.
- Failure injection only at step boundaries, initiated by rank 0.
- Overlapped checkpointing uses a background **process** for filesystem I/O only (no distributed APIs).

## Setup

```bash
uv venv --python 3.11.2 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

This project uses an `uv`-managed virtual environment (`.venv`) for all Python work.
Experiment scripts auto-download/extract CIFAR10/CIFAR100 into `./data` if missing.

## Repository Layout

```text
ecrl/
  README.md
  requirements.txt
  configs/
    exp1_failure.yaml
    exp2_elastic.yaml
    exp3_publishable.yaml
    exp3_publishable_1gpu.yaml
    exp3_publishable_4gpu.yaml
  ecrl/
    data/cifar_with_ids.py
    data/datasets_with_ids.py
    sampler/resumable_sampler.py
    statepack/statepack.py
    statepack/rng.py
    ckpt/atomic_writer.py
    ckpt/blocking.py
    ckpt/overlapped.py
    train/ddp_train.py
    orchestration/supervisor.py
    metrics/correctness.py
    metrics/goodput.py
    metrics/divergence.py
    metrics/plot.py
    metrics/aggregate.py
    metrics/report_publishable.py
  scripts/
    run_exp1_failure.sh
    run_exp2_elastic.sh
    run_exp3_publishable.sh
  results/
```

## Training and Recovery Semantics

### Dataset with IDs

Supported dataset wrappers return `(x, y, sample_id)` where `sample_id` is dataset index:

- `CIFAR10WithIDs`
- `CIFAR100WithIDs`
- `ImageFolderWithIDs`
- `FakeDataWithIDs`

Supported `dataset.name` values:

- `cifar10`
- `cifar100`
- `imagefolder` (reads `<root>/<split_subdir>`)
- `fake`

Supported `training.model_name` values:

- `small_cnn`
- `resnet18`
- `resnet34`
- `resnet50`
- `efficientnet_b0`
- `mobilenet_v3_large`

Recommended scaling knobs for larger runs:

- `training.precision`: `fp32`, `bf16`, `fp16` (CUDA only for bf16/fp16)
- `training.scheduler`: `none`, `cosine`
- `training.use_augmentation`: `true`/`false`
- `training.max_grad_norm`: gradient clipping threshold (`0` disables)

### Resumable Sampler

`ResumableGlobalBatchSampler` implements deterministic per-epoch permutation seeded by `(seed, epoch)`, fixed epoch truncation, and per-rank contiguous slicing of each global batch window. It stores and restores:

- `epoch`
- `cursor_step`
- `seed`

Resume and elastic `N -> M` reuse the same global windows with runtime `world_size` slicing.

### StatePack

Captured checkpoint payload includes:

- model, optimizer, optional scheduler state
- sampler state
- step state (`epoch`, `global_step`, `cursor_step`)
- RNG state (Python, NumPy, torch CPU, torch CUDA-all if available)

### Checkpoint Strategies

- Blocking periodic (`barrier -> capture -> rank0 atomic write -> barrier`)
- Overlapped periodic (`barrier -> capture -> rank0 enqueue -> barrier`) with bounded inflight writes and backpressure.

### Failure Injection

Configured global steps cause rank 0 to exit with code `137` at a post-step boundary after checkpoint boundary synchronization.

### Supervisor

The supervisor relaunches training via `python -m torch.distributed.run`, uses `latest.json` for resume, and continues until `target_steps` is reached.

## Logging

Per rank JSONL logs:

```text
results/logs/<run_id>/rank{rank}.jsonl
```

Per-step record fields:

- `time`, `run_id`, `rank`, `world_size`, `epoch`, `global_step`, `cursor_step`,
- `loss`, `sample_ids_hash`, `sample_ids_count`

Rank 0 debug IDs every `DEBUG_EVERY` steps:

- `results/logs/<run_id>/debug_rank0_ids.jsonl`

Checkpoint timing (rank 0):

- `results/logs/<run_id>/checkpoint_rank0.jsonl`

## Metrics

- `correctness.py`: deterministic data-progress checker using reconstructed expected windows and logged hashes.
- `goodput.py`: `goodput = useful_steps / wall_clock_time`, plus replay/restart and checkpoint stall breakdown.
- `divergence.py`: model distance at fixed steps (`200, 400, 800`) and loss divergence stats.
- `plot.py`: loss curves and goodput comparison plots.
- `aggregate.py`: multi-run aggregation (mean/std/95% CI) for publishable reporting.

## Experiments

### Exp1: Failure Recovery

Runs:

- reference
- failure + blocking
- failure + overlapped

Command:

```bash
scripts/run_exp1_failure.sh 4
```

Use `2` instead of `4` if resources are limited.
Common overrides:

```bash
TARGET=300 FAIL_STEPS=120,240 K=25 SEED=1337 scripts/run_exp1_failure.sh 2
```

### Exp2: Elastic Resume

Runs:

- reference (`N=4`)
- elastic (`4 -> 2`) via phase A then phase B resume

Command:

```bash
scripts/run_exp2_elastic.sh 4 2
```

Common overrides:

```bash
TARGET_FINAL=400 PHASE_A_TARGET=200 K=25 SEED=1337 scripts/run_exp2_elastic.sh 2 1
```

### Exp3: Publishable-Mode (Larger Model + Dataset + Multi-Seed)

Defaults:

- Dataset: CIFAR100
- Model: ResNet34
- Seeds: `1337,2027,4242`
- Precision: `bf16` (falls back to `fp32` on CPU/MPS)

Command:

```bash
scripts/run_exp3_publishable.sh 4
```

Alternate presets:

```bash
CONFIG=configs/exp3_publishable_1gpu.yaml scripts/run_exp3_publishable.sh 1
CONFIG=configs/exp3_publishable_4gpu.yaml scripts/run_exp3_publishable.sh 4
```

Common overrides:

```bash
SEEDS_CSV=1337,2027 TARGET=800 CHECKPOINT_EVERY=40 MAX_INFLIGHT=4 scripts/run_exp3_publishable.sh 4
```

Aggregate outputs:

- `results/metrics/_aggregate/exp3_reference.json`
- `results/metrics/_aggregate/exp3_failure_blocking.json`
- `results/metrics/_aggregate/exp3_failure_overlapped.json`
- `results/reports/exp3_publishable.md`
- `results/reports/exp3_publishable.json`

Budget notes:

- Single consumer GPU / laptop: use `NPROC=1..2`, keep `model_name=resnet18`, reduce `TARGET`.
- 24GB+ GPU: `resnet34` and `global_batch=256` are typically viable.
- Multi-GPU server: use `NPROC=4` and keep global batch fixed as required.

## Typical Outputs

- Checkpoints: `results/checkpoints/<run_id>/step_XXXXXXXX.pt`
- Latest pointer: `results/checkpoints/<run_id>/latest.json`
- Metrics: `results/metrics/<run_id>/...`
- Plots (Exp1): `results/plots/loss_curves_exp1.png`, `results/plots/goodput_exp1.png`
- Plots (Exp2): `results/plots/loss_curves_exp2.png`, `results/plots/goodput_exp2.png`
- Final summary: `results/final_summary.md`

## Validation

Run unit tests (inside the venv):

```bash
source .venv/bin/activate
uv run --python .venv/bin/python -m unittest discover -s tests -v
```

## Notes

- Overlapped checkpoint worker performs filesystem writes only and does not call `torch.distributed` APIs.
- Correctness checker uses expected-window reconstruction and hash validation; debug raw IDs are available for diagnosis.
