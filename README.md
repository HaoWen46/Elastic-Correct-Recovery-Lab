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
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

## Repository Layout

```text
ecrl/
  README.md
  requirements.txt
  configs/
    exp1_failure.yaml
    exp2_elastic.yaml
  ecrl/
    data/cifar_with_ids.py
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
  scripts/
    run_exp1_failure.sh
    run_exp2_elastic.sh
  results/
```

## Training and Recovery Semantics

### Dataset with IDs

`CIFAR10WithIDs` returns `(x, y, sample_id)` where `sample_id` is dataset index.

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

### Exp2: Elastic Resume

Runs:

- reference (`N=4`)
- elastic (`4 -> 2`) via phase A then phase B resume

Command:

```bash
scripts/run_exp2_elastic.sh 4 2
```

## Typical Outputs

- Checkpoints: `results/checkpoints/<run_id>/step_XXXXXXXX.pt`
- Latest pointer: `results/checkpoints/<run_id>/latest.json`
- Metrics: `results/metrics/<run_id>/...`
- Plots: `results/plots/loss_curves.png`, `results/plots/goodput.png`

## Notes

- Overlapped checkpoint worker performs filesystem writes only and does not call `torch.distributed` APIs.
- Correctness checker uses expected-window reconstruction and hash validation; debug raw IDs are available for diagnosis.
