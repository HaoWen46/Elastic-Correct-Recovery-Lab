# ECRL Comprehensive Project Report

Date: February 13, 2026

## 1. Executive Summary

ECRL (Elastic & Correct Recovery Lab) is a reproducible evaluation framework for fault-tolerant distributed training using PyTorch DDP data parallelism. The project is centered on recovery semantics, not on inventing a new checkpoint/restart algorithm.

The framework evaluates three core properties:

1. Correctness under failures (restart semantics and data-progress integrity).
2. Goodput under failures (useful training steps per wall-clock second).
3. Elastic resume (`N -> M`) with a constant global batch size.

The implementation enforces strict invariants for determinism and restart correctness, including fixed epoch geometry, rank-0-only atomic checkpoint writes, boundary-synchronized failure injection, and deterministic sampler state with per-step cursor tracking.

All required experiments were executed successfully in a `uv`-managed Python 3.11.2 environment with publication-style artifacts generated (JSON metrics, aggregate summaries, plots, and markdown reports).

## 2. Project Goals and Scope

### 2.1 Goals

- Build a reproducible, end-to-end framework to measure recovery quality in DDP training.
- Validate that recovery and elastic resume preserve data-progress correctness.
- Quantify runtime impact of checkpoint strategy and failures.
- Produce artifacts suitable for research reporting (metrics and plots).

### 2.2 Scope

- DDP data parallel only.
- Checkpoint/restart semantics at training-step boundaries.
- Rank-0 persistence only with atomic file semantics.
- Two checkpoint strategies:
  - Blocking periodic checkpoints.
  - Overlapped periodic checkpoints with a background process for I/O.

### 2.3 Explicit Non-Goals

- No FSDP.
- No OS-level or GPU-level checkpoint/restart.
- No paper-system reimplementation.
- No attempt to optimize model quality; focus is systems semantics and measurement.

## 3. Hard Constraints and Enforcement

The implementation enforces the non-negotiable requirements:

- Python `3.11.2` only.
- Virtual environment required (`uv` + `.venv`).
- `GLOBAL_BATCH` is constant across launches/resumes.
- Runtime assertion: `GLOBAL_BATCH % world_size == 0`.
- `LOCAL_BATCH = GLOBAL_BATCH // world_size`.
- `DataLoader(num_workers=0, drop_last=True)`.
- Fixed epoch geometry:
  - `D = len(dataset)`
  - `steps_per_epoch = floor(D / GLOBAL_BATCH)`
  - `epoch_samples = steps_per_epoch * GLOBAL_BATCH`
- Epoch progression by explicit step loop (not dataloader exhaustion).
- Rank-0-only checkpoint writing.
- Atomic checkpoint write protocol (temp file, fsync, rename, latest pointer update).
- Barriers around checkpoint capture boundaries.
- Failure injection only after step completion and synchronization.
- Overlapped writer is a background process doing filesystem I/O only.

## 4. Repository and Component Map

Primary repository path:

- `/Users/haowenchen/Files/projects/ecrl`

Key components:

- `ecrl/data/datasets_with_ids.py`
  - Dataset wrappers returning `(x, y, sample_id)`.
- `ecrl/sampler/resumable_sampler.py`
  - Deterministic global-window sampler with cursor state and elastic slicing.
- `ecrl/statepack/statepack.py`
  - Capture/restore model, optimizer, scheduler, sampler, step state, RNG.
- `ecrl/ckpt/atomic_writer.py`
  - Atomic rank-0 checkpoint persistence + `latest.json`.
- `ecrl/ckpt/blocking.py`
  - Blocking checkpoint strategy with synchronized stall accounting.
- `ecrl/ckpt/overlapped.py`
  - Overlapped checkpoint strategy with background I/O process and backpressure.
- `ecrl/train/ddp_train.py`
  - Core DDP training loop, checkpoint integration, failure injection.
- `ecrl/orchestration/supervisor.py`
  - Auto-restart launcher using latest checkpoint pointer.
- `ecrl/metrics/*.py`
  - Correctness, goodput, divergence, aggregation, plotting, publishable report generation.
- `scripts/run_exp1_failure.sh`
- `scripts/run_exp2_elastic.sh`
- `scripts/run_exp3_publishable.sh`

## 5. Core Semantics

### 5.1 Dataset IDs and Observable Data Progress

The framework uses dataset wrappers that expose sample IDs:

- `CIFAR10WithIDs`
- `CIFAR100WithIDs`
- `ImageFolderWithIDs`
- `FakeDataWithIDs`

Logging hashes of rank-local sample ID windows per step enables scalable correctness verification without always storing raw IDs.

### 5.2 Resumable Global-Batch Sampler

`ResumableGlobalBatchSampler` is the correctness backbone.

For each epoch `e`:

1. Build a deterministic permutation with seed derived from `(seed, e)`.
2. Truncate to `epoch_samples`.
3. At `cursor_step = s`, take global window:
   - `perm_epoch[s*GLOBAL_BATCH : (s+1)*GLOBAL_BATCH]`
4. Rank `r` gets its contiguous shard:
   - `window[r*LOCAL_BATCH : (r+1)*LOCAL_BATCH]`

State:

- `epoch`
- `cursor_step`
- `seed`

After each committed step, rank 0 advances `cursor_step` and broadcasts the update.

Why this matters:

- Restart is exact at step boundaries.
- Elastic resume naturally re-shards the same global window when `world_size` changes.
- Data-progress correctness can be tested against deterministic expected windows.

### 5.3 Step and Epoch Semantics

Epoch length is fixed by geometry and enforced with explicit step loops. This avoids silent semantic drift from dataloader exhaustion and ensures reproducible per-step window mapping.

## 6. Checkpointing Design

### 6.1 Atomic Writer

`atomic_writer.py` persists checkpoint payloads atomically:

1. Serialize to temp file.
2. Flush + `fsync`.
3. `os.replace` to final `step_XXXXXXXX.pt`.
4. `fsync` checkpoint directory.
5. Atomically update `latest.json` with:
   - path
   - global step
   - epoch
   - world size at save
   - timestamp

### 6.2 Blocking Strategy

`BlockingPeriodicCheckpointer` behavior every `K` steps:

1. Barrier before capture.
2. Rank 0 captures state.
3. Rank 0 writes checkpoint atomically.
4. Barrier after write.

Recorded timing:

- snapshot time
- write time
- total stall time

### 6.3 Overlapped Strategy

`OverlappedPeriodicCheckpointer` behavior every `K` steps:

1. Barrier before capture.
2. Rank 0 captures and serializes payload.
3. Rank 0 enqueues to a spawned background process.
4. Barrier after enqueue.

Features:

- Multiple in-flight writes.
- `MAX_INFLIGHT` backpressure.
- Completion queue for write completion records.
- `flush(wait=True)` before forced exit/finalization.

Important guarantee:

- Background worker performs filesystem I/O only.
- No `torch.distributed` API calls in background process.

## 7. Failure Injection and Recovery

Failure injection model:

- Triggered only on configured global steps.
- Executed by rank 0 with `os._exit(137)`.
- Only after:
  - step is complete,
  - checkpoint boundary logic is complete,
  - synchronization barriers are satisfied.

Supervisor behavior:

- Launches training via `torch.distributed.run`.
- On non-zero exit, reads `latest.json`.
- Restarts with `--resume-latest`.
- Stops when `target_steps` reached or restart limit exceeded.

This ensures failures are meaningful (boundary-safe) and recovery is testable.

## 8. Metrics Methodology

### 8.1 Correctness

`metrics/correctness.py` reconstructs expected windows and validates:

- rank/world-size consistency
- expected epoch/cursor per global step
- rank-local sample ID hash/count match

Per-epoch invariants:

- duplicates = 0
- missing = 0
- extra = 0

### 8.2 Goodput

`metrics/goodput.py` computes:

- `goodput = useful_steps / wall_clock_time`
- restarts
- replayed steps
- checkpoint timing totals:
  - snapshot
  - write
  - enqueue latency
  - backpressure wait
  - stall

### 8.3 Divergence

`metrics/divergence.py` compares checkpoint states at selected steps:

- parameter L2 distance
- deterministic digest comparison

Also computes trajectory-level loss divergence:

- max absolute loss diff
- mean absolute loss diff
- AUC of absolute loss difference

### 8.4 Aggregate and Report

- `metrics/aggregate.py` computes mean/std/95% CI across runs.
- `metrics/report_publishable.py` builds markdown/json publication summaries.

## 9. Experimental Protocol

Execution date:

- February 13, 2026

Environment:

- Python 3.11.2
- Torch 2.10.0
- `uv 0.10.2`
- `.venv` managed by `uv`
- CUDA unavailable on this machine; MPS available.

Full run output directory:

- `/Users/haowenchen/Files/projects/ecrl/results/pub_n2_20260213`

Experiments run:

1. Exp1 failure recovery:
   - `NPROC=2`
   - `T=1000`
   - `FAIL_STEPS=200,600`
2. Exp2 elastic resume:
   - `2 -> 1`
   - phase A `400`, final `800`
3. Exp3 publishable run:
   - dataset/model: CIFAR100 + ResNet18
   - seeds: `1337,2027`
   - `T=600`
   - `FAIL_STEPS=200,500`

## 10. Results

### 10.1 Exp1: Failure Recovery

Correctness:

- All three runs passed correctness checks.

Goodput (steps/s):

- Reference: `8.4707`
- Failure + Blocking: `7.7947` (`-7.98%` vs reference)
- Failure + Overlapped: `7.6332` (`-9.89%` vs reference)

Restarts:

- Reference: `0`
- Blocking: `2`
- Overlapped: `2`

Divergence:

- Loss divergence AUC vs reference:
  - Blocking: `0.0`
  - Overlapped: `0.0`

Interpretation:

- Recovery semantics were exact for this workload; failures did not alter final trajectory under this setup.

### 10.2 Exp2: Elastic Resume (2 -> 1)

Correctness:

- Reference passed.
- Elastic run passed.

Goodput (steps/s):

- Reference: `8.5400`
- Elastic: `7.6322` (`-10.63%`)

Divergence:

- Loss divergence AUC: `26.0168`
- `L2@800`: `12.2542`

Interpretation:

- Data-progress semantics remained correct under world-size change.
- Parameter trajectory diverged later in training despite deterministic data window semantics, reflecting expected optimization-path sensitivity when execution context changes.

### 10.3 Exp3: Larger Dataset/Model, Multi-Seed Aggregate

Configuration:

- CIFAR100 + ResNet18
- 2 seeds (`1337`, `2027`)
- 600 target steps

Correctness pass-rate:

- Reference: `1.00`
- Failure + Blocking: `1.00`
- Failure + Overlapped: `1.00`

Mean goodput (steps/s):

- Reference: `1.0334`
- Blocking: `0.9685` (`-6.28%`)
- Overlapped: `0.9521` (`-7.86%`)

Mean restarts:

- Reference: `0.0`
- Blocking: `2.0`
- Overlapped: `2.0`

Divergence across 4 failure-vs-reference pairs:

- Mean loss AUC abs diff: `32.2130`
- Mean max abs loss diff: `0.3015`
- Mean `L2@600`: `23.4734`

Interpretation:

- Correctness remained perfect even under heavier workload and repeated failures.
- Throughput loss under failure is measurable but bounded.
- Optimization trajectory divergence is present while data-progress semantics are preserved.

## 11. Discussion

### 11.1 What Worked Well

- Deterministic sampler + explicit cursor state enabled robust correctness checks.
- Atomic rank-0 checkpoints produced reliable restart points.
- Supervisor logic handled repeated injected failures cleanly.
- The metrics stack scales from single-run checks to aggregate reporting.

### 11.2 Key Observations

- Correctness invariants were satisfied in all reported runs.
- Overlapped checkpointing did not outperform blocking in this CPU-bound environment.
- Elastic resume preserved data-progress correctness but not identical optimizer trajectory at late steps.

### 11.3 Why Overlapped May Not Win Here

On this machine/workload:

- CPU-only training path.
- Serialization and process overhead can offset overlap benefit.
- Checkpoint intervals and model size interact with I/O contention.

In a GPU-heavy or I/O-heavy regime, overlap can shift the tradeoff.

## 12. Limitations and Threats to Validity

- Hardware-specific: results are from one environment without CUDA.
- Exp2 naming includes legacy run IDs (`exp2_reference_n4`, `exp2_elastic_4to2`) while executed shape was `2 -> 1` in this run set.
- Exp3 aggregate uses 2 seeds (not 3+) for this publication test pass.
- Divergence reflects trajectory differences, not correctness failure.

## 13. Reproducibility and Validation Checklist

- Python version fixed: `3.11.2`.
- Virtual environment: `uv` + `.venv`.
- Unit tests passed.
- Full experiment scripts executed end-to-end.
- Correctness metrics generated for all runs.
- Goodput/divergence metrics generated for all required comparisons.
- Plot artifacts generated.
- Consolidated publication summaries generated in markdown and JSON.

## 14. Artifact Index

Primary run directory:

- `/Users/haowenchen/Files/projects/ecrl/results/pub_n2_20260213`

Consolidated summaries:

- `/Users/haowenchen/Files/projects/ecrl/results/pub_n2_20260213/publication_summary.md`
- `/Users/haowenchen/Files/projects/ecrl/results/pub_n2_20260213/publication_summary.json`

Publishable experiment report:

- `/Users/haowenchen/Files/projects/ecrl/results/pub_n2_20260213/reports/exp3_publishable.md`
- `/Users/haowenchen/Files/projects/ecrl/results/pub_n2_20260213/reports/exp3_publishable.json`

Plots:

- `/Users/haowenchen/Files/projects/ecrl/results/pub_n2_20260213/plots/loss_curves_exp1.png`
- `/Users/haowenchen/Files/projects/ecrl/results/pub_n2_20260213/plots/goodput_exp1.png`
- `/Users/haowenchen/Files/projects/ecrl/results/pub_n2_20260213/plots/loss_curves_exp2.png`
- `/Users/haowenchen/Files/projects/ecrl/results/pub_n2_20260213/plots/goodput_exp2.png`
- `/Users/haowenchen/Files/projects/ecrl/results/pub_n2_20260213/plots/loss_curves_exp3_publishable.png`
- `/Users/haowenchen/Files/projects/ecrl/results/pub_n2_20260213/plots/goodput_exp3_publishable.png`

Per-run metrics and logs:

- `/Users/haowenchen/Files/projects/ecrl/results/pub_n2_20260213/metrics`
- `/Users/haowenchen/Files/projects/ecrl/results/pub_n2_20260213/logs`
- `/Users/haowenchen/Files/projects/ecrl/results/pub_n2_20260213/checkpoints`

## 15. Conclusion

ECRL now provides a complete, reproducible framework for evaluating distributed recovery semantics in DDP with strong invariants and measurable outcomes. The implemented system demonstrates:

- strict data-progress correctness under failure and elastic resume,
- quantitative goodput impacts across checkpoint strategies,
- practical automation from run orchestration to publishable summaries.

The project is suitable as a systems evaluation baseline and can be extended to broader hardware/model scales while preserving the same correctness contract.
