# ECRL: Elastic and Correct Recovery Lab (DDP)

ECRL is a reproducible systems-evaluation framework for failure recovery in PyTorch DDP training.

Current focus is the Exp4 matrix workflow:
- run many controlled failure-recovery suites,
- compare `blocking` vs `overlapped` checkpointing,
- aggregate publishable metrics and plots.

This repository is a recovery-semantics and measurement project, not a new checkpointing algorithm.

## What ECRL Guarantees

ECRL enforces the following runtime invariants:

- DDP data-parallel training only.
- Constant global batch across fresh runs and resumes.
- Runtime divisibility check: `global_batch % world_size == 0`.
- `LOCAL_BATCH = GLOBAL_BATCH // world_size`.
- Deterministic epoch geometry:
  - `steps_per_epoch = floor(dataset_size / GLOBAL_BATCH)`
  - `epoch_samples = steps_per_epoch * GLOBAL_BATCH`
- Rank-0-only checkpoint writes.
- Atomic checkpoint publication (`tmp -> fsync -> rename -> latest.json`).
- Failure injection only at synchronized step boundaries.
- Overlapped checkpoint writer performs filesystem I/O only (no distributed ops in worker).

Correctness criterion is strict data-progress equivalence (no duplicate/missing/extra sample semantics per epoch window).

## Environment

Recommended:

- Linux + CUDA GPUs for main experiments.
- Python 3.11.x.
- `uv` for environment management.

Basic setup:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt
```

Notes:

- Exp4 scripts auto-create `.venv` if missing; legacy Exp1/2/3 scripts expect it to exist.
- CIFAR archives are auto-fetched by scripts when needed (can be slow/blocked on some networks).

## Repository Layout

```text
ecrl/
  ecrl/
    ckpt/                  # blocking + overlapped checkpoint implementations
    data/                  # dataset wrappers returning (x, y, sample_id)
    metrics/               # correctness/goodput/divergence/aggregate/plot/report
    orchestration/         # restart supervisor
    sampler/               # resumable deterministic sampler
    statepack/             # checkpoint payload + RNG state handling
    train/                 # DDP train entrypoint
  configs/
    exp1_failure.yaml
    exp2_elastic.yaml
    exp3_publishable*.yaml
  scripts/
    run_exp1_failure.sh
    run_exp2_elastic.sh
    run_exp3_publishable.sh
    run_exp4_paperlite.sh
    run_exp4_paper_matrix.sh
    run_exp4_comprehensive_study.sh
  tests/
  results/                 # ignored by git (runtime outputs)
```

## Supported Dataset and Model Names

Dataset names (`dataset.name`):

- `cifar10`
- `cifar100`
- `imagefolder`
- `fake`

Model names (`training.model_name`):

- `small_cnn`
- `resnet18`
- `resnet34`
- `resnet50`
- `efficientnet_b0`
- `mobilenet_v3_large`

Precision (`training.precision`):

- `fp32`
- `bf16`
- `fp16`

## Recommended Workflows

### 1) Exp4 Paperlite (single suite, low quota)

Good default for quick, resume-safe runs with publishable JSON/MD output.

```bash
scripts/run_exp4_paperlite.sh 4
```

Common overrides:

```bash
PROFILE=small \
DATASET_NAME=cifar10 \
MODEL_NAME=resnet18 \
SEEDS_CSV=1337 \
TARGET_STEPS=800 \
FAIL_STEPS=200,600 \
CHECKPOINT_EVERY=50 \
MAX_INFLIGHT=4 \
scripts/run_exp4_paperlite.sh 1
```

Key Exp4 paperlite env vars:

- `RUN_PREFIX` (default `exp4_paperlite_<timestamp>`)
- `RESULTS_DIR` (default `results/<RUN_PREFIX>`)
- `PROFILE` (`small|balanced|large`)
- `RESUME_SUITE` (`1` to skip completed run/metrics/divergence units)
- `START_RESUME_LATEST` (`1` to start from latest checkpoint if available)
- `REQUIRE_CUDA` (`1` by default)

### 2) Exp4 Paper Matrix (multiple suites, resume-safe)

This is the main matrix orchestrator used for report-style analysis.

```bash
scripts/run_exp4_paper_matrix.sh 4
```

Typical matrix command:

```bash
PROFILE=small \
MODELS_CSV="resnet18,resnet50" \
DATASETS_CSV="cifar10,cifar100" \
FAILURE_SPECS="base:400,1200;late:800,1400" \
CHECKPOINT_EVERY_CSV="50" \
MAX_INFLIGHT_CSV="4" \
MATRIX_PREFIX=exp4_matrix_$(date +%Y%m%d_%H%M%S) \
scripts/run_exp4_paper_matrix.sh 4
```

Useful controls:

- `SKIP_IF_EXISTS=1` skip suite if publishable JSON already exists.
- `RESUME_MATRIX=1` reuse previous manifest-completed suites.
- `START_RESUME_LATEST=1` pass resume-latest behavior into each suite.
- `CONTINUE_ON_ERROR=1` continue matrix after failures.
- `DRY_RUN=1` generate manifest plan without executing suites.

Nohup example:

```bash
mkdir -p results
ts=$(date +%Y%m%d_%H%M%S)
nohup env PROFILE=small FAILURE_SPECS='base:400,1200;late:800,1400' \
  MATRIX_PREFIX="exp4_matrix_${ts}" \
  bash scripts/run_exp4_paper_matrix.sh 4 \
  > "results/exp4_matrix_nohup_${ts}.log" 2>&1 &
```

### 3) Exp4 Comprehensive Study (high budget)

Large orchestrator with baseline, checkpoint-frequency sweep, inflight sweep, failure-schedule sweep, and elastic `N->M`.

```bash
scripts/run_exp4_comprehensive_study.sh 4
```

This runner can consume substantial GPU hours. Use only when you have sufficient quota.

### 4) Legacy scripts

Exp1:

```bash
scripts/run_exp1_failure.sh 4
```

Exp2:

```bash
scripts/run_exp2_elastic.sh 4 2
```

Exp3:

```bash
scripts/run_exp3_publishable.sh 4
```

## Artifact Map

Per-run artifacts are under each suite `results_dir`.

Core files:

- Checkpoints: `results/.../checkpoints/<run_id>/step_XXXXXXXX.pt`
- Latest pointer: `results/.../checkpoints/<run_id>/latest.json`
- Rank logs: `results/.../logs/<run_id>/rank*.jsonl`
- Supervisor state:
  - `results/.../logs/<run_id>/supervisor_attempts.jsonl`
  - `results/.../logs/<run_id>/supervisor.json`
- Checkpoint timing (rank 0): `results/.../logs/<run_id>/checkpoint_rank0.jsonl`
- Metrics per run:
  - `results/.../metrics/<run_id>/correctness.json`
  - `results/.../metrics/<run_id>/goodput.json`
  - `results/.../metrics/<run_id>/divergence_vs_<reference>.json`
- Aggregate metrics: `results/.../metrics/_aggregate/*.json`
- Plots: `results/.../plots/*.png` (and optional PDF from `--save-pdf`)
- Publishable report (paperlite/exp3):
  - `results/.../reports/<prefix>_publishable.md`
  - `results/.../reports/<prefix>_publishable.json`

Exp4 matrix-level files (at `RESULTS_ROOT`, usually `results/`):

- `<MATRIX_PREFIX>_manifest.csv`
- `<MATRIX_PREFIX>_manifest.md`
- `<MATRIX_PREFIX>_summary.csv`
- `<MATRIX_PREFIX>_summary.md`

## Resume and Progress Behavior

- Supervisor restarts after injected failures until target steps are complete.
- Expected in failure runs:
  - attempt 1 ends near first fail step,
  - attempt 2 resumes from latest checkpoint,
  - additional attempts continue until target completion.
- `supervisor_attempts.jsonl` is the source of truth for restart progression.

When verifying progress quickly:

```bash
tail -n 20 results/<run>/logs/<run_id>/supervisor_attempts.jsonl
tail -n 5  results/<run>/logs/<run_id>/checkpoint_rank0.jsonl
tail -n 1  results/<run>/logs/<run_id>/rank0.jsonl
```

## Cleanup

After a run is fully completed and reports are generated, checkpoints are usually safe to remove.

Delete checkpoints for one run root:

```bash
find results/<run_prefix> -type d -name checkpoints -prune -exec rm -rf {} +
```

Or only remove `.pt` files while keeping `latest.json` directories:

```bash
find results/<run_prefix>/checkpoints -type f -name 'step_*.pt' -delete
```

## Troubleshooting

`address already in use` (`EADDRINUSE`):

- Another run is occupying `master_port`.
- Set a different `MASTER_PORT_BASE` or stop stale processes.

No progress after `attempt_start`:

- Check `rank0.jsonl` growth and `checkpoint_rank0.jsonl`.
- If `global_step` keeps advancing, training is active even if attempt log is quiet.

`TypeError: RNG state must be a torch.ByteTensor`:

- Indicates old code/runtime mismatch.
- Sync latest repo and rerun (scripts perform runtime patch checks in Exp4 runners).

CIFAR download hangs/fails:

- Network/firewall issue to Toronto dataset host.
- Pre-download archives into `data/` manually, then rerun.

GPU overuse/quota pressure:

- Reduce `NPROC`, `TARGET_STEPS`, model size, seed count, and matrix breadth.
- Prefer `run_exp4_paperlite.sh` or `PROFILE=small` matrix runs first.

## Tests

Run unit tests:

```bash
source .venv/bin/activate
uv run --python .venv/bin/python -m unittest discover -s tests -v
```

## Local Paper Artifacts

- `overleaf/` and `overleaf.zip` are intentionally git-ignored local assets.
- Keep final deliverables under `reports/final/` (recommended) if you want stable local report paths.
