#!/usr/bin/env bash
set -euo pipefail

# Low-quota Exp4 "paperlite" pipeline:
# - Small/balanced/large tiers
# - Per-seed 3 runs: reference, failure+blocking, failure+overlapped
# - correctness/goodput/divergence/aggregate/plots/publishable report
#
# Usage:
#   scripts/run_exp4_paperlite.sh [NPROC]
#
# Example:
#   scripts/run_exp4_paperlite.sh 2

NPROC="${1:-${NPROC:-2}}"
TARGET_STEPS="${TARGET_STEPS:-1600}"
SEED="${SEED:-1337}"
PROFILE="${PROFILE:-balanced}"   # small|balanced|large
SEEDS_CSV="${SEEDS_CSV:-}"
FAIL_STEPS="${FAIL_STEPS:-400,1200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
MAX_INFLIGHT="${MAX_INFLIGHT:-4}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-$((30000 + RANDOM % 20000))}"

MODEL_NAME="${MODEL_NAME:-resnet50}"
DATASET_NAME="${DATASET_NAME:-cifar100}"   # cifar10|cifar100|imagefolder|fake
PRECISION="${PRECISION:-bf16}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
LR="${LR:-0.1}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.0}"
USE_AUGMENTATION="${USE_AUGMENTATION:-true}"
LOG_EVERY="${LOG_EVERY:-10}"
DEBUG_EVERY="${DEBUG_EVERY:-100}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SETUP_ENV="${SETUP_ENV:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
MAX_RESTARTS_NO_CKPT="${MAX_RESTARTS_NO_CKPT:-3}"
RESTART_DELAY_SEC="${RESTART_DELAY_SEC:-1.0}"
DIVERGENCE_STEPS="${DIVERGENCE_STEPS:-200,400,800,1200,1600}"
START_RESUME_LATEST="${START_RESUME_LATEST:-1}"
RESUME_SUITE="${RESUME_SUITE:-1}"
IMAGEFOLDER_ROOT="${IMAGEFOLDER_ROOT:-}"
IMAGEFOLDER_SPLIT_SUBDIR="${IMAGEFOLDER_SPLIT_SUBDIR:-train}"
DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-10}"  # used by fake/imagefolder

RUN_PREFIX="${RUN_PREFIX:-exp4_paperlite_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-results/${RUN_PREFIX}}"
CONFIG_PATH="${RESULTS_DIR}/configs/${RUN_PREFIX}.yaml"

run_py() {
  uv run --python "${PYTHON_BIN}" "$@"
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  echo "${s}"
}

join_csv() {
  local IFS=","
  echo "$*"
}

resolve_seeds() {
  if [ -z "${SEEDS_CSV}" ]; then
    case "${PROFILE}" in
      small)
        SEEDS_CSV="${SEED}"
        ;;
      balanced)
        SEEDS_CSV="${SEED},2027"
        ;;
      large)
        SEEDS_CSV="${SEED},2027,4242"
        ;;
      *)
        echo "Unsupported PROFILE=${PROFILE}. Use small|balanced|large." >&2
        exit 1
        ;;
    esac
  fi

  IFS=',' read -r -a SEEDS_RAW <<< "${SEEDS_CSV}"
  SEEDS=()
  for s in "${SEEDS_RAW[@]}"; do
    s="$(trim "${s}")"
    [ -n "${s}" ] || continue
    SEEDS+=("${s}")
  done
  if [ "${#SEEDS[@]}" -eq 0 ]; then
    echo "SEEDS_CSV produced no valid seeds" >&2
    exit 1
  fi
  BASE_SEED="${SEEDS[0]}"
}

next_port() {
  local p="${MASTER_PORT_BASE}"
  MASTER_PORT_BASE="$((MASTER_PORT_BASE + 1))"
  echo "${p}"
}

ensure_env() {
  if [ ! -x "${PYTHON_BIN}" ]; then
    if [ "${PYTHON_BIN}" = ".venv/bin/python" ]; then
      local venv_python="${VENV_PYTHON:-}"
      if [ -z "${venv_python}" ]; then
        if command -v python3 >/dev/null 2>&1; then
          venv_python="python3"
        elif command -v python >/dev/null 2>&1; then
          venv_python="python"
        else
          venv_python="3.11"
        fi
      fi
      uv venv --python "${venv_python}" .venv
    else
      echo "Missing Python interpreter at ${PYTHON_BIN}" >&2
      exit 1
    fi
  fi

  if [ "${SETUP_ENV}" = "1" ]; then
    uv pip install --python "${PYTHON_BIN}" -r requirements.txt
  fi
}

verify_runtime_patches() {
  local patch_status
  patch_status="$(run_py - <<'PY'
import inspect
import sys
from ecrl.ckpt.overlapped import OverlappedPeriodicCheckpointer
from ecrl.statepack import rng
from ecrl.train import ddp_train

src = inspect.getsource(rng.restore_rng_state)
if "_as_cpu_byte_tensor" not in src:
    print("missing_rng_patch")
    sys.exit(1)
src_ovl = inspect.getsource(OverlappedPeriodicCheckpointer.flush)
if "_ensure_worker_alive" not in src_ovl:
    print("missing_overlapped_flush_patch")
    sys.exit(1)
src_train = inspect.getsource(ddp_train.main)
if "all ranks exiting with 137 after synchronized flush" not in src_train:
    print("missing_overlapped_fail_inject_patch")
    sys.exit(1)
if "close_done_sec" not in src_train:
    print("missing_overlapped_fail_inject_close_patch")
    sys.exit(1)
print("ok")
PY
  )" || {
    echo "Runtime patch check failed: required recovery patches are missing." >&2
    exit 1
  }
  echo "[INFO] Runtime patch check: ${patch_status}"
}

probe_cuda_count() {
  run_py - <<'PY'
import torch
print(torch.cuda.device_count())
PY
}

enforce_cuda_guard() {
  local cuda_count
  cuda_count="$(probe_cuda_count)"
  if [ "${REQUIRE_CUDA}" = "1" ]; then
    if [ "${cuda_count}" -lt 1 ]; then
      echo "REQUIRE_CUDA=1 but no CUDA devices visible." >&2
      exit 1
    fi
    if [ "${NPROC}" -gt "${cuda_count}" ]; then
      echo "NPROC=${NPROC} exceeds visible CUDA devices=${cuda_count}" >&2
      exit 1
    fi
  fi
}

ensure_cifar100() {
  mkdir -p data
  if [ -d data/cifar-100-python ]; then
    return
  fi
  if [ ! -f data/cifar-100-python.tar.gz ]; then
    curl -L --fail --retry 3 --retry-delay 2 \
      -o data/cifar-100-python.tar.gz \
      https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
  fi
  tar -xzf data/cifar-100-python.tar.gz -C data
}

ensure_cifar10() {
  mkdir -p data
  if [ -d data/cifar-10-batches-py ]; then
    return
  fi
  if [ ! -f data/cifar-10-python.tar.gz ]; then
    curl -L --fail --retry 3 --retry-delay 2 \
      -o data/cifar-10-python.tar.gz \
      https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  fi
  tar -xzf data/cifar-10-python.tar.gz -C data
}

count_imagefolder_samples() {
  local root="$1"
  local split="$2"
  run_py - "${root}" "${split}" <<'PY'
from pathlib import Path
import sys

base = Path(sys.argv[1]) / sys.argv[2]
if not base.is_dir():
    raise SystemExit(f"missing imagefolder split directory: {base}")

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
classes = [p for p in base.iterdir() if p.is_dir()]
if not classes:
    raise SystemExit(f"no class directories under {base}")

count = 0
for cls in classes:
    for fp in cls.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in exts:
            count += 1

if count <= 0:
    raise SystemExit(f"no image files found under {base}")

print(f"{count} {len(classes)}")
PY
}

prepare_dataset() {
  case "${DATASET_NAME}" in
    cifar100)
      ensure_cifar100
      DATASET_ROOT="./data"
      DATASET_DOWNLOAD="true"
      DATASET_SIZE="${DATASET_SIZE:-50000}"
      IMAGE_SIZE="${IMAGE_SIZE:-32}"
      ;;
    cifar10)
      ensure_cifar10
      DATASET_ROOT="./data"
      DATASET_DOWNLOAD="true"
      DATASET_SIZE="${DATASET_SIZE:-50000}"
      IMAGE_SIZE="${IMAGE_SIZE:-32}"
      ;;
    imagefolder)
      if [ -z "${IMAGEFOLDER_ROOT}" ]; then
        echo "DATASET_NAME=imagefolder requires IMAGEFOLDER_ROOT" >&2
        exit 1
      fi
      read -r DATASET_SIZE DATASET_NUM_CLASSES < <(count_imagefolder_samples "${IMAGEFOLDER_ROOT}" "${IMAGEFOLDER_SPLIT_SUBDIR}")
      DATASET_ROOT="${IMAGEFOLDER_ROOT}"
      DATASET_DOWNLOAD="false"
      IMAGE_SIZE="${IMAGE_SIZE:-224}"
      ;;
    fake)
      DATASET_ROOT="./data"
      DATASET_DOWNLOAD="false"
      DATASET_SIZE="${DATASET_SIZE:-50000}"
      IMAGE_SIZE="${IMAGE_SIZE:-32}"
      DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-10}"
      ;;
    *)
      echo "Unsupported DATASET_NAME=${DATASET_NAME}. Use cifar10|cifar100|imagefolder|fake." >&2
      exit 1
      ;;
  esac
}

generate_config() {
  mkdir -p "$(dirname "${CONFIG_PATH}")"
  cat > "${CONFIG_PATH}" <<EOF
seed: ${BASE_SEED}
dataset:
  name: ${DATASET_NAME}
  root: "${DATASET_ROOT}"
  train: true
  download: ${DATASET_DOWNLOAD}
  size: ${DATASET_SIZE}
  image_size: ${IMAGE_SIZE}
  split_subdir: "${IMAGEFOLDER_SPLIT_SUBDIR}"
training:
  model_name: ${MODEL_NAME}
  precision: ${PRECISION}
  use_augmentation: ${USE_AUGMENTATION}
  scheduler: cosine
  scheduler_t_max: ${TARGET_STEPS}
  scheduler_eta_min: 0.000001
  max_grad_norm: ${MAX_GRAD_NORM}
  global_batch: ${GLOBAL_BATCH}
  lr: ${LR}
  momentum: ${MOMENTUM}
  weight_decay: ${WEIGHT_DECAY}
  max_steps: ${TARGET_STEPS}
  log_every: ${LOG_EVERY}
  debug_every: ${DEBUG_EVERY}
checkpoint:
  strategy: blocking
  every_k_steps: ${CHECKPOINT_EVERY}
  max_inflight: ${MAX_INFLIGHT}
failure:
  enabled: true
  steps: [${FAIL_STEPS}]
EOF
  # overwrite resolved dataset fields in a structured way
  run_py - "${CONFIG_PATH}" "${DATASET_SIZE}" "${DATASET_NAME}" "${DATASET_NUM_CLASSES}" <<'PY'
from pathlib import Path
import sys
import yaml

cfg_path = Path(sys.argv[1])
size = int(sys.argv[2])
dataset_name = str(sys.argv[3]).lower()
num_classes = int(sys.argv[4])
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
dataset = cfg.setdefault("dataset", {})
dataset["size"] = size
if dataset_name in {"fake", "imagefolder"}:
    dataset["num_classes"] = num_classes
else:
    dataset.pop("num_classes", None)
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
}

run_supervisor() {
  local run_id="$1"
  local strategy="$2"
  local disable_failure="$3"
  local seed="$4"
  local port
  port="$(next_port)"

  local cmd=(
    -m ecrl.orchestration.supervisor
    --config "${CONFIG_PATH}"
    --run-id "${run_id}"
    --results-dir "${RESULTS_DIR}"
    --nproc-per-node "${NPROC}"
    --target-steps "${TARGET_STEPS}"
    --checkpoint-strategy "${strategy}"
    --checkpoint-every "${CHECKPOINT_EVERY}"
    --max-inflight "${MAX_INFLIGHT}"
    --seed "${seed}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${port}"
    --max-restarts "${MAX_RESTARTS}"
    --max-restarts-without-checkpoint "${MAX_RESTARTS_NO_CKPT}"
    --restart-delay-sec "${RESTART_DELAY_SEC}"
  )

  if [ "${START_RESUME_LATEST}" = "1" ]; then
    cmd+=(--start-resume-latest)
  fi

  if [ "${disable_failure}" = "1" ]; then
    cmd+=(--disable-failure)
  else
    cmd+=(--fail-steps "${FAIL_STEPS}")
  fi

  echo "[RUN] ${run_id} seed=${seed} nproc=${NPROC} target=${TARGET_STEPS} strategy=${strategy} fail=$([ "${disable_failure}" = "1" ] && echo none || echo "${FAIL_STEPS}")"
  run_py "${cmd[@]}"
}

is_run_completed() {
  local run_id="$1"
  local path="${RESULTS_DIR}/logs/${run_id}/supervisor.json"
  [ -f "${path}" ] || return 1
  grep -q '"status":[[:space:]]*"completed"' "${path}"
}

has_run_metrics() {
  local run_id="$1"
  [ -f "${RESULTS_DIR}/metrics/${run_id}/correctness.json" ] && [ -f "${RESULTS_DIR}/metrics/${run_id}/goodput.json" ]
}

has_divergence_metric() {
  local ref_run="$1"
  local cand_run="$2"
  [ -f "${RESULTS_DIR}/metrics/${cand_run}/divergence_vs_${ref_run}.json" ]
}

run_metrics_for() {
  local run_id="$1"
  local seed="$2"
  run_py -m ecrl.metrics.correctness \
    --config "${CONFIG_PATH}" \
    --run-id "${run_id}" \
    --results-dir "${RESULTS_DIR}" \
    --target-steps "${TARGET_STEPS}" \
    --seed "${seed}"
  run_py -m ecrl.metrics.goodput \
    --run-id "${run_id}" \
    --results-dir "${RESULTS_DIR}" \
    --target-steps "${TARGET_STEPS}"
}

run_divergence() {
  local ref_run="$1"
  local cand_run="$2"
  run_py -m ecrl.metrics.divergence \
    --reference-run "${ref_run}" \
    --candidate-run "${cand_run}" \
    --results-dir "${RESULTS_DIR}" \
    --steps "${DIVERGENCE_STEPS}"
}

echo "[INFO] Run prefix: ${RUN_PREFIX}"
echo "[INFO] Results dir: ${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

resolve_seeds
ensure_env
verify_runtime_patches
enforce_cuda_guard
prepare_dataset
generate_config

echo "[INFO] Profile: ${PROFILE}"
echo "[INFO] Dataset: ${DATASET_NAME} (size=${DATASET_SIZE})"
echo "[INFO] Model: ${MODEL_NAME}"
echo "[INFO] Seeds: $(join_csv "${SEEDS[@]}")"
echo "[INFO] NPROC: ${NPROC}"
echo "[INFO] Resume suite: ${RESUME_SUITE}"
echo "[INFO] Supervisor start-resume-latest: ${START_RESUME_LATEST}"

declare -a REF_RUNS=()
declare -a BLK_RUNS=()
declare -a OVL_RUNS=()

for seed in "${SEEDS[@]}"; do
  REF_RUN="${RUN_PREFIX}_s${seed}_reference"
  BLK_RUN="${RUN_PREFIX}_s${seed}_failure_blocking"
  OVL_RUN="${RUN_PREFIX}_s${seed}_failure_overlapped"

  REF_RUNS+=("${REF_RUN}")
  BLK_RUNS+=("${BLK_RUN}")
  OVL_RUNS+=("${OVL_RUN}")

  if [ "${RESUME_SUITE}" = "1" ] && is_run_completed "${REF_RUN}"; then
    echo "[SKIP] completed run: ${REF_RUN}"
  else
    run_supervisor "${REF_RUN}" "blocking" "1" "${seed}"
  fi
  if [ "${RESUME_SUITE}" = "1" ] && is_run_completed "${BLK_RUN}"; then
    echo "[SKIP] completed run: ${BLK_RUN}"
  else
    run_supervisor "${BLK_RUN}" "blocking" "0" "${seed}"
  fi
  if [ "${RESUME_SUITE}" = "1" ] && is_run_completed "${OVL_RUN}"; then
    echo "[SKIP] completed run: ${OVL_RUN}"
  else
    run_supervisor "${OVL_RUN}" "overlapped" "0" "${seed}"
  fi

  if [ "${RESUME_SUITE}" = "1" ] && has_run_metrics "${REF_RUN}"; then
    echo "[SKIP] metrics exist: ${REF_RUN}"
  else
    run_metrics_for "${REF_RUN}" "${seed}"
  fi
  if [ "${RESUME_SUITE}" = "1" ] && has_run_metrics "${BLK_RUN}"; then
    echo "[SKIP] metrics exist: ${BLK_RUN}"
  else
    run_metrics_for "${BLK_RUN}" "${seed}"
  fi
  if [ "${RESUME_SUITE}" = "1" ] && has_run_metrics "${OVL_RUN}"; then
    echo "[SKIP] metrics exist: ${OVL_RUN}"
  else
    run_metrics_for "${OVL_RUN}" "${seed}"
  fi

  if [ "${RESUME_SUITE}" = "1" ] && has_divergence_metric "${REF_RUN}" "${BLK_RUN}"; then
    echo "[SKIP] divergence exists: ${BLK_RUN} vs ${REF_RUN}"
  else
    run_divergence "${REF_RUN}" "${BLK_RUN}"
  fi
  if [ "${RESUME_SUITE}" = "1" ] && has_divergence_metric "${REF_RUN}" "${OVL_RUN}"; then
    echo "[SKIP] divergence exists: ${OVL_RUN} vs ${REF_RUN}"
  else
    run_divergence "${REF_RUN}" "${OVL_RUN}"
  fi
done

REF_CSV="$(join_csv "${REF_RUNS[@]}")"
BLK_CSV="$(join_csv "${BLK_RUNS[@]}")"
OVL_CSV="$(join_csv "${OVL_RUNS[@]}")"
ALL_CSV="$(join_csv "${REF_RUNS[@]}" "${BLK_RUNS[@]}" "${OVL_RUNS[@]}")"

REF_LABEL="${RUN_PREFIX}_reference"
BLK_LABEL="${RUN_PREFIX}_failure_blocking"
OVL_LABEL="${RUN_PREFIX}_failure_overlapped"

run_py -m ecrl.metrics.aggregate --results-dir "${RESULTS_DIR}" --run-ids "${REF_CSV}" --label "${REF_LABEL}"
run_py -m ecrl.metrics.aggregate --results-dir "${RESULTS_DIR}" --run-ids "${BLK_CSV}" --label "${BLK_LABEL}"
run_py -m ecrl.metrics.aggregate --results-dir "${RESULTS_DIR}" --run-ids "${OVL_CSV}" --label "${OVL_LABEL}"

run_py -m ecrl.metrics.plot \
  --results-dir "${RESULTS_DIR}" \
  --run-ids "${ALL_CSV}" \
  --output-prefix "${RUN_PREFIX}_paperlite"

run_py -m ecrl.metrics.report_publishable \
  --results-dir "${RESULTS_DIR}" \
  --output-prefix "${RUN_PREFIX}_publishable" \
  --reference-runs "${REF_CSV}" \
  --blocking-runs "${BLK_CSV}" \
  --overlapped-runs "${OVL_CSV}" \
  --aggregate-reference-label "${REF_LABEL}" \
  --aggregate-blocking-label "${BLK_LABEL}" \
  --aggregate-overlapped-label "${OVL_LABEL}"

echo "[DONE] Publishable markdown: ${RESULTS_DIR}/reports/${RUN_PREFIX}_publishable.md"
echo "[DONE] Publishable json: ${RESULTS_DIR}/reports/${RUN_PREFIX}_publishable.json"
