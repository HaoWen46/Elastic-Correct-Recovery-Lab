#!/usr/bin/env bash
set -euo pipefail

# Comprehensive ECRL study runner:
# - Baseline multi-seed recovery runs (reference, blocking-failure, overlapped-failure)
# - Checkpoint-frequency sweep
# - Overlapped inflight-depth sweep
# - Failure-schedule sweep
# - Elastic world-size matrix (N->M)
# - Aggregates, plots, and a summary report
#
# Usage:
#   scripts/run_exp4_comprehensive_study.sh [NPROC_MAIN]
#
# Example:
#   IMAGEFOLDER_ROOT=/mnt/imagenet100 \
#   PROFILE=imagefolder \
#   MODEL_NAME=resnet50 \
#   scripts/run_exp4_comprehensive_study.sh 4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NPROC_MAIN="${1:-4}"

STUDY_NAME="${STUDY_NAME:-exp4_comprehensive}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_PREFIX="${RUN_PREFIX:-${STUDY_NAME}_${RUN_STAMP}}"
RESULTS_DIR="${RESULTS_DIR:-results/${RUN_PREFIX}}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
VENV_PYTHON="${VENV_PYTHON:-}"
SETUP_ENV="${SETUP_ENV:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-$((30000 + RANDOM % 20000))}"
NEXT_PORT="${MASTER_PORT_BASE}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
MAX_RESTARTS_NO_CKPT="${MAX_RESTARTS_NO_CKPT:-3}"
RESTART_DELAY_SEC="${RESTART_DELAY_SEC:-1.0}"

TARGET_STEPS="${TARGET_STEPS:-1600}"
PHASE_A_TARGET="${PHASE_A_TARGET:-800}"
DIVERGENCE_STEPS="${DIVERGENCE_STEPS:-400,800,1200,1600}"

SEEDS_CSV="${SEEDS_CSV:-1337,2027,4242}"
BASE_CHECKPOINT_EVERY="${BASE_CHECKPOINT_EVERY:-50}"
BASE_FAIL_STEPS="${BASE_FAIL_STEPS:-400,1200}"
MAX_INFLIGHT_BASE="${MAX_INFLIGHT_BASE:-4}"

K_SWEEP="${K_SWEEP:-25,50,100}"
MAX_INFLIGHT_SWEEP="${MAX_INFLIGHT_SWEEP:-1,2,4,8}"
FAILURE_SCHEDULES="${FAILURE_SCHEDULES:-early:200,600;mid:400,1200;late:800,1400}"

PROFILE="${PROFILE:-auto}"  # auto|cifar100|imagefolder
IMAGEFOLDER_ROOT="${IMAGEFOLDER_ROOT:-}"
IMAGEFOLDER_SPLIT_SUBDIR="${IMAGEFOLDER_SPLIT_SUBDIR:-train}"

MODEL_NAME="${MODEL_NAME:-resnet50}"
PRECISION="${PRECISION:-bf16}"
USE_AUGMENTATION="${USE_AUGMENTATION:-true}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
LR="${LR:-}"
GLOBAL_BATCH="${GLOBAL_BATCH:-}"
IMAGE_SIZE="${IMAGE_SIZE:-}"

LOG_EVERY="${LOG_EVERY:-1}"
DEBUG_EVERY="${DEBUG_EVERY:-100}"

DEFAULT_HALF="$((NPROC_MAIN / 2))"
if [ "${DEFAULT_HALF}" -lt 1 ]; then
  DEFAULT_HALF=1
fi
if [ "${DEFAULT_HALF}" -eq "${NPROC_MAIN}" ]; then
  ELASTIC_PAIRS_DEFAULT="${NPROC_MAIN}->1"
else
  ELASTIC_PAIRS_DEFAULT="${NPROC_MAIN}->${DEFAULT_HALF},${NPROC_MAIN}->1"
fi
ELASTIC_PAIRS="${ELASTIC_PAIRS:-${ELASTIC_PAIRS_DEFAULT}}"

trim() {
  local x="$1"
  x="${x#"${x%%[![:space:]]*}"}"
  x="${x%"${x##*[![:space:]]}"}"
  printf '%s' "${x}"
}

run_py() {
  uv run --python "${PYTHON_BIN}" "$@"
}

probe_cuda_count() {
  run_py - <<'PY'
import torch
if torch.cuda.is_available():
    print(int(torch.cuda.device_count()))
else:
    print(0)
PY
}

probe_torch_cuda_info() {
  run_py - <<'PY'
import torch
print(
    f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
    f"cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0} "
    f"torch_cuda={torch.version.cuda}"
)
PY
}

next_port() {
  local p="${NEXT_PORT}"
  NEXT_PORT=$((NEXT_PORT + 1))
  printf '%s' "${p}"
}

join_csv() {
  local IFS=','
  echo "$*"
}

require_divisible() {
  local batch="$1"
  local ws="$2"
  if ! [[ "${batch}" =~ ^[0-9]+$ && "${ws}" =~ ^[0-9]+$ ]]; then
    echo "require_divisible expects integer values: batch=${batch}, world_size=${ws}" >&2
    exit 1
  fi
  if [ "${ws}" -le 0 ]; then
    echo "Invalid world size: ${ws}" >&2
    exit 1
  fi
  if [ $((batch % ws)) -ne 0 ]; then
    echo "GLOBAL_BATCH (${batch}) must be divisible by world size (${ws})" >&2
    exit 1
  fi
}

ensure_env() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install uv first: https://docs.astral.sh/uv/" >&2
    exit 1
  fi

  if [ ! -x "${PYTHON_BIN}" ]; then
    if [ "${PYTHON_BIN}" = ".venv/bin/python" ]; then
      local venv_python="${VENV_PYTHON}"
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
from ecrl.statepack import rng

src = inspect.getsource(rng.restore_rng_state)
if "_as_cpu_byte_tensor" not in src:
    print("missing_rng_patch")
    sys.exit(1)
print("ok")
PY
  )" || {
    echo "Runtime patch check failed: ecrl.statepack.rng.restore_rng_state is missing ByteTensor normalization." >&2
    echo "Sync latest code to this workstation before running Exp4." >&2
    exit 1
  }
  echo "[INFO] Runtime patch check: ${patch_status}"
}

enforce_cuda_guard() {
  TORCH_CUDA_INFO="$(probe_torch_cuda_info)"
  CUDA_DEVICE_COUNT="$(probe_cuda_count)"
  echo "[INFO] ${TORCH_CUDA_INFO}"

  if [ "${REQUIRE_CUDA}" = "1" ]; then
    if [ "${CUDA_DEVICE_COUNT}" -lt 1 ]; then
      echo "REQUIRE_CUDA=1 but no CUDA devices are visible to PyTorch." >&2
      exit 1
    fi
    if [ "${NPROC_MAIN}" -gt "${CUDA_DEVICE_COUNT}" ]; then
      echo "REQUIRE_CUDA=1 but NPROC_MAIN=${NPROC_MAIN} exceeds CUDA devices=${CUDA_DEVICE_COUNT}." >&2
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

prepare_profile() {
  local profile="${PROFILE}"

  if [ "${profile}" = "auto" ]; then
    if [ -n "${IMAGEFOLDER_ROOT}" ] && [ -d "${IMAGEFOLDER_ROOT}/${IMAGEFOLDER_SPLIT_SUBDIR}" ]; then
      profile="imagefolder"
    else
      profile="cifar100"
    fi
  fi

  case "${profile}" in
    cifar100)
      DATASET_NAME="cifar100"
      DATASET_ROOT="./data"
      DATASET_DOWNLOAD="true"
      DATASET_SIZE="50000"
      IMAGE_SIZE="${IMAGE_SIZE:-32}"
      GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
      LR="${LR:-0.1}"
      ensure_cifar100
      ;;
    imagefolder)
      if [ -z "${IMAGEFOLDER_ROOT}" ]; then
        echo "PROFILE=imagefolder requires IMAGEFOLDER_ROOT" >&2
        exit 1
      fi
      read -r DATASET_SIZE DATASET_CLASSES < <(count_imagefolder_samples "${IMAGEFOLDER_ROOT}" "${IMAGEFOLDER_SPLIT_SUBDIR}")
      DATASET_NAME="imagefolder"
      DATASET_ROOT="${IMAGEFOLDER_ROOT}"
      DATASET_DOWNLOAD="false"
      IMAGE_SIZE="${IMAGE_SIZE:-224}"
      GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
      LR="${LR:-0.05}"
      ;;
    *)
      echo "Unsupported PROFILE=${profile}. Use auto|cifar100|imagefolder." >&2
      exit 1
      ;;
  esac

  PROFILE="${profile}"
}

generate_config() {
  CONFIG_PATH="${RESULTS_DIR}/configs/${RUN_PREFIX}.yaml"
  mkdir -p "$(dirname "${CONFIG_PATH}")"

  cat > "${CONFIG_PATH}" <<EOF
seed: 1337
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
  every_k_steps: ${BASE_CHECKPOINT_EVERY}
  max_inflight: ${MAX_INFLIGHT_BASE}
failure:
  enabled: true
  steps: [${BASE_FAIL_STEPS}]
EOF
}

run_supervisor() {
  local run_id="$1"
  local nproc="$2"
  local target="$3"
  local strategy="$4"
  local checkpoint_every="$5"
  local seed="$6"
  local fail_steps="$7"
  local max_inflight="$8"
  local start_resume_latest="$9"
  local disable_failure="${10}"
  local port
  port="$(next_port)"

  if [ "${REQUIRE_CUDA}" = "1" ] && [ "${nproc}" -gt "${CUDA_DEVICE_COUNT}" ]; then
    echo "Run ${run_id} requires nproc=${nproc}, but only ${CUDA_DEVICE_COUNT} CUDA devices are visible." >&2
    exit 1
  fi

  local cmd=(
    -m ecrl.orchestration.supervisor
    --config "${CONFIG_PATH}"
    --run-id "${run_id}"
    --results-dir "${RESULTS_DIR}"
    --nproc-per-node "${nproc}"
    --target-steps "${target}"
    --checkpoint-strategy "${strategy}"
    --checkpoint-every "${checkpoint_every}"
    --seed "${seed}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${port}"
    --max-restarts "${MAX_RESTARTS}"
    --max-restarts-without-checkpoint "${MAX_RESTARTS_NO_CKPT}"
    --restart-delay-sec "${RESTART_DELAY_SEC}"
  )

  if [ "${strategy}" = "overlapped" ]; then
    cmd+=(--max-inflight "${max_inflight}")
  fi
  if [ -n "${fail_steps}" ]; then
    cmd+=(--fail-steps "${fail_steps}")
  fi
  if [ "${disable_failure}" = "1" ]; then
    cmd+=(--disable-failure)
  fi
  if [ "${start_resume_latest}" = "1" ]; then
    cmd+=(--start-resume-latest)
  fi

  echo "[RUN] ${run_id} nproc=${nproc} target=${target} strategy=${strategy} ckpt=${checkpoint_every} fail=${fail_steps:-none} inflight=${max_inflight} resume=${start_resume_latest}"
  run_py "${cmd[@]}"
}

run_metrics_for() {
  local run_id="$1"
  local target="$2"
  local seed="$3"
  run_py -m ecrl.metrics.correctness \
    --config "${CONFIG_PATH}" \
    --run-id "${run_id}" \
    --results-dir "${RESULTS_DIR}" \
    --target-steps "${target}" \
    --seed "${seed}"
  run_py -m ecrl.metrics.goodput \
    --run-id "${run_id}" \
    --results-dir "${RESULTS_DIR}" \
    --target-steps "${target}"
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

aggregate_group() {
  local label="$1"
  shift
  if [ "$#" -eq 0 ]; then
    return
  fi
  local csv
  csv="$(join_csv "$@")"
  run_py -m ecrl.metrics.aggregate \
    --results-dir "${RESULTS_DIR}" \
    --run-ids "${csv}" \
    --label "${label}"
}

plot_group() {
  local prefix="$1"
  shift
  if [ "$#" -eq 0 ]; then
    return
  fi
  local csv
  csv="$(join_csv "$@")"
  run_py -m ecrl.metrics.plot \
    --results-dir "${RESULTS_DIR}" \
    --run-ids "${csv}" \
    --output-prefix "${prefix}"
}

build_summary() {
  local summary_md="${RESULTS_DIR}/reports/${RUN_PREFIX}_summary.md"
  local summary_json="${RESULTS_DIR}/reports/${RUN_PREFIX}_summary.json"
  mkdir -p "${RESULTS_DIR}/reports"

  run_py - "${RESULTS_DIR}" "${RUN_PREFIX}" "${CONFIG_PATH}" "${summary_md}" "${summary_json}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

results_dir = Path(sys.argv[1])
run_prefix = sys.argv[2]
config_path = Path(sys.argv[3])
summary_md = Path(sys.argv[4])
summary_json = Path(sys.argv[5])

cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
agg_dir = results_dir / "metrics" / "_aggregate"
agg_paths = sorted(agg_dir.glob(f"{run_prefix}_*.json"))

aggregates = []
for p in agg_paths:
    data = json.loads(p.read_text(encoding="utf-8"))
    aggregates.append(
        {
            "label": data["label"],
            "num_runs": int(data["num_runs"]),
            "pass_rate": float(data["correctness"]["pass_rate"]),
            "all_passed": bool(data["correctness"]["all_passed"]),
            "goodput_mean": float(data["goodput_steps_per_sec"]["mean"]),
            "goodput_ci95": float(data["goodput_steps_per_sec"]["ci95"]),
            "restarts_mean": float(data["restarts"]["mean"]),
            "replayed_mean": float(data["replayed_steps"]["mean"]),
            "stall_mean": float(data["stall_time_sec_total"]["mean"]),
        }
    )

divergence_rows = []
metrics_root = results_dir / "metrics"
for run_dir in sorted(metrics_root.iterdir()) if metrics_root.exists() else []:
    if not run_dir.is_dir() or run_dir.name == "_aggregate":
        continue
    if not run_dir.name.startswith(run_prefix):
        continue
    for p in run_dir.glob("divergence_vs_*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        loss = data.get("loss_divergence", {})
        divergence_rows.append(
            {
                "candidate_run": data.get("candidate_run"),
                "reference_run": data.get("reference_run"),
                "num_common_steps": int(loss.get("num_common_steps") or 0),
                "max_abs_diff": loss.get("max_abs_diff"),
                "mean_abs_diff": loss.get("mean_abs_diff"),
                "auc_abs_diff": loss.get("auc_abs_diff"),
            }
        )

report = {
    "run_prefix": run_prefix,
    "results_dir": str(results_dir),
    "config_path": str(config_path),
    "dataset": cfg.get("dataset", {}),
    "training": cfg.get("training", {}),
    "aggregates": aggregates,
    "divergence": divergence_rows,
}
summary_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

lines = []
lines.append(f"# {run_prefix} Comprehensive Study Summary")
lines.append("")
lines.append(f"- Results root: `{results_dir}`")
lines.append(f"- Config: `{config_path}`")
lines.append(f"- Dataset: `{cfg.get('dataset', {}).get('name')}`")
lines.append(f"- Model: `{cfg.get('training', {}).get('model_name')}`")
lines.append(f"- Global batch: `{cfg.get('training', {}).get('global_batch')}`")
lines.append("")
lines.append("## Aggregate Groups")
lines.append("")
lines.append("| Label | Runs | Pass Rate | Goodput (mean ± CI95) | Restarts (mean) | Replayed (mean) | Stall sec (mean) |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for row in aggregates:
    lines.append(
        "| {label} | {num_runs} | {pass_rate:.2f} | {goodput_mean:.4f} ± {goodput_ci95:.4f} | {restarts_mean:.2f} | {replayed_mean:.2f} | {stall_mean:.4f} |".format(
            **row
        )
    )

lines.append("")
lines.append("## Divergence Pairs")
lines.append("")
if divergence_rows:
    lines.append("| Candidate | Reference | Common Steps | Max Abs Loss Diff | Mean Abs Loss Diff | AUC Abs Diff |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in divergence_rows:
        def fmt(v):
            if v is None:
                return "NA"
            return f"{float(v):.6f}"
        lines.append(
            f"| {row['candidate_run']} | {row['reference_run']} | {row['num_common_steps']} | {fmt(row['max_abs_diff'])} | {fmt(row['mean_abs_diff'])} | {fmt(row['auc_abs_diff'])} |"
        )
else:
    lines.append("No divergence files found.")

lines.append("")
lines.append("## Notes")
lines.append("")
lines.append("- Correctness pass is a hard gate for interpreting performance metrics.")
lines.append("- Compare checkpoint strategy groups at equal failure schedules/checkpoint cadence.")
lines.append("- Elastic groups compare fixed-N reference against N->M resume with constant global batch.")

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md)
print(summary_json)
PY
}

echo "[INFO] Study prefix: ${RUN_PREFIX}"
echo "[INFO] Results dir: ${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

ensure_env
verify_runtime_patches
enforce_cuda_guard
prepare_profile

if [ "${PHASE_A_TARGET}" -ge "${TARGET_STEPS}" ]; then
  echo "PHASE_A_TARGET (${PHASE_A_TARGET}) must be less than TARGET_STEPS (${TARGET_STEPS})" >&2
  exit 1
fi

require_divisible "${GLOBAL_BATCH}" "${NPROC_MAIN}"

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

REF_SEED="${SEEDS[0]}"

generate_config

echo "[INFO] Profile: ${PROFILE}"
echo "[INFO] Dataset: ${DATASET_NAME} (size=${DATASET_SIZE})"
echo "[INFO] Model: ${MODEL_NAME}"
echo "[INFO] Global batch: ${GLOBAL_BATCH}"
echo "[INFO] Config: ${CONFIG_PATH}"

declare -a BASE_REF_RUNS=()
declare -a BASE_BLK_RUNS=()
declare -a BASE_OVL_RUNS=()
declare -a K_BLK_RUNS=()
declare -a K_OVL_RUNS=()
declare -a INFLIGHT_OVL_RUNS=()
declare -a FAIL_BLK_RUNS=()
declare -a FAIL_OVL_RUNS=()
declare -a ELASTIC_REF_RUNS=()
declare -a ELASTIC_RUNS=()

# 1) Baseline multi-seed matrix.
for seed in "${SEEDS[@]}"; do
  REF_RUN="${RUN_PREFIX}_baseline_s${seed}_reference"
  BLK_RUN="${RUN_PREFIX}_baseline_s${seed}_failure_blocking"
  OVL_RUN="${RUN_PREFIX}_baseline_s${seed}_failure_overlapped"

  BASE_REF_RUNS+=("${REF_RUN}")
  BASE_BLK_RUNS+=("${BLK_RUN}")
  BASE_OVL_RUNS+=("${OVL_RUN}")

  run_supervisor "${REF_RUN}" "${NPROC_MAIN}" "${TARGET_STEPS}" "blocking" "${BASE_CHECKPOINT_EVERY}" "${seed}" "" "${MAX_INFLIGHT_BASE}" "0" "1"
  run_supervisor "${BLK_RUN}" "${NPROC_MAIN}" "${TARGET_STEPS}" "blocking" "${BASE_CHECKPOINT_EVERY}" "${seed}" "${BASE_FAIL_STEPS}" "${MAX_INFLIGHT_BASE}" "0" "0"
  run_supervisor "${OVL_RUN}" "${NPROC_MAIN}" "${TARGET_STEPS}" "overlapped" "${BASE_CHECKPOINT_EVERY}" "${seed}" "${BASE_FAIL_STEPS}" "${MAX_INFLIGHT_BASE}" "0" "0"

  run_metrics_for "${REF_RUN}" "${TARGET_STEPS}" "${seed}"
  run_metrics_for "${BLK_RUN}" "${TARGET_STEPS}" "${seed}"
  run_metrics_for "${OVL_RUN}" "${TARGET_STEPS}" "${seed}"

  run_divergence "${REF_RUN}" "${BLK_RUN}"
  run_divergence "${REF_RUN}" "${OVL_RUN}"
done

BASE_REF_MAIN="${RUN_PREFIX}_baseline_s${REF_SEED}_reference"

# 2) Checkpoint frequency sweep.
IFS=',' read -r -a K_SWEEP_RAW <<< "${K_SWEEP}"
for k in "${K_SWEEP_RAW[@]}"; do
  k="$(trim "${k}")"
  [ -n "${k}" ] || continue
  BLK_RUN="${RUN_PREFIX}_ksweep_k${k}_blocking"
  OVL_RUN="${RUN_PREFIX}_ksweep_k${k}_overlapped"
  K_BLK_RUNS+=("${BLK_RUN}")
  K_OVL_RUNS+=("${OVL_RUN}")

  run_supervisor "${BLK_RUN}" "${NPROC_MAIN}" "${TARGET_STEPS}" "blocking" "${k}" "${REF_SEED}" "${BASE_FAIL_STEPS}" "${MAX_INFLIGHT_BASE}" "0" "0"
  run_supervisor "${OVL_RUN}" "${NPROC_MAIN}" "${TARGET_STEPS}" "overlapped" "${k}" "${REF_SEED}" "${BASE_FAIL_STEPS}" "${MAX_INFLIGHT_BASE}" "0" "0"

  run_metrics_for "${BLK_RUN}" "${TARGET_STEPS}" "${REF_SEED}"
  run_metrics_for "${OVL_RUN}" "${TARGET_STEPS}" "${REF_SEED}"
  run_divergence "${BASE_REF_MAIN}" "${BLK_RUN}"
  run_divergence "${BASE_REF_MAIN}" "${OVL_RUN}"
done

# 3) Overlapped inflight sweep.
IFS=',' read -r -a INF_RAW <<< "${MAX_INFLIGHT_SWEEP}"
for inf in "${INF_RAW[@]}"; do
  inf="$(trim "${inf}")"
  [ -n "${inf}" ] || continue
  OVL_RUN="${RUN_PREFIX}_inflight_m${inf}_overlapped"
  INFLIGHT_OVL_RUNS+=("${OVL_RUN}")

  run_supervisor "${OVL_RUN}" "${NPROC_MAIN}" "${TARGET_STEPS}" "overlapped" "${BASE_CHECKPOINT_EVERY}" "${REF_SEED}" "${BASE_FAIL_STEPS}" "${inf}" "0" "0"
  run_metrics_for "${OVL_RUN}" "${TARGET_STEPS}" "${REF_SEED}"
  run_divergence "${BASE_REF_MAIN}" "${OVL_RUN}"
done

# 4) Failure schedule sweep.
IFS=';' read -r -a FAIL_SPECS <<< "${FAILURE_SCHEDULES}"
for spec in "${FAIL_SPECS[@]}"; do
  spec="$(trim "${spec}")"
  [ -n "${spec}" ] || continue

  if [[ "${spec}" != *:* ]]; then
    echo "Invalid FAILURE_SCHEDULES entry: ${spec} (expected label:step,step)" >&2
    exit 1
  fi
  label="$(trim "${spec%%:*}")"
  steps="$(trim "${spec#*:}")"
  safe_label="${label//[^a-zA-Z0-9_]/_}"

  BLK_RUN="${RUN_PREFIX}_failsweep_${safe_label}_blocking"
  OVL_RUN="${RUN_PREFIX}_failsweep_${safe_label}_overlapped"
  FAIL_BLK_RUNS+=("${BLK_RUN}")
  FAIL_OVL_RUNS+=("${OVL_RUN}")

  run_supervisor "${BLK_RUN}" "${NPROC_MAIN}" "${TARGET_STEPS}" "blocking" "${BASE_CHECKPOINT_EVERY}" "${REF_SEED}" "${steps}" "${MAX_INFLIGHT_BASE}" "0" "0"
  run_supervisor "${OVL_RUN}" "${NPROC_MAIN}" "${TARGET_STEPS}" "overlapped" "${BASE_CHECKPOINT_EVERY}" "${REF_SEED}" "${steps}" "${MAX_INFLIGHT_BASE}" "0" "0"

  run_metrics_for "${BLK_RUN}" "${TARGET_STEPS}" "${REF_SEED}"
  run_metrics_for "${OVL_RUN}" "${TARGET_STEPS}" "${REF_SEED}"
  run_divergence "${BASE_REF_MAIN}" "${BLK_RUN}"
  run_divergence "${BASE_REF_MAIN}" "${OVL_RUN}"
done

# 5) Elastic N->M matrix.
IFS=',' read -r -a ELASTIC_RAW <<< "${ELASTIC_PAIRS}"
for pair in "${ELASTIC_RAW[@]}"; do
  pair="$(trim "${pair}")"
  [ -n "${pair}" ] || continue
  if [[ "${pair}" != *"->"* ]]; then
    echo "Invalid ELASTIC_PAIRS entry: ${pair} (expected N->M)" >&2
    exit 1
  fi
  n="$(trim "${pair%%->*}")"
  m="$(trim "${pair##*->}")"
  if [ -z "${n}" ] || [ -z "${m}" ]; then
    echo "Invalid ELASTIC_PAIRS entry: ${pair}" >&2
    exit 1
  fi
  if [ "${n}" = "${m}" ]; then
    echo "[WARN] Skipping elastic pair ${n}->${m} (no world-size change)"
    continue
  fi

  require_divisible "${GLOBAL_BATCH}" "${n}"
  require_divisible "${GLOBAL_BATCH}" "${m}"

  REF_RUN="${RUN_PREFIX}_elastic_n${n}_reference"
  ELS_RUN="${RUN_PREFIX}_elastic_n${n}_to_m${m}"
  ELASTIC_REF_RUNS+=("${REF_RUN}")
  ELASTIC_RUNS+=("${ELS_RUN}")

  run_supervisor "${REF_RUN}" "${n}" "${TARGET_STEPS}" "blocking" "${BASE_CHECKPOINT_EVERY}" "${REF_SEED}" "" "${MAX_INFLIGHT_BASE}" "0" "1"

  run_supervisor "${ELS_RUN}" "${n}" "${PHASE_A_TARGET}" "blocking" "${BASE_CHECKPOINT_EVERY}" "${REF_SEED}" "" "${MAX_INFLIGHT_BASE}" "0" "1"
  run_supervisor "${ELS_RUN}" "${m}" "${TARGET_STEPS}" "blocking" "${BASE_CHECKPOINT_EVERY}" "${REF_SEED}" "" "${MAX_INFLIGHT_BASE}" "1" "1"

  run_metrics_for "${REF_RUN}" "${TARGET_STEPS}" "${REF_SEED}"
  run_metrics_for "${ELS_RUN}" "${TARGET_STEPS}" "${REF_SEED}"
  run_divergence "${REF_RUN}" "${ELS_RUN}"
done

# Aggregate groups.
BASE_REF_LABEL="${RUN_PREFIX}_baseline_reference"
BASE_BLK_LABEL="${RUN_PREFIX}_baseline_failure_blocking"
BASE_OVL_LABEL="${RUN_PREFIX}_baseline_failure_overlapped"
K_BLK_LABEL="${RUN_PREFIX}_ksweep_blocking"
K_OVL_LABEL="${RUN_PREFIX}_ksweep_overlapped"
INF_OVL_LABEL="${RUN_PREFIX}_inflight_overlapped"
FAIL_BLK_LABEL="${RUN_PREFIX}_failsweep_blocking"
FAIL_OVL_LABEL="${RUN_PREFIX}_failsweep_overlapped"
ELS_REF_LABEL="${RUN_PREFIX}_elastic_reference"
ELS_RUN_LABEL="${RUN_PREFIX}_elastic_resume"

aggregate_group "${BASE_REF_LABEL}" "${BASE_REF_RUNS[@]}"
aggregate_group "${BASE_BLK_LABEL}" "${BASE_BLK_RUNS[@]}"
aggregate_group "${BASE_OVL_LABEL}" "${BASE_OVL_RUNS[@]}"
aggregate_group "${K_BLK_LABEL}" "${K_BLK_RUNS[@]}"
aggregate_group "${K_OVL_LABEL}" "${K_OVL_RUNS[@]}"
aggregate_group "${INF_OVL_LABEL}" "${INFLIGHT_OVL_RUNS[@]}"
aggregate_group "${FAIL_BLK_LABEL}" "${FAIL_BLK_RUNS[@]}"
aggregate_group "${FAIL_OVL_LABEL}" "${FAIL_OVL_RUNS[@]}"
aggregate_group "${ELS_REF_LABEL}" "${ELASTIC_REF_RUNS[@]}"
aggregate_group "${ELS_RUN_LABEL}" "${ELASTIC_RUNS[@]}"

# Group-specific plots (avoid unreadable mega-plot).
plot_group "${RUN_PREFIX}_baseline" "${BASE_REF_RUNS[@]}" "${BASE_BLK_RUNS[@]}" "${BASE_OVL_RUNS[@]}"
plot_group "${RUN_PREFIX}_ksweep" "${K_BLK_RUNS[@]}" "${K_OVL_RUNS[@]}"
plot_group "${RUN_PREFIX}_inflight" "${INFLIGHT_OVL_RUNS[@]}"
plot_group "${RUN_PREFIX}_failsweep" "${FAIL_BLK_RUNS[@]}" "${FAIL_OVL_RUNS[@]}"
plot_group "${RUN_PREFIX}_elastic" "${ELASTIC_REF_RUNS[@]}" "${ELASTIC_RUNS[@]}"

build_summary

echo "[DONE] Comprehensive study completed."
echo "[DONE] Results: ${RESULTS_DIR}"
echo "[DONE] Config:  ${CONFIG_PATH}"
echo "[DONE] Summary: ${RESULTS_DIR}/reports/${RUN_PREFIX}_summary.md"
