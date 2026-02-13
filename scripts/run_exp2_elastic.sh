#!/usr/bin/env bash
set -euo pipefail

NPROC_A="${1:-4}"
NPROC_B="${2:-2}"
RESULTS_DIR="${RESULTS_DIR:-results}"
CONFIG="${CONFIG:-configs/exp2_elastic.yaml}"
K="${K:-50}"
PHASE_A_TARGET="${PHASE_A_TARGET:-400}"
TARGET_FINAL="${TARGET_FINAL:-800}"
DIVERGENCE_STEPS="${DIVERGENCE_STEPS:-200,400,800}"
SEED="${SEED:-}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29600}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

run_py() {
  uv run --python "${PYTHON_BIN}" "$@"
}

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Missing Python env at ${PYTHON_BIN}. Run: uv venv --python 3.11.2 .venv && uv pip install -r requirements.txt" >&2
  exit 1
fi

REF_RUN="exp2_reference_n4"
ELS_RUN="exp2_elastic_4to2"

SEED_ARGS=()
CORRECTNESS_SEED_ARGS=()
if [ -n "${SEED}" ]; then
  SEED_ARGS=(--seed "${SEED}")
  CORRECTNESS_SEED_ARGS=(--seed "${SEED}")
fi

ensure_cifar() {
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

ensure_cifar

run_py -m ecrl.orchestration.supervisor \
  --config "${CONFIG}" \
  --run-id "${REF_RUN}" \
  --results-dir "${RESULTS_DIR}" \
  --nproc-per-node "${NPROC_A}" \
  --target-steps "${TARGET_FINAL}" \
  --checkpoint-strategy blocking \
  --checkpoint-every "${K}" \
  "${SEED_ARGS[@]}" \
  --disable-failure \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT_BASE}"

run_py -m ecrl.orchestration.supervisor \
  --config "${CONFIG}" \
  --run-id "${ELS_RUN}" \
  --results-dir "${RESULTS_DIR}" \
  --nproc-per-node "${NPROC_A}" \
  --target-steps "${PHASE_A_TARGET}" \
  --checkpoint-strategy blocking \
  --checkpoint-every "${K}" \
  "${SEED_ARGS[@]}" \
  --disable-failure \
  --master-addr "${MASTER_ADDR}" \
  --master-port "$((MASTER_PORT_BASE + 1))"

run_py -m ecrl.orchestration.supervisor \
  --config "${CONFIG}" \
  --run-id "${ELS_RUN}" \
  --results-dir "${RESULTS_DIR}" \
  --nproc-per-node "${NPROC_B}" \
  --target-steps "${TARGET_FINAL}" \
  --checkpoint-strategy blocking \
  --checkpoint-every "${K}" \
  "${SEED_ARGS[@]}" \
  --disable-failure \
  --start-resume-latest \
  --master-addr "${MASTER_ADDR}" \
  --master-port "$((MASTER_PORT_BASE + 2))"

run_py -m ecrl.metrics.correctness \
  --config "${CONFIG}" \
  --run-id "${REF_RUN}" \
  --results-dir "${RESULTS_DIR}" \
  --target-steps "${TARGET_FINAL}" \
  "${CORRECTNESS_SEED_ARGS[@]}"
run_py -m ecrl.metrics.correctness \
  --config "${CONFIG}" \
  --run-id "${ELS_RUN}" \
  --results-dir "${RESULTS_DIR}" \
  --target-steps "${TARGET_FINAL}" \
  "${CORRECTNESS_SEED_ARGS[@]}"

run_py -m ecrl.metrics.goodput --run-id "${REF_RUN}" --results-dir "${RESULTS_DIR}" --target-steps "${TARGET_FINAL}"
run_py -m ecrl.metrics.goodput --run-id "${ELS_RUN}" --results-dir "${RESULTS_DIR}" --target-steps "${TARGET_FINAL}"

run_py -m ecrl.metrics.divergence --reference-run "${REF_RUN}" --candidate-run "${ELS_RUN}" --results-dir "${RESULTS_DIR}" --steps "${DIVERGENCE_STEPS}"
run_py -m ecrl.metrics.plot --results-dir "${RESULTS_DIR}" --run-ids "${REF_RUN},${ELS_RUN}" --output-prefix exp2
