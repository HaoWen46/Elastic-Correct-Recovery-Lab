#!/usr/bin/env bash
set -euo pipefail

NPROC="${1:-4}"
RESULTS_DIR="${RESULTS_DIR:-results}"
CONFIG="${CONFIG:-configs/exp1_failure.yaml}"
TARGET="${TARGET:-1000}"
FAIL_STEPS="${FAIL_STEPS:-200,600}"
K="${K:-50}"
DIVERGENCE_STEPS="${DIVERGENCE_STEPS:-200,400,800}"
MAX_INFLIGHT="${MAX_INFLIGHT:-4}"
SEED="${SEED:-}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

run_py() {
  uv run --python "${PYTHON_BIN}" "$@"
}

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Missing Python env at ${PYTHON_BIN}. Run: uv venv --python 3.11.2 .venv && uv pip install -r requirements.txt" >&2
  exit 1
fi

REF_RUN="exp1_reference"
BLK_RUN="exp1_failure_blocking"
OVL_RUN="exp1_failure_overlapped"

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
  --nproc-per-node "${NPROC}" \
  --target-steps "${TARGET}" \
  --checkpoint-strategy blocking \
  --checkpoint-every "${K}" \
  "${SEED_ARGS[@]}" \
  --disable-failure \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT_BASE}"

run_py -m ecrl.orchestration.supervisor \
  --config "${CONFIG}" \
  --run-id "${BLK_RUN}" \
  --results-dir "${RESULTS_DIR}" \
  --nproc-per-node "${NPROC}" \
  --target-steps "${TARGET}" \
  --checkpoint-strategy blocking \
  --checkpoint-every "${K}" \
  "${SEED_ARGS[@]}" \
  --fail-steps "${FAIL_STEPS}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "$((MASTER_PORT_BASE + 1))"

run_py -m ecrl.orchestration.supervisor \
  --config "${CONFIG}" \
  --run-id "${OVL_RUN}" \
  --results-dir "${RESULTS_DIR}" \
  --nproc-per-node "${NPROC}" \
  --target-steps "${TARGET}" \
  --checkpoint-strategy overlapped \
  --checkpoint-every "${K}" \
  --max-inflight "${MAX_INFLIGHT}" \
  "${SEED_ARGS[@]}" \
  --fail-steps "${FAIL_STEPS}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "$((MASTER_PORT_BASE + 2))"

for RUN in "${REF_RUN}" "${BLK_RUN}" "${OVL_RUN}"; do
  run_py -m ecrl.metrics.correctness \
    --config "${CONFIG}" \
    --run-id "${RUN}" \
    --results-dir "${RESULTS_DIR}" \
    --target-steps "${TARGET}" \
    "${CORRECTNESS_SEED_ARGS[@]}"
  run_py -m ecrl.metrics.goodput --run-id "${RUN}" --results-dir "${RESULTS_DIR}" --target-steps "${TARGET}"
done

run_py -m ecrl.metrics.divergence --reference-run "${REF_RUN}" --candidate-run "${BLK_RUN}" --results-dir "${RESULTS_DIR}" --steps "${DIVERGENCE_STEPS}"
run_py -m ecrl.metrics.divergence --reference-run "${REF_RUN}" --candidate-run "${OVL_RUN}" --results-dir "${RESULTS_DIR}" --steps "${DIVERGENCE_STEPS}"

run_py -m ecrl.metrics.plot --results-dir "${RESULTS_DIR}" --run-ids "${REF_RUN},${BLK_RUN},${OVL_RUN}" --output-prefix exp1
