#!/usr/bin/env bash
set -euo pipefail

NPROC="${1:-4}"
RESULTS_DIR="${RESULTS_DIR:-results}"
CONFIG="configs/exp1_failure.yaml"
TARGET="1000"
FAIL_STEPS="200,600"
K="50"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"

REF_RUN="exp1_reference"
BLK_RUN="exp1_failure_blocking"
OVL_RUN="exp1_failure_overlapped"

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

python -m ecrl.orchestration.supervisor \
  --config "${CONFIG}" \
  --run-id "${REF_RUN}" \
  --results-dir "${RESULTS_DIR}" \
  --nproc-per-node "${NPROC}" \
  --target-steps "${TARGET}" \
  --checkpoint-strategy blocking \
  --checkpoint-every "${K}" \
  --disable-failure \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT_BASE}"

python -m ecrl.orchestration.supervisor \
  --config "${CONFIG}" \
  --run-id "${BLK_RUN}" \
  --results-dir "${RESULTS_DIR}" \
  --nproc-per-node "${NPROC}" \
  --target-steps "${TARGET}" \
  --checkpoint-strategy blocking \
  --checkpoint-every "${K}" \
  --fail-steps "${FAIL_STEPS}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "$((MASTER_PORT_BASE + 1))"

python -m ecrl.orchestration.supervisor \
  --config "${CONFIG}" \
  --run-id "${OVL_RUN}" \
  --results-dir "${RESULTS_DIR}" \
  --nproc-per-node "${NPROC}" \
  --target-steps "${TARGET}" \
  --checkpoint-strategy overlapped \
  --checkpoint-every "${K}" \
  --max-inflight 4 \
  --fail-steps "${FAIL_STEPS}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "$((MASTER_PORT_BASE + 2))"

for RUN in "${REF_RUN}" "${BLK_RUN}" "${OVL_RUN}"; do
  python -m ecrl.metrics.correctness --config "${CONFIG}" --run-id "${RUN}" --results-dir "${RESULTS_DIR}" --target-steps "${TARGET}"
  python -m ecrl.metrics.goodput --run-id "${RUN}" --results-dir "${RESULTS_DIR}" --target-steps "${TARGET}"
done

python -m ecrl.metrics.divergence --reference-run "${REF_RUN}" --candidate-run "${BLK_RUN}" --results-dir "${RESULTS_DIR}" --steps "200,400,800"
python -m ecrl.metrics.divergence --reference-run "${REF_RUN}" --candidate-run "${OVL_RUN}" --results-dir "${RESULTS_DIR}" --steps "200,400,800"

python -m ecrl.metrics.plot --results-dir "${RESULTS_DIR}" --run-ids "${REF_RUN},${BLK_RUN},${OVL_RUN}" --output-prefix exp1
