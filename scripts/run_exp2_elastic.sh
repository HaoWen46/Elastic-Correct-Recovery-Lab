#!/usr/bin/env bash
set -euo pipefail

NPROC_A="${1:-4}"
NPROC_B="${2:-2}"
RESULTS_DIR="${RESULTS_DIR:-results}"
CONFIG="configs/exp2_elastic.yaml"
K="50"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29600}"

REF_RUN="exp2_reference_n4"
ELS_RUN="exp2_elastic_4to2"

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
  --nproc-per-node "${NPROC_A}" \
  --target-steps 800 \
  --checkpoint-strategy blocking \
  --checkpoint-every "${K}" \
  --disable-failure \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT_BASE}"

python -m ecrl.orchestration.supervisor \
  --config "${CONFIG}" \
  --run-id "${ELS_RUN}" \
  --results-dir "${RESULTS_DIR}" \
  --nproc-per-node "${NPROC_A}" \
  --target-steps 400 \
  --checkpoint-strategy blocking \
  --checkpoint-every "${K}" \
  --disable-failure \
  --master-addr "${MASTER_ADDR}" \
  --master-port "$((MASTER_PORT_BASE + 1))"

python -m ecrl.orchestration.supervisor \
  --config "${CONFIG}" \
  --run-id "${ELS_RUN}" \
  --results-dir "${RESULTS_DIR}" \
  --nproc-per-node "${NPROC_B}" \
  --target-steps 800 \
  --checkpoint-strategy blocking \
  --checkpoint-every "${K}" \
  --disable-failure \
  --start-resume-latest \
  --master-addr "${MASTER_ADDR}" \
  --master-port "$((MASTER_PORT_BASE + 2))"

python -m ecrl.metrics.correctness --config "${CONFIG}" --run-id "${REF_RUN}" --results-dir "${RESULTS_DIR}" --target-steps 800
python -m ecrl.metrics.correctness --config "${CONFIG}" --run-id "${ELS_RUN}" --results-dir "${RESULTS_DIR}" --target-steps 800

python -m ecrl.metrics.goodput --run-id "${REF_RUN}" --results-dir "${RESULTS_DIR}" --target-steps 800
python -m ecrl.metrics.goodput --run-id "${ELS_RUN}" --results-dir "${RESULTS_DIR}" --target-steps 800

python -m ecrl.metrics.divergence --reference-run "${REF_RUN}" --candidate-run "${ELS_RUN}" --results-dir "${RESULTS_DIR}" --steps "200,400,800"
python -m ecrl.metrics.plot --results-dir "${RESULTS_DIR}" --run-ids "${REF_RUN},${ELS_RUN}" --output-prefix exp2
