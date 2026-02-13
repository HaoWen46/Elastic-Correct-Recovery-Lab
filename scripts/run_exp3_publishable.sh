#!/usr/bin/env bash
set -euo pipefail

NPROC="${1:-4}"
RESULTS_DIR="${RESULTS_DIR:-results}"
CONFIG="${CONFIG:-configs/exp3_publishable.yaml}"
TARGET="${TARGET:-1200}"
FAIL_STEPS="${FAIL_STEPS:-300,900}"
SEEDS_CSV="${SEEDS_CSV:-1337,2027,4242}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-32000}"
DIVERGENCE_STEPS="${DIVERGENCE_STEPS:-300,600,1200}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
MAX_INFLIGHT="${MAX_INFLIGHT:-4}"

run_py() {
  uv run --python "${PYTHON_BIN}" "$@"
}

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Missing Python env at ${PYTHON_BIN}. Run: uv venv --python 3.11.2 .venv && uv pip install -r requirements.txt" >&2
  exit 1
fi

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

DATASET_NAME="$(
  run_py - "${CONFIG}" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(str(cfg.get("dataset", {}).get("name", "cifar100")).lower())
PY
)"

if [ "${DATASET_NAME}" = "cifar100" ]; then
  ensure_cifar100
fi

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

REF_RUNS=()
BLK_RUNS=()
OVL_RUNS=()

PORT="${MASTER_PORT_BASE}"
for seed in "${SEEDS[@]}"; do
  seed="$(echo "${seed}" | xargs)"
  [ -n "${seed}" ] || continue

  REF_RUN="exp3_s${seed}_reference"
  BLK_RUN="exp3_s${seed}_failure_blocking"
  OVL_RUN="exp3_s${seed}_failure_overlapped"

  REF_RUNS+=("${REF_RUN}")
  BLK_RUNS+=("${BLK_RUN}")
  OVL_RUNS+=("${OVL_RUN}")

  run_py -m ecrl.orchestration.supervisor \
    --config "${CONFIG}" \
    --run-id "${REF_RUN}" \
    --results-dir "${RESULTS_DIR}" \
    --nproc-per-node "${NPROC}" \
    --target-steps "${TARGET}" \
    --checkpoint-strategy blocking \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --seed "${seed}" \
    --disable-failure \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${PORT}"
  PORT=$((PORT + 1))

  run_py -m ecrl.orchestration.supervisor \
    --config "${CONFIG}" \
    --run-id "${BLK_RUN}" \
    --results-dir "${RESULTS_DIR}" \
    --nproc-per-node "${NPROC}" \
    --target-steps "${TARGET}" \
    --checkpoint-strategy blocking \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --seed "${seed}" \
    --fail-steps "${FAIL_STEPS}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${PORT}"
  PORT=$((PORT + 1))

  run_py -m ecrl.orchestration.supervisor \
    --config "${CONFIG}" \
    --run-id "${OVL_RUN}" \
    --results-dir "${RESULTS_DIR}" \
    --nproc-per-node "${NPROC}" \
    --target-steps "${TARGET}" \
    --checkpoint-strategy overlapped \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --max-inflight "${MAX_INFLIGHT}" \
    --seed "${seed}" \
    --fail-steps "${FAIL_STEPS}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${PORT}"
  PORT=$((PORT + 1))

  for RUN in "${REF_RUN}" "${BLK_RUN}" "${OVL_RUN}"; do
    run_py -m ecrl.metrics.correctness \
      --config "${CONFIG}" \
      --run-id "${RUN}" \
      --results-dir "${RESULTS_DIR}" \
      --target-steps "${TARGET}" \
      --seed "${seed}"
    run_py -m ecrl.metrics.goodput --run-id "${RUN}" --results-dir "${RESULTS_DIR}" --target-steps "${TARGET}"
  done

  run_py -m ecrl.metrics.divergence --reference-run "${REF_RUN}" --candidate-run "${BLK_RUN}" --results-dir "${RESULTS_DIR}" --steps "${DIVERGENCE_STEPS}"
  run_py -m ecrl.metrics.divergence --reference-run "${REF_RUN}" --candidate-run "${OVL_RUN}" --results-dir "${RESULTS_DIR}" --steps "${DIVERGENCE_STEPS}"
done

join_csv() {
  local IFS=','
  echo "$*"
}

REF_CSV="$(join_csv "${REF_RUNS[@]}")"
BLK_CSV="$(join_csv "${BLK_RUNS[@]}")"
OVL_CSV="$(join_csv "${OVL_RUNS[@]}")"

run_py -m ecrl.metrics.aggregate --results-dir "${RESULTS_DIR}" --run-ids "${REF_CSV}" --label "exp3_reference"
run_py -m ecrl.metrics.aggregate --results-dir "${RESULTS_DIR}" --run-ids "${BLK_CSV}" --label "exp3_failure_blocking"
run_py -m ecrl.metrics.aggregate --results-dir "${RESULTS_DIR}" --run-ids "${OVL_CSV}" --label "exp3_failure_overlapped"

run_py -m ecrl.metrics.plot --results-dir "${RESULTS_DIR}" --run-ids "${REF_CSV},${BLK_CSV},${OVL_CSV}" --output-prefix exp3_publishable
run_py -m ecrl.metrics.report_publishable \
  --results-dir "${RESULTS_DIR}" \
  --output-prefix "exp3_publishable" \
  --reference-runs "${REF_CSV}" \
  --blocking-runs "${BLK_CSV}" \
  --overlapped-runs "${OVL_CSV}" \
  --aggregate-reference-label "exp3_reference" \
  --aggregate-blocking-label "exp3_failure_blocking" \
  --aggregate-overlapped-label "exp3_failure_overlapped"
