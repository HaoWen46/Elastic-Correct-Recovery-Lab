#!/usr/bin/env bash
set -euo pipefail

# Matrix orchestrator for Exp4 paperlite suites.
# Runs multiple paperlite suites across model/dataset/failure/checkpoint combinations.
#
# Usage:
#   scripts/run_exp4_paper_matrix.sh [NPROC]
#
# Example:
#   PROFILE=balanced MODELS_CSV="resnet18,resnet50" DATASETS_CSV="cifar10,cifar100" \
#   scripts/run_exp4_paper_matrix.sh 2

NPROC="${1:-${NPROC:-2}}"
PROFILE="${PROFILE:-balanced}"  # forwarded to run_exp4_paperlite.sh
SEEDS_CSV="${SEEDS_CSV:-}"      # optional override, forwarded

MODELS_CSV="${MODELS_CSV:-resnet18,resnet50}"
DATASETS_CSV="${DATASETS_CSV:-cifar10,cifar100}"
FAILURE_SPECS="${FAILURE_SPECS:-base:400,1200}"  # "label:steps;label:steps"
CHECKPOINT_EVERY_CSV="${CHECKPOINT_EVERY_CSV:-50}"  # e.g. "50,100"
MAX_INFLIGHT_CSV="${MAX_INFLIGHT_CSV:-4}"          # e.g. "2,4"
TARGET_STEPS="${TARGET_STEPS:-}"

MATRIX_PREFIX="${MATRIX_PREFIX:-exp4_matrix_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
SETUP_ENV_FIRST="${SETUP_ENV_FIRST:-1}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-$((30000 + RANDOM % 20000))}"
PYTHON_MANIFEST_BIN="${PYTHON_MANIFEST_BIN:-python3}"

MANIFEST_CSV="${RESULTS_ROOT}/${MATRIX_PREFIX}_manifest.csv"
MANIFEST_MD="${RESULTS_ROOT}/${MATRIX_PREFIX}_manifest.md"
SUMMARY_CSV="${RESULTS_ROOT}/${MATRIX_PREFIX}_summary.csv"
SUMMARY_MD="${RESULTS_ROOT}/${MATRIX_PREFIX}_summary.md"

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  echo "${s}"
}

mkdir -p "${RESULTS_ROOT}"

if ! command -v "${PYTHON_MANIFEST_BIN}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_MANIFEST_BIN="python"
  else
    echo "Need python3 or python available to write manifest markdown." >&2
    exit 1
  fi
fi

IFS=',' read -r -a MODELS_RAW <<< "${MODELS_CSV}"
MODELS=()
for v in "${MODELS_RAW[@]}"; do
  v="$(trim "${v}")"
  [ -n "${v}" ] || continue
  MODELS+=("${v}")
done
if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "Empty MODELS_CSV: ${MODELS_CSV}" >&2
  exit 1
fi

IFS=',' read -r -a DATASETS_RAW <<< "${DATASETS_CSV}"
DATASETS=()
for v in "${DATASETS_RAW[@]}"; do
  v="$(trim "${v}")"
  [ -n "${v}" ] || continue
  DATASETS+=("${v}")
done
if [ "${#DATASETS[@]}" -eq 0 ]; then
  echo "Empty DATASETS_CSV: ${DATASETS_CSV}" >&2
  exit 1
fi

IFS=',' read -r -a CHECKPOINT_RAW <<< "${CHECKPOINT_EVERY_CSV}"
CHECKPOINT_VALUES=()
for v in "${CHECKPOINT_RAW[@]}"; do
  v="$(trim "${v}")"
  [ -n "${v}" ] || continue
  CHECKPOINT_VALUES+=("${v}")
done
if [ "${#CHECKPOINT_VALUES[@]}" -eq 0 ]; then
  echo "Empty CHECKPOINT_EVERY_CSV: ${CHECKPOINT_EVERY_CSV}" >&2
  exit 1
fi

IFS=',' read -r -a INFLIGHT_RAW <<< "${MAX_INFLIGHT_CSV}"
INFLIGHT_VALUES=()
for v in "${INFLIGHT_RAW[@]}"; do
  v="$(trim "${v}")"
  [ -n "${v}" ] || continue
  INFLIGHT_VALUES+=("${v}")
done
if [ "${#INFLIGHT_VALUES[@]}" -eq 0 ]; then
  echo "Empty MAX_INFLIGHT_CSV: ${MAX_INFLIGHT_CSV}" >&2
  exit 1
fi

IFS=';' read -r -a FAIL_RAW <<< "${FAILURE_SPECS}"
FAIL_LABELS=()
FAIL_STEPS_LIST=()
for spec in "${FAIL_RAW[@]}"; do
  spec="$(trim "${spec}")"
  [ -n "${spec}" ] || continue
  if [[ "${spec}" != *:* ]]; then
    echo "Invalid FAILURE_SPECS entry: ${spec} (expected label:step,step)" >&2
    exit 1
  fi
  label="${spec%%:*}"
  steps="${spec#*:}"
  label="$(trim "${label}")"
  steps="$(trim "${steps}")"
  if [ -z "${label}" ] || [ -z "${steps}" ]; then
    echo "Invalid FAILURE_SPECS entry: ${spec}" >&2
    exit 1
  fi
  FAIL_LABELS+=("${label}")
  FAIL_STEPS_LIST+=("${steps}")
done
if [ "${#FAIL_LABELS[@]}" -eq 0 ]; then
  echo "No valid FAILURE_SPECS entries" >&2
  exit 1
fi

echo "[INFO] Matrix prefix: ${MATRIX_PREFIX}"
echo "[INFO] Results root: ${RESULTS_ROOT}"
echo "[INFO] NPROC: ${NPROC}"
echo "[INFO] PROFILE: ${PROFILE}"
echo "[INFO] MODELS: ${MODELS_CSV}"
echo "[INFO] DATASETS: ${DATASETS_CSV}"
echo "[INFO] FAILURE_SPECS: ${FAILURE_SPECS}"
echo "[INFO] CHECKPOINT_EVERY_CSV: ${CHECKPOINT_EVERY_CSV}"
echo "[INFO] MAX_INFLIGHT_CSV: ${MAX_INFLIGHT_CSV}"
echo "[INFO] SKIP_IF_EXISTS: ${SKIP_IF_EXISTS}"
echo "[INFO] DRY_RUN: ${DRY_RUN}"
if [ -n "${TARGET_STEPS}" ]; then
  echo "[INFO] TARGET_STEPS override: ${TARGET_STEPS}"
fi

planned_total=$(( ${#DATASETS[@]} * ${#MODELS[@]} * ${#FAIL_LABELS[@]} * ${#CHECKPOINT_VALUES[@]} * ${#INFLIGHT_VALUES[@]} ))
echo "[INFO] Planned suites: ${planned_total}"

echo "matrix_prefix,run_prefix,results_dir,dataset,model,failure_label,failure_steps,checkpoint_every,max_inflight,status,publishable_md,publishable_json" > "${MANIFEST_CSV}"

setup_env_value="${SETUP_ENV_FIRST}"
current_port_base="${MASTER_PORT_BASE}"
suite_idx=0

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for idx in "${!FAIL_LABELS[@]}"; do
      label="${FAIL_LABELS[$idx]}"
      fail_steps="${FAIL_STEPS_LIST[$idx]}"
      for checkpoint_every in "${CHECKPOINT_VALUES[@]}"; do
        for max_inflight in "${INFLIGHT_VALUES[@]}"; do
          suite_idx=$((suite_idx + 1))
          safe_label="${label//[^a-zA-Z0-9_]/_}"
          run_prefix="${MATRIX_PREFIX}_${dataset}_${model}_f${safe_label}_k${checkpoint_every}_m${max_inflight}"
          run_dir="${RESULTS_ROOT}/${run_prefix}"
          pub_md="${run_dir}/reports/${run_prefix}_publishable.md"
          pub_json="${run_dir}/reports/${run_prefix}_publishable.json"

          echo "[MATRIX ${suite_idx}/${planned_total}] run_prefix=${run_prefix} dataset=${dataset} model=${model} fail=${fail_steps} k=${checkpoint_every} inflight=${max_inflight} port_base=${current_port_base}"

          if [ "${DRY_RUN}" = "1" ]; then
            status="planned"
          elif [ -d "${run_dir}" ] && [ "$(ls -A "${run_dir}" 2>/dev/null | wc -l)" -gt 0 ] && [ "${SKIP_IF_EXISTS}" = "1" ]; then
            status="skipped_exists"
            echo "[WARN] skipping existing run dir: ${run_dir}"
          else
            set +e
            RUN_PREFIX="${run_prefix}" \
            RESULTS_DIR="${run_dir}" \
            NPROC="${NPROC}" \
            PROFILE="${PROFILE}" \
            SEEDS_CSV="${SEEDS_CSV}" \
            MODEL_NAME="${model}" \
            DATASET_NAME="${dataset}" \
            FAIL_STEPS="${fail_steps}" \
            CHECKPOINT_EVERY="${checkpoint_every}" \
            MAX_INFLIGHT="${max_inflight}" \
            TARGET_STEPS="${TARGET_STEPS}" \
            MASTER_PORT_BASE="${current_port_base}" \
            SETUP_ENV="${setup_env_value}" \
            bash scripts/run_exp4_paperlite.sh "${NPROC}"
            rc=$?
            set -e

            if [ "${rc}" -eq 0 ]; then
              status="completed"
            else
              status="failed_rc_${rc}"
              echo "[ERROR] run ${run_prefix} failed with rc=${rc}" >&2
              if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
                echo "${MATRIX_PREFIX},${run_prefix},${run_dir},${dataset},${model},${label},\"${fail_steps}\",${checkpoint_every},${max_inflight},${status},${pub_md},${pub_json}" >> "${MANIFEST_CSV}"
                echo "[INFO] Manifest: ${MANIFEST_CSV}"
                exit "${rc}"
              fi
            fi
          fi

          echo "${MATRIX_PREFIX},${run_prefix},${run_dir},${dataset},${model},${label},\"${fail_steps}\",${checkpoint_every},${max_inflight},${status},${pub_md},${pub_json}" >> "${MANIFEST_CSV}"

          # Avoid re-installing env repeatedly unless explicitly requested.
          setup_env_value="0"
          # Keep ports disjoint across suites.
          current_port_base="$((current_port_base + 50))"
        done
      done
    done
  done
done

"${PYTHON_MANIFEST_BIN}" - "${MANIFEST_CSV}" "${MANIFEST_MD}" "${MATRIX_PREFIX}" "${RESULTS_ROOT}" <<'PY'
import csv
from pathlib import Path
import sys

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
matrix_prefix = sys.argv[3]
results_root = sys.argv[4]

rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))

lines = []
lines.append(f"# {matrix_prefix} Manifest")
lines.append("")
lines.append(f"- Results root: `{results_root}`")
lines.append(f"- CSV manifest: `{csv_path}`")
lines.append("")
lines.append("| run_prefix | dataset | model | failure | k | inflight | status | publishable_md | publishable_json |")
lines.append("|---|---|---|---|---:|---:|---|---|---|")

for r in rows:
    failure = f"{r['failure_label']}:{r['failure_steps']}"
    lines.append(
        f"| {r['run_prefix']} | {r['dataset']} | {r['model']} | {failure} | {r['checkpoint_every']} | {r['max_inflight']} | {r['status']} | "
        f"`{r['publishable_md']}` | `{r['publishable_json']}` |"
    )

md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

"${PYTHON_MANIFEST_BIN}" - "${MANIFEST_CSV}" "${SUMMARY_CSV}" "${SUMMARY_MD}" "${MATRIX_PREFIX}" <<'PY'
import csv
import json
from pathlib import Path
import sys

manifest_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
summary_md = Path(sys.argv[3])
matrix_prefix = sys.argv[4]

rows = list(csv.DictReader(manifest_csv.open("r", encoding="utf-8")))

out_fields = [
    "run_prefix",
    "dataset",
    "model",
    "failure_label",
    "failure_steps",
    "checkpoint_every",
    "max_inflight",
    "status",
    "num_seeds",
    "ref_pass_rate",
    "blk_pass_rate",
    "ovl_pass_rate",
    "ref_goodput_mean",
    "blk_goodput_mean",
    "ovl_goodput_mean",
    "blk_restarts_mean",
    "ovl_restarts_mean",
    "blk_loss_diff_mean",
    "ovl_loss_diff_mean",
    "publishable_json",
]

summary_rows = []

def _get(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

for r in rows:
    record = {k: "" for k in out_fields}
    for k in ("run_prefix", "dataset", "model", "failure_label", "failure_steps", "checkpoint_every", "max_inflight", "status", "publishable_json"):
        record[k] = r.get(k, "")

    p = Path(r.get("publishable_json", ""))
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            record["num_seeds"] = _get(data, "num_seeds") or ""
            record["ref_pass_rate"] = _get(data, "variants", "reference", "correctness", "pass_rate") or ""
            record["blk_pass_rate"] = _get(data, "variants", "failure_blocking", "correctness", "pass_rate") or ""
            record["ovl_pass_rate"] = _get(data, "variants", "failure_overlapped", "correctness", "pass_rate") or ""
            record["ref_goodput_mean"] = _get(data, "variants", "reference", "goodput_steps_per_sec", "mean") or ""
            record["blk_goodput_mean"] = _get(data, "variants", "failure_blocking", "goodput_steps_per_sec", "mean") or ""
            record["ovl_goodput_mean"] = _get(data, "variants", "failure_overlapped", "goodput_steps_per_sec", "mean") or ""
            record["blk_restarts_mean"] = _get(data, "variants", "failure_blocking", "restarts", "mean") or ""
            record["ovl_restarts_mean"] = _get(data, "variants", "failure_overlapped", "restarts", "mean") or ""
            record["blk_loss_diff_mean"] = _get(data, "divergence", "failure_blocking", "loss_mean_abs_diff", "mean") or ""
            record["ovl_loss_diff_mean"] = _get(data, "divergence", "failure_overlapped", "loss_mean_abs_diff", "mean") or ""
        except Exception:
            pass

    summary_rows.append(record)

with summary_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out_fields)
    w.writeheader()
    w.writerows(summary_rows)

md_lines = []
md_lines.append(f"# {matrix_prefix} Summary")
md_lines.append("")
md_lines.append(f"- Source manifest: `{manifest_csv}`")
md_lines.append(f"- Summary csv: `{summary_csv}`")
md_lines.append("")
md_lines.append("| run_prefix | dataset | model | failure | k | inflight | status | seeds | ref gp | blk gp | ovl gp | blk restarts | ovl restarts |")
md_lines.append("|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
for r in summary_rows:
    failure = f"{r['failure_label']}:{r['failure_steps']}"
    md_lines.append(
        f"| {r['run_prefix']} | {r['dataset']} | {r['model']} | {failure} | "
        f"{r['checkpoint_every']} | {r['max_inflight']} | {r['status']} | {r['num_seeds']} | "
        f"{r['ref_goodput_mean']} | {r['blk_goodput_mean']} | {r['ovl_goodput_mean']} | "
        f"{r['blk_restarts_mean']} | {r['ovl_restarts_mean']} |"
    )

summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
PY

echo "[DONE] Manifest CSV: ${MANIFEST_CSV}"
echo "[DONE] Manifest MD: ${MANIFEST_MD}"
echo "[DONE] Summary CSV: ${SUMMARY_CSV}"
echo "[DONE] Summary MD: ${SUMMARY_MD}"
