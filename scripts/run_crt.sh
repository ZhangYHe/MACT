#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate mact
fi

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [extra tqa.py args]

Runs the CRT answerable dataset in parallel. Defaults target the GPT-5 config.

Options:
  --dataset_path PATH  CRT JSONL path. Default: output/crt_answerable.jsonl
  --model_config PATH  Model config JSON. Default: config/gpt-5.json
  --workers N          Parallel worker count. Default: WORKERS env or 15
  --limit N            Global example limit before sharding. Default: RUN_LIMIT env or 0; 0 means all
  --run_dir PATH       Output run directory. Default: output/runs/crt_<configured-model>_<timestamp>
  -h, --help           Show this help message.

Blocked in this parallel script:
  --task, --output_path

Fixed behavior:
  --task crt
  --plan_sample 1
  --code_sample 1
  --use_router
  --use_verifier
  --use_code_repair

Not enabled intentionally:
  --direct_reasoning, since this script is for the step-wise MACT flow.
  --postprocess_pred_answer, since the latest full-set replay was net negative.
EOF
}

WORKERS="${WORKERS:-15}"
RUN_LIMIT="${RUN_LIMIT:-0}"
DATASET_PATH="${PROJECT_ROOT}/output/crt_answerable.jsonl"
MODEL_CONFIG="${PROJECT_ROOT}/config/gpt-5.json"
RUN_DIR=""

PLAN_SAMPLE=1
CODE_SAMPLE=1
MAX_ACTUAL_STEP=10
SUPPLEMENTAL_ARGS=(
  --use_router
  --use_verifier
  --use_code_repair
)

EXTRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dataset_path)
      if [[ "$#" -lt 2 ]]; then
        echo "--dataset_path requires a path." >&2
        exit 1
      fi
      DATASET_PATH="$2"
      shift 2
      ;;
    --dataset_path=*)
      DATASET_PATH="${1#*=}"
      shift
      ;;
    --model_config)
      if [[ "$#" -lt 2 ]]; then
        echo "--model_config requires a path." >&2
        exit 1
      fi
      MODEL_CONFIG="$2"
      shift 2
      ;;
    --model_config=*)
      MODEL_CONFIG="${1#*=}"
      shift
      ;;
    --workers)
      if [[ "$#" -lt 2 ]]; then
        echo "--workers requires a positive integer." >&2
        exit 1
      fi
      WORKERS="$2"
      shift 2
      ;;
    --workers=*)
      WORKERS="${1#*=}"
      shift
      ;;
    --limit)
      if [[ "$#" -lt 2 ]]; then
        echo "--limit requires a non-negative integer." >&2
        exit 1
      fi
      RUN_LIMIT="$2"
      shift 2
      ;;
    --limit=*)
      RUN_LIMIT="${1#*=}"
      shift
      ;;
    --run_dir)
      if [[ "$#" -lt 2 ]]; then
        echo "--run_dir requires a path." >&2
        exit 1
      fi
      RUN_DIR="$2"
      shift 2
      ;;
    --run_dir=*)
      RUN_DIR="${1#*=}"
      shift
      ;;
    --task|--task=*|--output_path|--output_path=*)
      echo "${1%%=*} is not supported by this parallel script." >&2
      echo "This script fixes --task crt and writes one output per worker." >&2
      exit 1
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${DATASET_PATH}" != /* ]]; then
  DATASET_PATH="${PROJECT_ROOT}/${DATASET_PATH}"
fi

if [[ "${MODEL_CONFIG}" != /* ]]; then
  MODEL_CONFIG="${PROJECT_ROOT}/${MODEL_CONFIG}"
fi

if [[ -n "${RUN_DIR}" && "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${PROJECT_ROOT}/${RUN_DIR}"
fi

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS must be a positive integer, got: ${WORKERS}" >&2
  exit 1
fi

if ! [[ "${RUN_LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "RUN_LIMIT must be a non-negative integer, got: ${RUN_LIMIT}" >&2
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset file does not exist: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "Model config file does not exist: ${MODEL_CONFIG}" >&2
  exit 1
fi

PLAN_MODEL_NAME=$(python -c '
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as input_file:
    print(str(json.load(input_file).get("plan_model_name", "")).strip())
' "${MODEL_CONFIG}")
if [[ -z "${PLAN_MODEL_NAME}" ]]; then
  echo "Model config missing required field: plan_model_name" >&2
  exit 1
fi
MODEL_SLUG="${PLAN_MODEL_NAME//\//-}"
MODEL_SLUG="${MODEL_SLUG// /-}"
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="${PROJECT_ROOT}/output/runs/crt_${MODEL_SLUG}_$(date +%Y%m%d_%H%M)"
fi

RESULT_PATH="${RUN_DIR}/results.jsonl"
COMMAND_LOG="${RUN_DIR}/run_command.log"
SHARD_DIR="${RUN_DIR}/shards"

if [[ -e "${RUN_DIR}" ]]; then
  echo "Run directory already exists: ${RUN_DIR}" >&2
  echo "Wait until the next minute or choose a different --run_dir." >&2
  exit 1
fi

MODEL_CONFIG_ARGS_FILE="$(mktemp)"
trap 'rm -f "${MODEL_CONFIG_ARGS_FILE}"' EXIT

python - "${MODEL_CONFIG}" > "${MODEL_CONFIG_ARGS_FILE}" <<'PY'
import json
import sys

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

required_fields = [
    ("plan_backend", "--plan_backend"),
    ("code_backend", "--code_backend"),
    ("plan_model_name", "--plan_model_name"),
    ("code_model_name", "--code_model_name"),
]
for key, flag in required_fields:
    value = str(config.get(key, "")).strip()
    if not value:
        raise SystemExit(f"Model config missing required field: {key}")
    print(flag)
    print(value)
PY

mapfile -t MODEL_CONFIG_ARGS < "${MODEL_CONFIG_ARGS_FILE}"

mkdir -p "${SHARD_DIR}"

for ((worker = 0; worker < WORKERS; worker++)); do
  shard_path=$(printf "%s/shard_%02d.jsonl" "${SHARD_DIR}" "${worker}")
  : > "${shard_path}"
done

awk -v workers="${WORKERS}" -v out_dir="${SHARD_DIR}" -v limit="${RUN_LIMIT}" '
  NF {
    if (limit > 0 && count >= limit) {
      next
    }
    shard = count % workers
    path = sprintf("%s/shard_%02d.jsonl", out_dir, shard)
    print > path
    count++
  }
  END {
    print count
  }
' "${DATASET_PATH}" > "${RUN_DIR}/shard_count.txt"

BASE_COMMAND=(
  python "${PROJECT_ROOT}/code/tqa.py"
  --env_file "${PROJECT_ROOT}/.env"
  "${MODEL_CONFIG_ARGS[@]}"
  --task crt
  --plan_sample "${PLAN_SAMPLE}"
  --code_sample "${CODE_SAMPLE}"
  --max_actual_step "${MAX_ACTUAL_STEP}"
  "${SUPPLEMENTAL_ARGS[@]}"
)

{
  printf 'PROJECT_ROOT=%q\n' "${PROJECT_ROOT}"
  printf 'RUN_DIR=%q\n' "${RUN_DIR}"
  printf 'DATASET_PATH=%q\n' "${DATASET_PATH}"
  printf 'MODEL_CONFIG=%q\n' "${MODEL_CONFIG}"
  printf 'WORKERS=%q\n' "${WORKERS}"
  printf 'RUN_LIMIT=%q\n' "${RUN_LIMIT}"
  printf 'SUPPLEMENTAL_ARGS='
  printf '%q ' "${SUPPLEMENTAL_ARGS[@]}"
  printf '\n'
  printf 'BASE_COMMAND='
  printf '%q ' "${BASE_COMMAND[@]}"
  printf '\n'
  printf 'EXTRA_ARGS='
  printf '%q ' "${EXTRA_ARGS[@]}"
  printf '\n'
} > "${COMMAND_LOG}"

echo "Run dir: ${RUN_DIR}"
echo "Dataset: ${DATASET_PATH}"
echo "Model config: ${MODEL_CONFIG}"
echo "Workers: ${WORKERS}"
echo "Global limit: ${RUN_LIMIT}"
echo "Supplemental modules: ${SUPPLEMENTAL_ARGS[*]}"
echo "Examples: $(cat "${RUN_DIR}/shard_count.txt")"

pids=()
for ((worker = 0; worker < WORKERS; worker++)); do
  shard_path=$(printf "%s/shard_%02d.jsonl" "${SHARD_DIR}" "${worker}")
  shard_total=$(awk 'END { print NR }' "${shard_path}")
  worker_dir="${RUN_DIR}/worker_${worker}"
  worker_workdir="${worker_dir}/workdir"
  worker_result="${worker_dir}/results.jsonl"
  worker_log="${worker_dir}/run_output.log"
  mkdir -p "${worker_workdir}"

  (
    cd "${worker_workdir}"
    "${BASE_COMMAND[@]}" \
      --dataset_path "${shard_path}" \
      --output_path "${worker_result}" \
      "${EXTRA_ARGS[@]}" 2>&1 \
      | tee "${worker_log}" \
      | awk -v worker="${worker}" -v total="${shard_total}" '
          /^Finished Trial [0-9]+, Correct: / {
            trial += 1
            printf("[W%s] %s\n", worker, $0)
            printf("[W%s] [PROGRESS] %d/%d\n", worker, trial, total)
            fflush()
          }
        '
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "At least one worker failed. Check ${RUN_DIR}/worker_*/run_output.log" >&2
  exit 1
fi

: > "${RESULT_PATH}"
for ((worker = 0; worker < WORKERS; worker++)); do
  worker_result="${RUN_DIR}/worker_${worker}/results.jsonl"
  if [[ ! -f "${worker_result}" ]]; then
    echo "Missing worker result file: ${worker_result}" >&2
    exit 1
  fi
  cat "${worker_result}" >> "${RESULT_PATH}"
done

echo "Saved merged results to ${RESULT_PATH}"

python "${PROJECT_ROOT}/scripts/evaluate_crt_by_type.py" \
  --result_jsonl "${RESULT_PATH}" \
  --output_dir "${RUN_DIR}"
echo "Saved CRT evaluation to ${RUN_DIR}"
