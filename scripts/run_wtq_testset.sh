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

Runs the full WTQ test set in parallel. Defaults are intentionally local-vLLM
to avoid accidental external API usage.

Options:
  --model_config PATH   Model config JSON. Default: config/qwen3.5-9b-vllm.json
  --workers N          Parallel worker count. Default: WORKERS env or 4
  --run_dir PATH       Output run directory. Default: output/runs/wtq_testset_qwen3.5-9b_<timestamp>
  -h, --help           Show this help message.

Blocked in this parallel script:
  --limit, --dataset_path, --output_path

Always enabled:
  --use_router --use_verifier --use_code_repair

Already default in tqa.py:
  --log_router

Not enabled intentionally:
  --use_repair is a deprecated alias of --use_code_repair.
  --direct_reasoning changes the agent to direct prompting instead of step-wise MACT.
  --without_tool, --disable_search, --disable_calculate, and --disable_coding_agent
  are ablation/disable switches.
  --debugging limits execution to one example.
  --code_as_observation changes observation formatting instead of adding a module.
EOF
}

# User-facing defaults.
WORKERS="${WORKERS:-4}"
DATASET_PATH="/home/zhangyunhe/nas/code/table/MACT/output/wtq_test_set.jsonl"
MODEL_CONFIG="${PROJECT_ROOT}/config/qwen3.5-9b-vllm.json"
RUN_DIR="${PROJECT_ROOT}/output/runs/wtq_testset_qwen3.5-9b_$(date +%Y%m%d_%H%M)"

# Fixed WTQ settings.
TASK="wtq"
PLAN_SAMPLE=3
CODE_SAMPLE=3
MAX_ACTUAL_STEP=15
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
    --limit|--limit=*|--dataset_path|--dataset_path=*|--output_path|--output_path=*)
      echo "${1%%=*} is not supported by this parallel script." >&2
      echo "Use a non-parallel run if you need to override limit, dataset_path, or output_path." >&2
      exit 1
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${MODEL_CONFIG}" != /* ]]; then
  MODEL_CONFIG="${PROJECT_ROOT}/${MODEL_CONFIG}"
fi

if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${PROJECT_ROOT}/${RUN_DIR}"
fi

RESULT_PATH="${RUN_DIR}/results.jsonl"
COMMAND_LOG="${RUN_DIR}/run_command.log"
SHARD_DIR="${RUN_DIR}/shards"

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS must be a positive integer, got: ${WORKERS}" >&2
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

if [[ -e "${RUN_DIR}" ]]; then
  echo "Run directory already exists: ${RUN_DIR}" >&2
  echo "Wait until the next minute or remove the existing run directory before rerunning." >&2
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
optional_fields = [
    ("api_key", "--api_key"),
    ("base_url", "--base_url"),
]

for key, flag in required_fields:
    value = str(config.get(key, "")).strip()
    if not value:
        raise SystemExit(f"Model config missing required field: {key}")
    print(flag)
    print(value)

for key, flag in optional_fields:
    value = str(config.get(key, "")).strip()
    if value:
        print(flag)
        print(value)
PY

mapfile -t MODEL_CONFIG_ARGS < "${MODEL_CONFIG_ARGS_FILE}"

mkdir -p "${SHARD_DIR}"

for ((worker = 0; worker < WORKERS; worker++)); do
  shard_path=$(printf "%s/shard_%02d.jsonl" "${SHARD_DIR}" "${worker}")
  : > "${shard_path}"
done

awk -v workers="${WORKERS}" -v out_dir="${SHARD_DIR}" '
  NF {
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
  --task "${TASK}"
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
