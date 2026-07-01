#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate mact
fi

WORKERS="${WORKERS:-4}"
DATASET_PATH="/home/zhangyunhe/nas/code/table/MACT/output/wtq_test_random_50_error.jsonl"
RUN_DIR="${PROJECT_ROOT}/output/runs/wtq_$(date +%Y%m%d_%H%M)"
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

if [[ -e "${RUN_DIR}" ]]; then
  echo "Run directory already exists: ${RUN_DIR}" >&2
  echo "Wait until the next minute or remove the existing run directory before rerunning." >&2
  exit 1
fi

for arg in "$@"; do
  case "${arg}" in
    --limit|--limit=*|--dataset_path|--dataset_path=*|--output_path|--output_path=*)
      echo "${arg%%=*} is not supported by this parallel script." >&2
      echo "Use a non-parallel run if you need to override limit, dataset_path, or output_path." >&2
      exit 1
      ;;
  esac
done

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
  --plan_backend openai
  --code_backend openai
  --plan_model_name gpt-5.4
  --code_model_name gpt-5.4
  --task wtq
  --plan_sample 1
  --code_sample 1
)

{
  printf 'PROJECT_ROOT=%q\n' "${PROJECT_ROOT}"
  printf 'RUN_DIR=%q\n' "${RUN_DIR}"
  printf 'DATASET_PATH=%q\n' "${DATASET_PATH}"
  printf 'WORKERS=%q\n' "${WORKERS}"
  printf 'BASE_COMMAND='
  printf '%q ' "${BASE_COMMAND[@]}"
  printf '\n'
  printf 'EXTRA_ARGS='
  printf '%q ' "$@"
  printf '\n'
} > "${COMMAND_LOG}"

echo "Run dir: ${RUN_DIR}"
echo "Dataset: ${DATASET_PATH}"
echo "Workers: ${WORKERS}"
echo "Examples: $(cat "${RUN_DIR}/shard_count.txt")"

pids=()
for ((worker = 0; worker < WORKERS; worker++)); do
  shard_path=$(printf "%s/shard_%02d.jsonl" "${SHARD_DIR}" "${worker}")
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
      "$@" 2>&1 \
      | tee "${worker_log}" \
      | awk '/^Finished Trial [0-9]+, Correct: / { print; fflush() }' \
      | sed -u "s/^/[W${worker}] /"
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
