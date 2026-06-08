#!/usr/bin/env bash
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${PROJECT_ROOT}/output/runs/wtq_$(date +%Y%m%d_%H%M)"
RESULT_PATH="${RUN_DIR}/results.jsonl"
COMMAND_LOG="${RUN_DIR}/run_command.log"
OUTPUT_LOG="${RUN_DIR}/run_output.log"

mkdir -p "${RUN_DIR}"

if [[ -f "${RESULT_PATH}" ]]; then
  echo "Result file already exists: ${RESULT_PATH}" >&2
  echo "Wait until the next minute or remove the existing run directory before rerunning." >&2
  exit 1
fi

COMMAND=(
  python tqa.py
  --env_file "${PROJECT_ROOT}/.env"
  --plan_backend openai
  --code_backend openai
  --plan_model_name gpt-5.4
  --code_model_name gpt-5.4
  --dataset_path "${PROJECT_ROOT}/output/wtq_test_random_50.jsonl"
  --task wtq
  --plan_sample 1
  --code_sample 1
  --output_path "${RESULT_PATH}"
)

printf 'cd %q\n' "${PROJECT_ROOT}/code" > "${COMMAND_LOG}"
printf '%q ' "${COMMAND[@]}" >> "${COMMAND_LOG}"
printf '\n' >> "${COMMAND_LOG}"

cd "${PROJECT_ROOT}/code"
"${COMMAND[@]}" 2>&1 | tee "${OUTPUT_LOG}"
