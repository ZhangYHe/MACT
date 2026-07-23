#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate mact
fi

DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/output/crt_answerable.jsonl}"
MODEL_CONFIG="${MODEL_CONFIG:-${SCRIPT_DIR}/gpt_5.yaml}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/output/gpt_5_direct_llm_crt_answerable_$(date +%m%d%H%M)}"

RESULT_PATH="${RUN_DIR}/results.jsonl"

WORKERS="${WORKERS:-15}"
REQUEST_INTERVAL_S="${REQUEST_INTERVAL_S:-0.7}"

#和主实验一样
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-2000}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0}"

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS must be a positive integer, got: ${WORKERS}" >&2
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset file does not exist: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "Model config does not exist: ${MODEL_CONFIG}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"

echo "Run dir: ${RUN_DIR}"
echo "Streaming results to: ${RESULT_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Model config: ${MODEL_CONFIG}"
echo "Workers: ${WORKERS}"
echo "Temperature: ${TEMPERATURE}"
echo "Top p: ${TOP_P}"
echo "Max tokens: ${MAX_TOKENS}"
echo "Frequency penalty: ${FREQUENCY_PENALTY}"
echo "Presence penalty: ${PRESENCE_PENALTY}"
echo "Request interval: ${REQUEST_INTERVAL_S}s"

python "${SCRIPT_DIR}/direct_llm_baseline.py" \
  --env_file "${PROJECT_ROOT}/.env" \
  --model_config "${MODEL_CONFIG}" \
  --dataset_path "${DATASET_PATH}" \
  --output_path "${RESULT_PATH}" \
  --workers "${WORKERS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --max_tokens "${MAX_TOKENS}" \
  --frequency_penalty "${FREQUENCY_PENALTY}" \
  --presence_penalty "${PRESENCE_PENALTY}" \
  --request_interval_s "${REQUEST_INTERVAL_S}" \
  "$@" 2>&1 | tee "${RUN_DIR}/run.log"

echo "Saved results to ${RESULT_PATH}"
