#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate tablezoomer
fi

# 需要改
DATASET_PATH="${DATASET_PATH:-output/crt_answerable.jsonl}"
MODEL_CONFIG="${MODEL_CONFIG:-config/model_configs/gpt_5_4.yaml}"
RUN_DIR="${RUN_DIR:-baselines/output/gpt_5_4_direct_llm_crt_answerable_$(date +%m%d%H%M)}"

RESULT_PATH="${RUN_DIR}/results.jsonl"

WORKERS="${WORKERS:-15}"
REQUEST_INTERVAL_S="${REQUEST_INTERVAL_S:-0.7}"

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
echo "Request interval: ${REQUEST_INTERVAL_S}s"

python baselines/direct_llm_baseline.py \
  --env_file .env \
  --model_config "${MODEL_CONFIG}" \
  --dataset_path "${DATASET_PATH}" \
  --output_path "${RESULT_PATH}" \
  --workers "${WORKERS}" \
  --request_interval_s "${REQUEST_INTERVAL_S}" \
  "$@" 2>&1 | tee "${RUN_DIR}/run.log"

echo "Saved results to ${RESULT_PATH}"
