#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate mact
fi

MODEL_CONFIG="${MODEL_CONFIG:-${PROJECT_ROOT}/baselines/DirectLLM/gpt_5.yaml}"
WORKERS="${WORKERS:-15}"
REQUEST_INTERVAL_S="${REQUEST_INTERVAL_S:-0.7}"

echo "Model config: ${MODEL_CONFIG}"
echo "Workers: ${WORKERS}"
echo "Request interval: ${REQUEST_INTERVAL_S}s"

python "${SCRIPT_DIR}/self_refine_supplement.py" \
  --env_file "${PROJECT_ROOT}/.env" \
  --model_config "${MODEL_CONFIG}" \
  --workers "${WORKERS}" \
  --request_interval_s "${REQUEST_INTERVAL_S}" \
  "$@"
