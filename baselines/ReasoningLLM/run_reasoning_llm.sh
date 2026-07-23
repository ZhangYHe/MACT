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
MODEL_CONFIG="${MODEL_CONFIG:-${PROJECT_ROOT}/baselines/DirectLLM/gpt_5.yaml}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/output/gpt_5_reasoning_crt_answerable_$(date +%m%d%H%M)}"
WORKERS="${WORKERS:-15}"
REQUEST_INTERVAL_S="${REQUEST_INTERVAL_S:-0.7}"
MULTI_CHOICE_MODE="${MULTI_CHOICE_MODE:-}"

HAS_MODE_ARG=0
for argument in "$@"; do
  case "${argument}" in
    --multi_choice_mode|--multi_choice_mode=*)
      HAS_MODE_ARG=1
      ;;
  esac
done

MODE_ARGS=()
if [[ "${HAS_MODE_ARG}" -eq 0 ]]; then
  if [[ -z "${MULTI_CHOICE_MODE}" ]]; then
    echo "Choose n=3 behavior explicitly." >&2
    echo "Run test_n3_request.py, then pass --multi_choice_mode n or separate." >&2
    exit 2
  fi
  MODE_ARGS=(--multi_choice_mode "${MULTI_CHOICE_MODE}")
fi

echo "Run dir: ${RUN_DIR}"
echo "Dataset: ${DATASET_PATH}"
echo "Model config: ${MODEL_CONFIG}"
echo "Workers: ${WORKERS}"
echo "Request interval: ${REQUEST_INTERVAL_S}s"

python "${SCRIPT_DIR}/reasoning_llm_baseline.py" \
  --env_file "${PROJECT_ROOT}/.env" \
  --model_config "${MODEL_CONFIG}" \
  --dataset_path "${DATASET_PATH}" \
  --output_dir "${RUN_DIR}" \
  --workers "${WORKERS}" \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_tokens 2000 \
  --frequency_penalty 0 \
  --presence_penalty 0 \
  --request_interval_s "${REQUEST_INTERVAL_S}" \
  "${MODE_ARGS[@]}" \
  "$@"
