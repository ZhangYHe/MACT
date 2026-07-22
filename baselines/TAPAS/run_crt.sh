#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_MODEL_PATH="/home/zhangyunhe/nas/model/tapas-large-finetuned-wtq"
DEFAULT_DATASET_PATH="${PROJECT_ROOT}/output/crt_answerable.jsonl"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
DATASET_PATH="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
CRT_DATASET_PATH="${CRT_DATASET_PATH:-/home/zhangyunhe/nas/dataset/CRT-QA/CRT-QA}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-1024}"
CELL_CLASSIFICATION_THRESHOLD="${CELL_CLASSIFICATION_THRESHOLD:-0.5}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is required." >&2
  echo "Example: CUDA_VISIBLE_DEVICES=2 bash ${SCRIPT_DIR}/run_crt.sh --limit 5" >&2
  exit 2
fi

if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer, got: ${BATCH_SIZE}" >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to activate the mact environment." >&2
  exit 2
fi

eval "$(conda shell.bash hook)"
conda activate mact

resume_requested=0
limit_requested=0
for arg in "$@"; do
  case "${arg}" in
    --resume)
      resume_requested=1
      ;;
    --limit|--limit=*)
      limit_requested=1
      ;;
  esac
done

if [[ "${resume_requested}" -eq 1 && -z "${RUN_DIR:-}" ]]; then
  echo "Set RUN_DIR to the existing run directory when using --resume." >&2
  exit 2
fi

if [[ -z "${RUN_DIR:-}" ]]; then
  RUN_DIR="${SCRIPT_DIR}/output/crt_tapas-large-wtq_$(date +%Y%m%d_%H%M%S)"
fi

RESULT_PATH="${RUN_DIR}/results.jsonl"
RUN_CONFIG_PATH="${RUN_DIR}/run_config.json"
RUN_LOG="${RUN_DIR}/run.log"

if [[ -e "${RUN_DIR}" && "${resume_requested}" -ne 1 ]]; then
  echo "Run directory already exists: ${RUN_DIR}" >&2
  echo "Choose another RUN_DIR, or pass --resume for an interrupted run." >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"

export TOKENIZERS_PARALLELISM=false

INFERENCE_COMMAND=(
  python "${SCRIPT_DIR}/run_tapas_crt.py"
  --model_name_or_path "${MODEL_PATH}"
  --input_path "${DATASET_PATH}"
  --output_path "${RESULT_PATH}"
  --run_config_path "${RUN_CONFIG_PATH}"
  --batch_size "${BATCH_SIZE}"
  --max_source_length "${MAX_SOURCE_LENGTH}"
  --cell_classification_threshold "${CELL_CLASSIFICATION_THRESHOLD}"
  --device cuda:0
  "$@"
)

if [[ "${resume_requested}" -eq 1 ]]; then
  TEE_ARGS=(-a)
else
  TEE_ARGS=()
fi

{
  echo "TAPAS CRT-QA baseline"
  echo "Run directory: ${RUN_DIR}"
  echo "Model: ${MODEL_PATH}"
  echo "Dataset: ${DATASET_PATH}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "Batch size: ${BATCH_SIZE}"
  echo "Precision: checkpoint/PyTorch default"
  printf "Inference command:"
  printf " %q" "${INFERENCE_COMMAND[@]}"
  printf "\n"

  "${INFERENCE_COMMAND[@]}"

  python "${PROJECT_ROOT}/scripts/evaluate_crt_by_type.py" \
    --result_jsonl "${RESULT_PATH}" \
    --crt_dataset_path "${CRT_DATASET_PATH}" \
    --output_dir "${RUN_DIR}"

  if [[ "${limit_requested}" -eq 0 && "${DATASET_PATH}" == "${DEFAULT_DATASET_PATH}" ]]; then
    python - "${RUN_DIR}/crt_type_metrics.json" <<'PY'
import json
import sys

metrics_path = sys.argv[1]
with open(metrics_path, encoding="utf-8") as metrics_file:
    metrics = json.load(metrics_file)

expected = {
    "total_results": 728,
    "evaluated": 726,
    "invalid_gold_count": 2,
}
actual = {
    "total_results": metrics.get("total_results"),
    "evaluated": metrics.get("evaluated"),
    "invalid_gold_count": metrics.get("diagnostics", {}).get("invalid_gold_count"),
}
if actual != expected:
    raise SystemExit(f"Full-run acceptance check failed: expected {expected}, got {actual}")
print(f"Full-run acceptance check passed: {actual}")
PY
  fi

  echo "Baseline outputs saved to ${RUN_DIR}"
} 2>&1 | tee "${TEE_ARGS[@]}" "${RUN_LOG}"
