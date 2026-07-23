#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_MODEL_PATH="/home/zhangyunhe/nas/model/TableLlama"
DEFAULT_DATASET_PATH="${PROJECT_ROOT}/output/crt_answerable.jsonl"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
DATASET_PATH="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
CRT_DATASET_PATH="${CRT_DATASET_PATH:-/home/zhangyunhe/nas/dataset/CRT-QA/CRT-QA}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-4096}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-3968}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SEED="${SEED:-42}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is required." >&2
  echo "Example: CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=16 bash ${SCRIPT_DIR}/run_crt.sh --limit 16" >&2
  exit 2
fi
for value_name in BATCH_SIZE MAX_MODEL_LENGTH MAX_INPUT_TOKENS MAX_NEW_TOKENS TENSOR_PARALLEL_SIZE; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer: ${value}" >&2
    exit 2
  fi
done
IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#VISIBLE_GPUS[@]} < TENSOR_PARALLEL_SIZE )); then
  echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} exceeds visible GPU count ${#VISIBLE_GPUS[@]}." >&2
  exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required for the existing mact environment." >&2
  exit 2
fi
eval "$(conda shell.bash hook)"
conda activate mact

resume_requested=0
limit_requested=0
for arg in "$@"; do
  case "${arg}" in
    --resume) resume_requested=1 ;;
    --limit|--limit=*) limit_requested=1 ;;
  esac
done
if [[ "${resume_requested}" -eq 1 && -z "${RUN_DIR:-}" ]]; then
  echo "Set RUN_DIR to the interrupted run directory when using --resume." >&2
  exit 2
fi
if [[ -z "${RUN_DIR:-}" ]]; then
  RUN_DIR="${SCRIPT_DIR}/output/crt_tablellama-7b_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -e "${RUN_DIR}" && "${resume_requested}" -ne 1 ]]; then
  echo "Run directory already exists: ${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"

RESULT_PATH="${RUN_DIR}/results.jsonl"
RUN_CONFIG_PATH="${RUN_DIR}/run_config.json"
RUN_LOG="${RUN_DIR}/run.log"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

COMMAND=(
  python "${SCRIPT_DIR}/run_tablellama_crt.py"
  --model_name_or_path "${MODEL_PATH}"
  --input_path "${DATASET_PATH}"
  --output_path "${RESULT_PATH}"
  --run_config_path "${RUN_CONFIG_PATH}"
  --batch_size "${BATCH_SIZE}"
  --max_model_length "${MAX_MODEL_LENGTH}"
  --max_input_tokens "${MAX_INPUT_TOKENS}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --seed "${SEED}"
)
if [[ "${ENFORCE_EAGER}" == "1" ]]; then COMMAND+=(--enforce_eager); fi
COMMAND+=("$@")
if [[ "${resume_requested}" -eq 1 ]]; then TEE_ARGS=(-a); else TEE_ARGS=(); fi

{
  echo "TableLlama CRT-QA baseline"
  echo "Run directory: ${RUN_DIR}"
  echo "Model: ${MODEL_PATH}"
  echo "Dataset: ${DATASET_PATH}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
  echo "Batch size: ${BATCH_SIZE}"
  echo "Precision: vLLM auto from checkpoint config (BF16)"
  "${COMMAND[@]}"
  python "${PROJECT_ROOT}/scripts/evaluate_crt_by_type.py" \
    --result_jsonl "${RESULT_PATH}" \
    --crt_dataset_path "${CRT_DATASET_PATH}" \
    --output_dir "${RUN_DIR}"
  if [[ "${limit_requested}" -eq 0 && "${DATASET_PATH}" == "${DEFAULT_DATASET_PATH}" ]]; then
    python - "${RUN_DIR}/crt_type_metrics.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as metrics_file:
    metrics = json.load(metrics_file)
actual = {
    "total_results": metrics.get("total_results"),
    "evaluated": metrics.get("evaluated"),
    "invalid_gold_count": metrics.get("diagnostics", {}).get("invalid_gold_count"),
}
expected = {"total_results": 728, "evaluated": 726, "invalid_gold_count": 2}
if actual != expected:
    raise SystemExit(f"Acceptance check failed: expected {expected}, got {actual}")
print(f"Acceptance check passed: {actual}")
PY
  fi
  echo "Outputs saved to ${RUN_DIR}"
} 2>&1 | tee "${TEE_ARGS[@]}" "${RUN_LOG}"
