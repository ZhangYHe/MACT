#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_MODEL_PATH="/home/zhangyunhe/nas/model/omnitab-large-finetuned-wtq"
DEFAULT_DATASET_PATH="${PROJECT_ROOT}/output/crt_answerable.jsonl"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
DATASET_PATH="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
CRT_DATASET_PATH="${CRT_DATASET_PATH:-/home/zhangyunhe/nas/dataset/CRT-QA/CRT-QA}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
SEED="${SEED:-42}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is required." >&2
  echo "Example: CUDA_VISIBLE_DEVICES=2 bash ${SCRIPT_DIR}/run_crt.sh --limit 5" >&2
  exit 2
fi
if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer: ${BATCH_SIZE}" >&2
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
  RUN_DIR="${SCRIPT_DIR}/output/crt_omnitab-large-wtq_$(date +%Y%m%d_%H%M%S)"
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
  python "${SCRIPT_DIR}/run_omnitab_crt.py"
  --model_name_or_path "${MODEL_PATH}"
  --input_path "${DATASET_PATH}"
  --output_path "${RESULT_PATH}"
  --run_config_path "${RUN_CONFIG_PATH}"
  --batch_size "${BATCH_SIZE}"
  --max_source_length "${MAX_SOURCE_LENGTH}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --seed "${SEED}"
  --device cuda:0
  "$@"
)
if [[ "${resume_requested}" -eq 1 ]]; then TEE_ARGS=(-a); else TEE_ARGS=(); fi

{
  echo "OmniTab CRT-QA baseline"
  echo "Run directory: ${RUN_DIR}"
  echo "Model: ${MODEL_PATH}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "Batch size: ${BATCH_SIZE}"
  echo "Precision: checkpoint/PyTorch default"
  "${COMMAND[@]}"
  python "${PROJECT_ROOT}/scripts/evaluate_crt_by_type.py" \
    --result_jsonl "${RESULT_PATH}" \
    --crt_dataset_path "${CRT_DATASET_PATH}" \
    --output_dir "${RUN_DIR}"
  if [[ "${limit_requested}" -eq 0 && "${DATASET_PATH}" == "${DEFAULT_DATASET_PATH}" ]]; then
    python - "${RUN_DIR}/crt_type_metrics.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    metrics = json.load(f)
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
