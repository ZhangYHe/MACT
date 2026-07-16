#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate mact
fi

if [[ $# -gt 0 ]]; then
  RESULT_PATH="$1"
  EVAL_DIR="$(dirname "${RESULT_PATH}")"
else
  echo "No result file found" >&2
  exit 1
fi

mkdir -p "${EVAL_DIR}"

python scripts/evaluate_mixed_results.py \
  --result_jsonl "${RESULT_PATH}" \
  --prediction_path "${EVAL_DIR}/predictions.tsv" \
  --metrics_path "${EVAL_DIR}/metrics.json" \
  --metrics_text_path "${EVAL_DIR}/metrics.txt" \
  --details_path "${EVAL_DIR}/eval_details.jsonl" \
  --summary_markdown_path "${EVAL_DIR}/eval_summary.md" \
  > "${EVAL_DIR}/eval.log" 2>&1
