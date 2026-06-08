#!/usr/bin/env bash
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_DIR="/home/zhangyunhe/nas/code/table/MACT/output/runs/wtq_20260608_1848"

RESULT_PATH="${RUN_DIR}/results.jsonl"
COMMAND_LOG="${RUN_DIR}/eval_command.log"
OUTPUT_LOG="${RUN_DIR}/eval_output.log"

COMMAND=(
  python "${PROJECT_ROOT}/scripts/evaluate_wtq_official.py"
  --dataset wtq
  --result_jsonl "${RESULT_PATH}"
  --tagged_dataset_path /home/zhangyunhe/nas/dataset/WikiTableQuestions/tagged/data
  --prediction_path "${RUN_DIR}/predictions.tsv"
  --metrics_path "${RUN_DIR}/metrics.json"
  --metrics_text_path "${RUN_DIR}/metrics.txt"
  --details_path "${RUN_DIR}/eval_details.jsonl"
  --summary_markdown_path "${RUN_DIR}/eval_summary.md"
)

printf '%q ' "${COMMAND[@]}" > "${COMMAND_LOG}"
printf '\n' >> "${COMMAND_LOG}"

"${COMMAND[@]}" 2>&1 | tee "${OUTPUT_LOG}"
