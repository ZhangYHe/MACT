#!/usr/bin/env python3
"""Evaluate mixed WTQ/CRT/SciTab MACT results with WTQ denotation matching."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_wtq_official import (
    check_denotation,
    load_results,
    markdown_cell,
    prediction_to_items,
    prediction_to_string,
    prediction_to_tsv_string,
    to_value_list,
    write_json,
    write_jsonl,
)


TASK_METRICS = {
    "wtq": "wtq_denotation_accuracy_from_answer_field",
    "crt": "crt_denotation_accuracy",
    "scitab": "scitab_denotation_accuracy",
}


def normalize_task(item: dict, line_index: int) -> str:
    task = str(item.get("task", "")).strip().lower()
    if task not in TASK_METRICS:
        raise ValueError(
            f"Unsupported or missing task at result index {line_index}: {task!r}. "
            f"Expected one of: {', '.join(sorted(TASK_METRICS))}"
        )
    return task


def answer_to_items(answer: object) -> list[object]:
    return prediction_to_items(answer)


def evaluate_item(item: dict, task: str) -> dict:
    example_id = item["id"]
    prediction = prediction_to_string(item.get("pred_answer"))
    prediction_items = prediction_to_items(item.get("pred_answer"))
    gold_items = answer_to_items(item.get("answer"))
    target_values = [] if not gold_items else to_value_list(gold_items)
    predicted_values = [] if not prediction_items else to_value_list(prediction_items)

    error = ""
    if "answer" not in item:
        error = "missing_answer"
    elif not gold_items:
        error = "empty_gold"

    correct = False if error else check_denotation(target_values, predicted_values)

    return {
        "id": example_id,
        "task": task,
        "metric": TASK_METRICS[task],
        "correct": correct,
        "error": error,
        "gold_answer": item.get("answer"),
        "pred_answer": prediction,
        "gold_items": [str(value) for value in gold_items],
        "pred_items": [str(value) for value in prediction_items],
        "target_values": [repr(value) for value in target_values],
        "predicted_values": [repr(value) for value in predicted_values],
    }


def summarize(details: list[dict], total_results: int) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for detail in details:
        by_task[detail["task"]].append(detail)

    task_metrics = {}
    total_evaluated = 0
    total_correct = 0
    total_empty_predictions = 0
    total_invalid = 0

    for task in sorted(TASK_METRICS):
        task_details = by_task.get(task, [])
        evaluated = len(task_details)
        correct = sum(1 for detail in task_details if detail["correct"])
        empty_predictions = sum(
            1 for detail in task_details if detail["pred_answer"].strip() == ""
        )
        invalid = sum(1 for detail in task_details if detail["error"])
        accuracy = correct / evaluated if evaluated else 0.0
        task_metrics[task] = {
            "metric": TASK_METRICS[task],
            "total_results": evaluated,
            "evaluated": evaluated,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "empty_prediction_count": empty_predictions,
            "invalid_gold_count": invalid,
        }
        total_evaluated += evaluated
        total_correct += correct
        total_empty_predictions += empty_predictions
        total_invalid += invalid

    overall_accuracy = total_correct / total_evaluated if total_evaluated else 0.0
    return {
        "metric": "mixed_denotation_accuracy",
        "total_results": total_results,
        "evaluated": total_evaluated,
        "correct": total_correct,
        "accuracy": round(overall_accuracy, 4),
        "empty_prediction_count": total_empty_predictions,
        "invalid_gold_count": total_invalid,
        "task_counts": dict(Counter(detail["task"] for detail in details)),
        "by_task": task_metrics,
    }


def write_metrics_text(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        output_file.write(f"Metric: {metrics['metric']}\n")
        output_file.write(f"Evaluated: {metrics['evaluated']}\n")
        output_file.write(f"Correct: {metrics['correct']}\n")
        output_file.write(f"Accuracy: {metrics['accuracy']}\n")
        output_file.write(f"Empty predictions: {metrics['empty_prediction_count']}\n")
        output_file.write(f"Invalid gold: {metrics['invalid_gold_count']}\n")
        output_file.write("\nPer-task metrics:\n")
        for task, task_metrics in metrics["by_task"].items():
            output_file.write(
                f"- {task}: {task_metrics['correct']}/{task_metrics['evaluated']} "
                f"accuracy={task_metrics['accuracy']} "
                f"metric={task_metrics['metric']}\n"
            )


def write_summary_markdown(path: Path, metrics: dict, result_jsonl: Path) -> None:
    run_dir = result_jsonl.parent.name
    rows = []
    for task, task_metrics in metrics["by_task"].items():
        rows.append(
            [
                task,
                f"{task_metrics['accuracy']:.4f}",
                task_metrics["correct"],
                task_metrics["evaluated"],
                task_metrics["total_results"],
                task_metrics["metric"],
                run_dir,
                task_metrics["empty_prediction_count"],
                task_metrics["invalid_gold_count"],
            ]
        )
    rows.append(
        [
            "overall",
            f"{metrics['accuracy']:.4f}",
            metrics["correct"],
            metrics["evaluated"],
            metrics["total_results"],
            metrics["metric"],
            run_dir,
            metrics["empty_prediction_count"],
            metrics["invalid_gold_count"],
        ]
    )

    header = [
        "dataset",
        "accuracy",
        "correct",
        "evaluated",
        "total_results",
        "metric",
        "run_dir",
        "empty_predictions",
        "invalid_gold",
    ]
    alignment = ["---", "---:", "---:", "---:", "---:", "---", "---", "---:", "---:"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        output_file.write("| " + " | ".join(header) + " |\n")
        output_file.write("| " + " | ".join(alignment) + " |\n")
        for row in rows:
            output_file.write("| " + " | ".join(markdown_cell(item) for item in row) + " |\n")


def write_prediction_tsv(results: list[dict], prediction_path: Path) -> None:
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as output_file:
        for item in results:
            prediction = prediction_to_tsv_string(item.get("pred_answer"))
            output_file.write(f"{item['id']}\t{prediction}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate mixed WTQ/CRT/SciTab MACT results using the WTQ official "
            "denotation matching logic and gold answers stored in each result row."
        )
    )
    parser.add_argument("--result_jsonl", required=True, help="Path to MACT result JSONL.")
    parser.add_argument("--prediction_path", default="", help="Optional prediction TSV path.")
    parser.add_argument("--metrics_path", required=True, help="Output metrics JSON.")
    parser.add_argument("--details_path", required=True, help="Output per-example JSONL details.")
    parser.add_argument(
        "--metrics_text_path",
        default="",
        help="Optional human-readable metrics text path.",
    )
    parser.add_argument(
        "--summary_markdown_path",
        default="",
        help="Optional Markdown summary table path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_jsonl = Path(args.result_jsonl).expanduser()
    metrics_path = Path(args.metrics_path).expanduser()
    details_path = Path(args.details_path).expanduser()
    prediction_path = Path(args.prediction_path).expanduser() if args.prediction_path else None
    metrics_text_path = Path(args.metrics_text_path).expanduser() if args.metrics_text_path else None
    summary_markdown_path = (
        Path(args.summary_markdown_path).expanduser() if args.summary_markdown_path else None
    )

    if not result_jsonl.exists():
        raise SystemExit(f"Result file does not exist: {result_jsonl}")
    if result_jsonl.stat().st_size == 0:
        raise SystemExit(
            f"Result file is empty: {result_jsonl}. "
            "The run likely failed before writing predictions."
        )

    results = load_results(result_jsonl)
    if not results:
        raise SystemExit(
            f"Result file has no result rows: {result_jsonl}. "
            "The run likely failed before writing predictions."
        )
    details = [
        evaluate_item(item, normalize_task(item, index))
        for index, item in enumerate(results, start=1)
    ]
    metrics = summarize(details, total_results=len(results))

    if prediction_path is not None:
        write_prediction_tsv(results, prediction_path)
    write_json(metrics_path, metrics)
    write_jsonl(details_path, details)
    if metrics_text_path is not None:
        write_metrics_text(metrics_text_path, metrics)
    if summary_markdown_path is not None:
        write_summary_markdown(summary_markdown_path, metrics, result_jsonl)

    print(f"Evaluated {metrics['evaluated']} examples")
    print(f"Correct: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']}")
    for task, task_metrics in metrics["by_task"].items():
        print(
            f"{task}: {task_metrics['correct']}/{task_metrics['evaluated']} "
            f"accuracy={task_metrics['accuracy']}"
        )


if __name__ == "__main__":
    main()
