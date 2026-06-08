#!/usr/bin/env python3
"""Evaluate MACT WTQ JSONL results with the WikiTableQuestions denotation metric.

This is a Python 3 compatible implementation of the core logic from the
official WikiTableQuestions evaluator. It reads MACT result JSONL files and
writes a prediction TSV that mirrors the official evaluator input format.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_TAGGED_DATASET_PATH = (
    "/home/zhangyunhe/nas/dataset/WikiTableQuestions/tagged/data"
)


def normalize(value: object) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore")
    else:
        text = str(value)

    text = "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if unicodedata.category(char) != "Mn"
    )
    text = re.sub(r"[‘’´`]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‐‑‒–—−]", "-", text)

    while True:
        old_text = text
        text = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", text.strip())
        text = re.sub(r"(?<!^)( \([^)]*\))*$", "", text.strip())
        text = re.sub(r'^"([^"]*)"$', r"\1", text.strip())
        if text == old_text:
            break

    if text.endswith("."):
        text = text[:-1]
    return re.sub(r"\s+", " ", text).lower().strip()


class Value(ABC):
    @property
    @abstractmethod
    def normalized(self) -> str:
        pass

    @abstractmethod
    def match(self, other: "Value") -> bool:
        pass


@dataclass(frozen=True)
class StringValue(Value):
    content: str

    @property
    def normalized(self) -> str:
        return normalize(self.content)

    def match(self, other: Value) -> bool:
        return self.normalized == other.normalized

    def __repr__(self) -> str:
        return f"S{[self.normalized]}"


@dataclass(frozen=True)
class NumberValue(Value):
    amount: int | float
    original_string: str | None = None

    def __post_init__(self) -> None:
        if abs(float(self.amount) - round(float(self.amount))) < 1e-6:
            object.__setattr__(self, "amount", int(round(float(self.amount))))
        else:
            object.__setattr__(self, "amount", float(self.amount))

    @property
    def normalized(self) -> str:
        if self.original_string:
            return normalize(self.original_string)
        return str(self.amount)

    def match(self, other: Value) -> bool:
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(float(self.amount) - float(other.amount)) < 1e-6
        return False

    def __repr__(self) -> str:
        return f"N({float(self.amount):f}){[self.normalized]}"

    @staticmethod
    def parse(text: str) -> int | float | None:
        try:
            return int(text)
        except (TypeError, ValueError):
            try:
                amount = float(text)
                if math.isnan(amount) or math.isinf(amount):
                    return None
                return amount
            except (TypeError, ValueError):
                return None


@dataclass(frozen=True)
class DateValue(Value):
    year: int
    month: int
    day: int
    original_string: str | None = None

    @property
    def normalized(self) -> str:
        if self.original_string:
            return normalize(self.original_string)
        year = self.year if self.year != -1 else "xx"
        month = self.month if self.month != -1 else "xx"
        day = self.day if self.day != -1 else "xx"
        return f"{year}-{month}-{day}"

    def match(self, other: Value) -> bool:
        if self.normalized == other.normalized:
            return True
        if isinstance(other, DateValue):
            return (self.year, self.month, self.day) == (
                other.year,
                other.month,
                other.day,
            )
        return False

    def __repr__(self) -> str:
        return f"D({self.year},{self.month},{self.day}){[self.normalized]}"

    @staticmethod
    def parse(text: str) -> tuple[int, int, int] | None:
        try:
            year_text, month_text, day_text = text.lower().split("-")
            year = -1 if year_text in {"xx", "xxxx"} else int(year_text)
            month = -1 if month_text == "xx" else int(month_text)
            day = -1 if day_text == "xx" else int(day_text)
            if year == month == day == -1:
                return None
            if month != -1 and not 1 <= month <= 12:
                return None
            if day != -1 and not 1 <= day <= 31:
                return None
            return year, month, day
        except (AttributeError, ValueError):
            return None


def to_value(original_string: object, corenlp_value: object | None = None) -> Value:
    if isinstance(original_string, Value):
        return original_string

    original = str(original_string)
    canonical = original if corenlp_value in (None, "") else str(corenlp_value)

    amount = NumberValue.parse(canonical)
    if amount is not None:
        return NumberValue(amount, original)

    ymd = DateValue.parse(canonical)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return NumberValue(ymd[0], original)
        return DateValue(ymd[0], ymd[1], ymd[2], original)

    return StringValue(original)


def to_value_list(
    original_strings: Iterable[object],
    corenlp_values: Iterable[object] | None = None,
) -> list[Value]:
    if corenlp_values is None:
        return list(set(to_value(item) for item in original_strings))

    originals = list(original_strings)
    canonicals = list(corenlp_values)
    if len(originals) != len(canonicals):
        raise ValueError("Original and canonical value lists have different lengths.")
    return list(set(to_value(item, canonical) for item, canonical in zip(originals, canonicals)))


def check_denotation(target_values: list[Value], predicted_values: list[Value]) -> bool:
    if len(target_values) != len(predicted_values):
        return False
    return all(any(target.match(prediction) for prediction in predicted_values) for target in target_values)


def tsv_unescape(value: str) -> str:
    return value.replace(r"\n", "\n").replace(r"\p", "|").replace(r"\\", "\\")


def tsv_unescape_list(value: str) -> list[str]:
    return [tsv_unescape(part) for part in value.split("|")]


def load_target_values(tagged_dataset_path: Path) -> dict[str, list[Value]]:
    if not tagged_dataset_path.is_dir():
        raise FileNotFoundError(f"Tagged dataset directory does not exist: {tagged_dataset_path}")

    target_values_map: dict[str, list[Value]] = {}
    for tagged_file in sorted(path for path in tagged_dataset_path.iterdir() if path.is_file()):
        with tagged_file.open("r", encoding="utf-8") as input_file:
            header = input_file.readline().rstrip("\n").split("\t")
            required_fields = {"id", "targetValue", "targetCanon"}
            if not required_fields.issubset(header):
                raise ValueError(
                    f"Unexpected WTQ tagged fields in {tagged_file}: {header}. "
                    f"Expected at least: {sorted(required_fields)}"
                )
            for line in input_file:
                fields = line.rstrip("\n").split("\t")
                row = dict(zip(header, fields))
                example_id = row["id"]
                target_values_map[example_id] = to_value_list(
                    tsv_unescape_list(row["targetValue"]),
                    tsv_unescape_list(row["targetCanon"]),
                )
    return target_values_map


def prediction_to_string(prediction: object) -> str:
    if prediction is None:
        return ""
    if isinstance(prediction, str):
        return prediction
    return str(prediction)


def load_results(result_jsonl: Path) -> list[dict]:
    if not result_jsonl.is_file():
        raise FileNotFoundError(f"Result JSONL does not exist: {result_jsonl}")

    results = []
    with result_jsonl.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "id" not in item:
                raise ValueError(f"Missing 'id' in {result_jsonl} line {line_number}")
            results.append(item)
    return results


def write_prediction_tsv(results: list[dict], prediction_path: Path) -> None:
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as output_file:
        for item in results:
            prediction = prediction_to_string(item.get("pred_answer"))
            output_file.write(f"{item['id']}\t{prediction}\n")


def evaluate_results(
    results: list[dict],
    target_values_map: dict[str, list[Value]],
) -> tuple[dict, list[dict]]:
    details = []
    num_examples = 0
    num_correct = 0
    missing_ids = []
    empty_predictions = []

    for item in results:
        example_id = item["id"]
        prediction = prediction_to_string(item.get("pred_answer"))
        predicted_values = [] if prediction == "" else to_value_list([prediction])

        if example_id not in target_values_map:
            missing_ids.append(example_id)
            details.append(
                {
                    "id": example_id,
                    "correct": None,
                    "pred_answer": prediction,
                    "target_values": None,
                    "predicted_values": [repr(value) for value in predicted_values],
                    "error": "missing_id",
                }
            )
            continue

        if prediction == "":
            empty_predictions.append(example_id)

        target_values = target_values_map[example_id]
        correct = check_denotation(target_values, predicted_values)
        num_examples += 1
        if correct:
            num_correct += 1

        details.append(
            {
                "id": example_id,
                "correct": correct,
                "pred_answer": prediction,
                "target_values": [repr(value) for value in target_values],
                "predicted_values": [repr(value) for value in predicted_values],
            }
        )

    accuracy = num_correct / num_examples if num_examples else 0.0
    metrics = {
        "metric": "wtq_official_denotation_accuracy",
        "total_results": len(results),
        "evaluated": num_examples,
        "correct": num_correct,
        "accuracy": round(accuracy, 4),
        "missing_id_count": len(missing_ids),
        "missing_ids": missing_ids,
        "empty_prediction_count": len(empty_predictions),
        "empty_prediction_ids": empty_predictions,
    }
    return metrics, details


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metrics_text(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        output_file.write(f"Metric: {metrics['metric']}\n")
        output_file.write(f"Evaluated: {metrics['evaluated']}\n")
        output_file.write(f"Correct: {metrics['correct']}\n")
        output_file.write(f"Accuracy: {metrics['accuracy']}\n")
        output_file.write(f"Missing IDs: {metrics['missing_id_count']}\n")
        output_file.write(f"Empty predictions: {metrics['empty_prediction_count']}\n")


def infer_dataset_name(result_jsonl: Path) -> str:
    run_dir_name = result_jsonl.parent.name
    if "_" in run_dir_name:
        dataset = run_dir_name.split("_", maxsplit=1)[0]
        return dataset or "unknown"
    return "unknown"


def markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def write_summary_markdown(
    path: Path,
    metrics: dict,
    dataset: str,
    result_jsonl: Path,
) -> None:
    run_dir = result_jsonl.parent.name
    rows = [
        [
            dataset,
            f"{metrics['accuracy']:.4f}",
            metrics["correct"],
            metrics["evaluated"],
            metrics["total_results"],
            metrics["metric"],
            run_dir,
            metrics["missing_id_count"],
            metrics["empty_prediction_count"],
        ]
    ]
    header = [
        "dataset",
        "accuracy",
        "correct",
        "evaluated",
        "total_results",
        "metric",
        "run_dir",
        "missing_ids",
        "empty_predictions",
    ]
    alignment = ["---", "---:", "---:", "---:", "---:", "---", "---", "---:", "---:"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        output_file.write("| " + " | ".join(header) + " |\n")
        output_file.write("| " + " | ".join(alignment) + " |\n")
        for row in rows:
            output_file.write("| " + " | ".join(markdown_cell(item) for item in row) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MACT WTQ results with the official denotation metric."
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Dataset name for summary output. Defaults to inferring from result_jsonl parent directory.",
    )
    parser.add_argument("--result_jsonl", required=True, help="Path to MACT result JSONL.")
    parser.add_argument(
        "--tagged_dataset_path",
        default=DEFAULT_TAGGED_DATASET_PATH,
        help=f"WTQ tagged data directory. Default: {DEFAULT_TAGGED_DATASET_PATH}",
    )
    parser.add_argument("--prediction_path", required=True, help="Output official-format TSV.")
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
        help="Optional one-row Markdown summary table path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_jsonl = Path(args.result_jsonl).expanduser()
    tagged_dataset_path = Path(args.tagged_dataset_path).expanduser()
    prediction_path = Path(args.prediction_path).expanduser()
    metrics_path = Path(args.metrics_path).expanduser()
    details_path = Path(args.details_path).expanduser()
    metrics_text_path = Path(args.metrics_text_path).expanduser() if args.metrics_text_path else None
    summary_markdown_path = (
        Path(args.summary_markdown_path).expanduser() if args.summary_markdown_path else None
    )
    dataset = args.dataset or infer_dataset_name(result_jsonl)

    results = load_results(result_jsonl)
    target_values_map = load_target_values(tagged_dataset_path)
    write_prediction_tsv(results, prediction_path)

    metrics, details = evaluate_results(results, target_values_map)
    write_json(metrics_path, metrics)
    write_jsonl(details_path, details)
    if metrics_text_path is not None:
        write_metrics_text(metrics_text_path, metrics)
    if summary_markdown_path is not None:
        write_summary_markdown(summary_markdown_path, metrics, dataset, result_jsonl)

    print(f"Evaluated {metrics['evaluated']} examples")
    print(f"Correct: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']}")
    if metrics["missing_id_count"]:
        print(f"Missing IDs: {metrics['missing_id_count']}")
    if metrics["empty_prediction_count"]:
        print(f"Empty predictions: {metrics['empty_prediction_count']}")


if __name__ == "__main__":
    main()
