#!/usr/bin/env python3
"""Extract failed evaluation rows into a directly rerunnable dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} line {line_number}")
            rows.append(row)
    return rows


def index_unique(rows: list[dict], path: Path) -> dict[str, dict]:
    indexed = {}
    for line_number, row in enumerate(rows, start=1):
        example_id = str(row.get("id", "")).strip()
        if not example_id:
            raise ValueError(f"Missing id in {path} row {line_number}")
        if example_id in indexed:
            raise ValueError(f"Duplicate id {example_id!r} in {path}")
        indexed[example_id] = row
    return indexed


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join evaluation failures back to the source dataset for reruns."
    )
    parser.add_argument("--dataset_path", required=True, help="Original input JSONL.")
    parser.add_argument("--details_path", required=True, help="Evaluation details JSONL.")
    parser.add_argument("--output_path", required=True, help="Failed source samples JSONL.")
    parser.add_argument(
        "--manifest_path",
        default="",
        help="Optional failure manifest containing question, gold, prediction, and category.",
    )
    parser.add_argument("--ids_path", default="", help="Optional failed ID text file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path).expanduser()
    details_path = Path(args.details_path).expanduser()
    output_path = Path(args.output_path).expanduser()

    dataset_rows = read_jsonl(dataset_path)
    detail_rows = read_jsonl(details_path)
    dataset_by_id = index_unique(dataset_rows, dataset_path)
    index_unique(detail_rows, details_path)

    failed_details = [row for row in detail_rows if row.get("correct") is False]
    failed_ids = {str(row["id"]) for row in failed_details}
    missing_ids = sorted(failed_ids - dataset_by_id.keys())
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} failed IDs are absent from {dataset_path}: "
            + ", ".join(missing_ids)
        )

    # Preserve source dataset order so rerun output is easy to compare with the original run.
    failed_samples = [row for row in dataset_rows if str(row["id"]) in failed_ids]
    if len(failed_samples) != len(failed_details):
        raise ValueError(
            f"Expected {len(failed_details)} failed samples, extracted {len(failed_samples)}"
        )
    write_jsonl(output_path, failed_samples)

    if args.manifest_path:
        details_by_id = {str(row["id"]): row for row in failed_details}
        manifest_rows = []
        for source_index, sample in enumerate(dataset_rows, start=1):
            example_id = str(sample["id"])
            if example_id not in failed_ids:
                continue
            detail = details_by_id[example_id]
            manifest_rows.append(
                {
                    "source_index": source_index,
                    "id": example_id,
                    "task": sample.get("task"),
                    "statement": sample.get("statement"),
                    "gold_answer": detail.get("gold_answer"),
                    "pred_answer": detail.get("pred_answer"),
                    "error_category": detail.get("error_category", ""),
                    "error": detail.get("error", ""),
                }
            )
        write_jsonl(Path(args.manifest_path).expanduser(), manifest_rows)

    if args.ids_path:
        ids_path = Path(args.ids_path).expanduser()
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.write_text(
            "".join(f"{sample['id']}\n" for sample in failed_samples),
            encoding="utf-8",
        )

    task_counts = Counter(str(sample.get("task", "unknown")) for sample in failed_samples)
    category_counts = Counter(
        str(row.get("error_category") or "unclassified") for row in failed_details
    )
    print(f"Extracted {len(failed_samples)} failed samples to {output_path}")
    print("By task: " + ", ".join(f"{key}={task_counts[key]}" for key in sorted(task_counts)))
    print(
        "By category: "
        + ", ".join(f"{key}={category_counts[key]}" for key in sorted(category_counts))
    )


if __name__ == "__main__":
    main()
