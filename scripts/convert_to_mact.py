#!/usr/bin/env python3
"""Convert supported table QA datasets to MACT JSONL format."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable


DEFAULT_WTQ_PATH = "/home/zhangyunhe/nas/dataset/WikiTableQuestions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert supported table QA datasets to MACT JSONL format."
    )
    parser.add_argument(
        "--task",
        default="wtq",
        choices=["wtq"],
        help="Dataset conversion task. Default: wtq.",
    )
    parser.add_argument(
        "--dataset_path",
        default=DEFAULT_WTQ_PATH,
        help=(
            "Path to the source dataset root directory. "
            f"For --task wtq, default: {DEFAULT_WTQ_PATH}"
        ),
    )
    parser.add_argument(
        "--split",
        required=True,
        help=(
            "Dataset split to convert. For WTQ, supported aliases include train, "
            "test, dev, validation, pristine-unseen-tables, pristine-seen-tables, "
            "training-before300, and random-split-{1..5}-{train,dev}."
        ),
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to write. Default: write all examples.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle examples before applying --limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is set. Default: 42.",
    )
    parser.add_argument(
        "--include_metadata",
        action="store_true",
        help="Include id, dataset, split, and table_id in each output item.",
    )
    return parser.parse_args()


def resolve_wtq_split_file(dataset_path: Path, split: str) -> Path:
    split_map = {
        "train": "training.tsv",
        "training": "training.tsv",
        "test": "pristine-unseen-tables.tsv",
        "pristine-unseen-tables": "pristine-unseen-tables.tsv",
        "dev": "pristine-seen-tables.tsv",
        "validation": "pristine-seen-tables.tsv",
        "pristine-seen-tables": "pristine-seen-tables.tsv",
        "training-before300": "training-before300.tsv",
    }

    if split.startswith("random-split-"):
        parts = split.split("-")
        if len(parts) == 4 and parts[0] == "random" and parts[1] == "split":
            seed, subset = parts[2], parts[3]
            if seed in {"1", "2", "3", "4", "5"} and subset in {"train", "dev"}:
                split_map[split] = f"{split}.tsv"

    if split not in split_map:
        valid_splits = sorted(
            list(split_map)
            + [
                f"random-split-{seed}-{subset}"
                for seed in range(1, 6)
                for subset in ("train", "dev")
            ]
        )
        raise ValueError(
            f"Invalid WTQ split '{split}'. Supported splits: {', '.join(valid_splits)}"
        )

    split_file = dataset_path / "data" / split_map[split]
    if not split_file.is_file():
        raise FileNotFoundError(f"WTQ split file does not exist: {split_file}")
    return split_file


def unescape_wtq_value(value: str) -> str:
    """Undo WTQ TSV escaping for values read from questions/answers/tables."""
    return value.replace(r"\n", " ").replace(r"\p", "|").replace(r"\\", "\\").strip()


def split_wtq_answer(answer: str) -> list[str]:
    if answer == "":
        return []
    return [unescape_wtq_value(part) for part in answer.split("|")]


def normalize_wtq_row(row: Iterable[str]) -> list[str]:
    return [unescape_wtq_value(cell) for cell in row]


def read_wtq_table(dataset_path: Path, table_context: str) -> tuple[str, list[list[str]]]:
    table_context = table_context.replace(".csv", ".tsv")
    table_path = dataset_path / table_context
    if not table_path.is_file():
        raise FileNotFoundError(f"WTQ table file does not exist: {table_path}")

    with table_path.open("r", encoding="utf-8", newline="") as table_file:
        rows = [
            normalize_wtq_row(row)
            for row in csv.reader(table_file, delimiter="\t")
        ]

    if not rows:
        raise ValueError(f"WTQ table file is empty: {table_path}")

    return table_context, rows


def load_wtq_examples(
    dataset_path: Path,
    split: str,
    stop_after: int | None = None,
) -> list[dict]:
    split_file = resolve_wtq_split_file(dataset_path, split)
    examples = []

    if stop_after is not None and stop_after < 0:
        raise ValueError(f"--limit must be non-negative, got {stop_after}")

    with split_file.open("r", encoding="utf-8", newline="") as data_file:
        reader = csv.DictReader(data_file, delimiter="\t")
        required_fields = {"id", "utterance", "context", "targetValue"}
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError(
                f"Unexpected WTQ fields in {split_file}: {reader.fieldnames}. "
                f"Expected at least: {sorted(required_fields)}"
            )

        for row in reader:
            table_id, table_text = read_wtq_table(dataset_path, row["context"])
            examples.append(
                {
                    "statement": unescape_wtq_value(row["utterance"]),
                    "table_text": table_text,
                    "answer": split_wtq_answer(row["targetValue"]),
                    "_metadata": {
                        "id": row["id"],
                        "dataset": "WikiTQ",
                        "split": split,
                        "table_id": table_id,
                    },
                }
            )
            if stop_after is not None and len(examples) >= stop_after:
                break

    return examples


def load_examples(args: argparse.Namespace) -> list[dict]:
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_path}")

    stop_after = args.limit if not args.shuffle else None
    if args.task == "wtq":
        return load_wtq_examples(
            dataset_path=dataset_path,
            split=args.split,
            stop_after=stop_after,
        )

    raise ValueError(f"Unsupported task: {args.task}")


def prepare_examples(
    examples: list[dict],
    limit: int | None,
    shuffle: bool,
    seed: int,
    include_metadata: bool,
) -> list[dict]:
    if limit is not None and limit < 0:
        raise ValueError(f"--limit must be non-negative, got {limit}")

    selected = list(examples)
    if shuffle:
        random.Random(seed).shuffle(selected)
    if limit is not None:
        selected = selected[:limit]

    output_examples = []
    for example in selected:
        item = {
            "statement": example["statement"],
            "table_text": example["table_text"],
            "answer": example["answer"],
        }
        if include_metadata:
            item.update(example["_metadata"])
        output_examples.append(item)

    return output_examples


def write_jsonl(examples: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for example in examples:
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path).expanduser()

    examples = load_examples(args)
    output_examples = prepare_examples(
        examples=examples,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
        include_metadata=args.include_metadata,
    )
    write_jsonl(output_examples, output_path)

    print(
        f"Converted {len(output_examples)} examples for task '{args.task}' "
        f"from split '{args.split}' to {output_path}"
    )


if __name__ == "__main__":
    main()
