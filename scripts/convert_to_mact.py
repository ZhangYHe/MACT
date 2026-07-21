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
DEFAULT_CRT_PATH = "/home/zhangyunhe/nas/dataset/CRT-QA/CRT-QA"
DEFAULT_TAT_PATH = "/home/zhangyunhe/nas/dataset/TAT-QA"
DEFAULT_SCITAB_PATH = "/home/zhangyunhe/nas/dataset/SciTab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert supported table QA datasets to MACT JSONL format."
    )
    parser.add_argument(
        "--task",
        default="wtq",
        choices=["wtq", "crt", "tat", "scitab"],
        help="Dataset conversion task. Default: wtq.",
    )
    parser.add_argument(
        "--dataset_path",
        default=None,
        help=(
            "Path to the source dataset root directory. "
            f"For --task wtq, default: {DEFAULT_WTQ_PATH}. "
            f"For --task crt, default: {DEFAULT_CRT_PATH}. "
            f"For --task tat, default: {DEFAULT_TAT_PATH}. "
            f"For --task scitab, default: {DEFAULT_SCITAB_PATH}"
        ),
    )
    parser.add_argument(
        "--split",
        required=True,
        help=(
            "Dataset split to convert. For WTQ, supported aliases include train, "
            "test, dev, validation, pristine-unseen-tables, pristine-seen-tables, "
            "training-before300, and random-split-{1..5}-{train,dev}. "
            "For CRT-QA, supported splits are all, answerable, and unanswerable. "
            "For TAT-QA, supported splits are train, dev, validation, test, "
            "test_gold, and test-gold. For SciTab, supported split is all."
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
    parser.add_argument(
        "--include_task",
        action="store_true",
        help="Include the converter task name in each output item.",
    )
    parser.add_argument(
        "--exclude_ids_path",
        action="append",
        default=[],
        help=(
            "JSONL file containing ids to exclude. Can be passed multiple times. "
            "Rows without an id are ignored."
        ),
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
                        "dataset": "wtq",
                        "task": "wtq",
                        "split": split,
                        "table_id": table_id,
                    },
                }
            )
            if stop_after is not None and len(examples) >= stop_after:
                break

    return examples


def resolve_crt_data_file(dataset_path: Path, split: str) -> Path:
    split_files = {
        "answerable": "dataset.json",
        "unanswerable": "unanswerable.json",
    }
    if split not in split_files:
        raise ValueError(
            f"Invalid CRT-QA data split '{split}'. Supported data files: "
            "answerable, unanswerable"
        )

    data_file = dataset_path / split_files[split]
    if not data_file.is_file():
        raise FileNotFoundError(f"CRT-QA data file does not exist: {data_file}")
    return data_file


def normalize_crt_row(row: Iterable[str]) -> list[str]:
    return [cell.strip() for cell in row]


def read_crt_table(dataset_path: Path, table_id: str) -> list[list[str]]:
    table_path = dataset_path / "all_csv" / table_id
    if not table_path.is_file():
        raise FileNotFoundError(f"CRT-QA table file does not exist: {table_path}")

    with table_path.open("r", encoding="utf-8", newline="") as table_file:
        rows = [
            normalize_crt_row(row)
            for row in csv.reader(table_file, delimiter="#")
        ]

    if not rows:
        raise ValueError(f"CRT-QA table file is empty: {table_path}")

    return rows


def normalize_crt_answer(answer: object) -> list[str]:
    if answer is None:
        return []
    if isinstance(answer, list):
        return [str(item).strip() for item in answer if str(item).strip()]
    answer_text = str(answer).strip()
    if answer_text == "":
        return []
    return [answer_text]


def load_crt_answerable_examples(
    dataset_path: Path,
    requested_split: str,
    table_cache: dict[str, list[list[str]]],
    stop_after: int | None = None,
) -> list[dict]:
    data_file = resolve_crt_data_file(dataset_path, "answerable")
    with data_file.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected CRT-QA dataset format in {data_file}")

    examples = []
    for table_id, table_examples in data.items():
        if table_id not in table_cache:
            table_cache[table_id] = read_crt_table(dataset_path, table_id)
        for question_index, row in enumerate(table_examples):
            examples.append(
                {
                    "statement": row["Question name"],
                    "table_text": table_cache[table_id],
                    "answer": normalize_crt_answer(row.get("Answer")),
                    "_metadata": {
                        "id": f"crt:answerable:{table_id}:{question_index}",
                        "dataset": "crt",
                        "task": "crt",
                        "split": requested_split,
                        "source_split": "answerable",
                        "table_id": table_id,
                        "title": row.get("Tittle") or row.get("Title"),
                        "directness": row.get("Directness"),
                        "composition_type": row.get("Composition Type"),
                    },
                }
            )
            if stop_after is not None and len(examples) >= stop_after:
                return examples

    return examples


def load_crt_unanswerable_examples(
    dataset_path: Path,
    requested_split: str,
    table_cache: dict[str, list[list[str]]],
    stop_after: int | None = None,
) -> list[dict]:
    data_file = resolve_crt_data_file(dataset_path, "unanswerable")
    with data_file.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    if not isinstance(data, list):
        raise ValueError(f"Unexpected CRT-QA unanswerable format in {data_file}")

    examples = []
    for question_index, row in enumerate(data):
        table_id = row["csv file"]
        if table_id not in table_cache:
            table_cache[table_id] = read_crt_table(dataset_path, table_id)
        examples.append(
            {
                "statement": row["Question name"],
                "table_text": table_cache[table_id],
                "answer": [],
                "_metadata": {
                    "id": f"crt:unanswerable:{table_id}:{question_index}",
                    "dataset": "crt",
                    "task": "crt",
                    "split": requested_split,
                    "source_split": "unanswerable",
                    "table_id": table_id,
                    "title": row.get("Tittle") or row.get("Title"),
                    "directness": row.get("Directness"),
                    "composition_type": row.get("Composition Type"),
                },
            }
        )
        if stop_after is not None and len(examples) >= stop_after:
            return examples

    return examples


def load_crt_examples(
    dataset_path: Path,
    split: str,
    stop_after: int | None = None,
) -> list[dict]:
    if stop_after is not None and stop_after < 0:
        raise ValueError(f"--limit must be non-negative, got {stop_after}")

    if split not in {"all", "answerable", "unanswerable"}:
        raise ValueError(
            "Invalid CRT-QA split "
            f"'{split}'. Supported splits: all, answerable, unanswerable"
        )

    table_cache: dict[str, list[list[str]]] = {}
    if split == "answerable":
        return load_crt_answerable_examples(
            dataset_path=dataset_path,
            requested_split=split,
            table_cache=table_cache,
            stop_after=stop_after,
        )
    if split == "unanswerable":
        return load_crt_unanswerable_examples(
            dataset_path=dataset_path,
            requested_split=split,
            table_cache=table_cache,
            stop_after=stop_after,
        )

    answerable_examples = load_crt_answerable_examples(
        dataset_path=dataset_path,
        requested_split=split,
        table_cache=table_cache,
        stop_after=stop_after,
    )
    if stop_after is not None and len(answerable_examples) >= stop_after:
        return answerable_examples

    remaining = None if stop_after is None else stop_after - len(answerable_examples)
    return answerable_examples + load_crt_unanswerable_examples(
        dataset_path=dataset_path,
        requested_split=split,
        table_cache=table_cache,
        stop_after=remaining,
    )


def resolve_tat_split_file(dataset_path: Path, split: str) -> Path:
    split_map = {
        "train": "tatqa_dataset_train.json",
        "training": "tatqa_dataset_train.json",
        "dev": "tatqa_dataset_dev.json",
        "validation": "tatqa_dataset_dev.json",
        "test": "tatqa_dataset_test.json",
        "test_gold": "tatqa_dataset_test_gold.json",
        "test-gold": "tatqa_dataset_test_gold.json",
    }

    if split not in split_map:
        raise ValueError(
            "Invalid TAT-QA split "
            f"'{split}'. Supported splits: dev, test, test_gold, train, validation"
        )

    split_file = dataset_path / "dataset_raw" / split_map[split]
    if not split_file.is_file():
        raise FileNotFoundError(f"TAT-QA split file does not exist: {split_file}")
    return split_file


def normalize_tat_table(table: object) -> list[list[str]]:
    if not isinstance(table, list):
        raise ValueError(f"Unexpected TAT-QA table value: {table!r}")
    rows = []
    for row in table:
        if not isinstance(row, list):
            raise ValueError(f"Unexpected TAT-QA table row: {row!r}")
        rows.append([str(cell).strip() for cell in row])
    if not rows:
        raise ValueError("TAT-QA table is empty")
    return rows


def format_tat_paragraphs(paragraphs: object) -> str:
    if not isinstance(paragraphs, list):
        return ""

    ordered_paragraphs = sorted(
        paragraphs,
        key=lambda paragraph: paragraph.get("order", 0)
        if isinstance(paragraph, dict)
        else 0,
    )
    formatted = []
    for index, paragraph in enumerate(ordered_paragraphs, start=1):
        if not isinstance(paragraph, dict):
            continue
        order = paragraph.get("order", index)
        text = str(paragraph.get("text", "")).strip()
        if text:
            formatted.append(f"Paragraph {order}: {text}")
    return " ".join(formatted) + (" " if formatted else "")


def normalize_tat_answer(answer: object) -> list[str]:
    if answer is None:
        return []
    if isinstance(answer, list):
        return [str(item).strip() for item in answer if str(item).strip()]
    answer_text = str(answer).strip()
    if answer_text == "":
        return []
    return [answer_text]


def load_tat_examples(
    dataset_path: Path,
    split: str,
    stop_after: int | None = None,
) -> list[dict]:
    if stop_after is not None and stop_after < 0:
        raise ValueError(f"--limit must be non-negative, got {stop_after}")

    split_file = resolve_tat_split_file(dataset_path, split)
    with split_file.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    if not isinstance(data, list):
        raise ValueError(f"Unexpected TAT-QA split format in {split_file}")

    examples = []
    for document_index, document in enumerate(data):
        table = document.get("table", {})
        table_id = table.get("uid", f"document-{document_index}")
        table_text = normalize_tat_table(table.get("table"))
        context_text = format_tat_paragraphs(document.get("paragraphs", []))

        for question in document.get("questions", []):
            question_uid = question.get("uid")
            examples.append(
                {
                    "statement": question["question"],
                    "table_text": table_text,
                    "answer": normalize_tat_answer(question.get("answer")),
                    "_extra": {
                        "text": context_text,
                        "answer_from": question.get("answer_from"),
                        "answer_type": question.get("answer_type"),
                        "rel_paragraph": question.get("rel_paragraphs", []),
                        "answer_scale": question.get("scale", ""),
                    },
                    "_metadata": {
                        "id": question_uid
                        or f"tat:{split}:{table_id}:{question.get('order', len(examples))}",
                        "dataset": "tat",
                        "task": "tat",
                        "split": split,
                        "table_id": table_id,
                        "question_order": question.get("order"),
                        "derivation": question.get("derivation"),
                        "req_comparison": question.get("req_comparison"),
                    },
                }
            )
            if stop_after is not None and len(examples) >= stop_after:
                return examples

    return examples


def resolve_scitab_data_file(dataset_path: Path, split: str) -> Path:
    if split != "all":
        raise ValueError(f"Invalid SciTab split '{split}'. Supported split: all")

    data_file = dataset_path / "dataset" / "sci_tab.json"
    if not data_file.is_file():
        raise FileNotFoundError(f"SciTab data file does not exist: {data_file}")
    return data_file


def normalize_scitab_table(row: dict) -> list[list[str]]:
    column_names = row.get("table_column_names")
    content_values = row.get("table_content_values")

    if not isinstance(column_names, list):
        raise ValueError(f"Unexpected SciTab table_column_names: {column_names!r}")
    if not isinstance(content_values, list):
        raise ValueError(f"Unexpected SciTab table_content_values: {content_values!r}")

    table = [[str(cell).strip() for cell in column_names]]
    for table_row in content_values:
        if not isinstance(table_row, list):
            raise ValueError(f"Unexpected SciTab table row: {table_row!r}")
        table.append([str(cell).strip() for cell in table_row])

    if len(table) == 1:
        raise ValueError("SciTab table has no content rows")
    return table


def load_scitab_examples(
    dataset_path: Path,
    split: str,
    stop_after: int | None = None,
) -> list[dict]:
    if stop_after is not None and stop_after < 0:
        raise ValueError(f"--limit must be non-negative, got {stop_after}")

    data_file = resolve_scitab_data_file(dataset_path, split)
    with data_file.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    if not isinstance(data, list):
        raise ValueError(f"Unexpected SciTab data format in {data_file}")

    examples = []
    for row_index, row in enumerate(data):
        label = str(row.get("label", "")).strip()
        if label == "":
            raise ValueError(f"Missing SciTab label at index {row_index}")

        examples.append(
            {
                "statement": (
                    f"Claim: {row['claim']}\n"
                    "Question: Is the above claim supported, refuted, "
                    "or not enough info?"
                ),
                "table_text": normalize_scitab_table(row),
                "answer": [label],
                "_extra": {
                    "text": f"Table caption: {str(row.get('table_caption', '')).strip()}",
                },
                "_metadata": {
                    "id": row.get("id", f"scitab:{row_index}"),
                    "dataset": "scitab",
                    "task": "scitab",
                    "split": "all",
                    "table_id": row.get("table_id"),
                    "paper": row.get("paper"),
                    "paper_id": row.get("paper_id"),
                },
            }
        )
        if stop_after is not None and len(examples) >= stop_after:
            return examples

    return examples


def load_examples(args: argparse.Namespace) -> list[dict]:
    if args.dataset_path is None:
        default_paths = {
            "crt": DEFAULT_CRT_PATH,
            "scitab": DEFAULT_SCITAB_PATH,
            "tat": DEFAULT_TAT_PATH,
            "wtq": DEFAULT_WTQ_PATH,
        }
        dataset_path = Path(default_paths[args.task])
    else:
        dataset_path = Path(args.dataset_path).expanduser()
    dataset_path = dataset_path.resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_path}")

    stop_after = args.limit if not args.shuffle else None
    if args.task == "wtq":
        return load_wtq_examples(
            dataset_path=dataset_path,
            split=args.split,
            stop_after=stop_after,
        )
    if args.task == "crt":
        return load_crt_examples(
            dataset_path=dataset_path,
            split=args.split,
            stop_after=stop_after,
        )
    if args.task == "tat":
        return load_tat_examples(
            dataset_path=dataset_path,
            split=args.split,
            stop_after=stop_after,
        )
    if args.task == "scitab":
        return load_scitab_examples(
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
    task: str | None = None,
    include_task: bool = False,
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    if limit is not None and limit < 0:
        raise ValueError(f"--limit must be non-negative, got {limit}")

    exclude_ids = exclude_ids or set()
    selected = [
        example
        for example in examples
        if str(example.get("_metadata", {}).get("id", "")) not in exclude_ids
    ]
    if shuffle:
        random.Random(seed).shuffle(selected)
    if limit is not None:
        if len(selected) < limit:
            raise ValueError(
                f"Requested {limit} examples, but only {len(selected)} remain "
                "after applying exclusions."
            )
        selected = selected[:limit]

    output_examples = []
    for example in selected:
        item = {
            "statement": example["statement"],
            "table_text": example["table_text"],
            "answer": example["answer"],
        }
        item.update(example.get("_extra", {}))
        if include_metadata:
            item.update(example["_metadata"])
        if include_task:
            item["task"] = task
        output_examples.append(item)

    return output_examples


def load_excluded_ids(paths: list[str]) -> set[str]:
    excluded_ids = set()
    for path_text in paths:
        path = Path(path_text).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Exclude ids file does not exist: {path}")
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path} at line {line_number}: {exc}"
                    ) from exc
                item_id = item.get("id")
                if item_id is not None:
                    excluded_ids.add(str(item_id))
    return excluded_ids


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
        task=args.task,
        include_task=args.include_task,
        exclude_ids=load_excluded_ids(args.exclude_ids_path),
    )
    write_jsonl(output_examples, output_path)

    print(
        f"Converted {len(output_examples)} examples for task '{args.task}' "
        f"from split '{args.split}' to {output_path}"
    )


if __name__ == "__main__":
    main()
