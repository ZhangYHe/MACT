#!/usr/bin/env python3
"""Create a mixed prompt-dev set for WTQ, CRT-QA, and SciTab."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from convert_to_mact import (
    DEFAULT_CRT_PATH,
    DEFAULT_SCITAB_PATH,
    DEFAULT_WTQ_PATH,
    load_examples,
    load_excluded_ids,
    prepare_examples,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one mixed prompt-dev JSONL with 25 examples per task."
    )
    parser.add_argument(
        "--output_path",
        default="output/prompt_dev/mixed_prompt_dev_75.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--manifest_path",
        default="output/prompt_dev/mixed_prompt_dev_75_manifest.json",
        help="Output manifest JSON path.",
    )
    parser.add_argument(
        "--per_task_limit",
        type=int,
        default=25,
        help="Number of examples to sample for each task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for per-task and final mixed ordering.",
    )
    parser.add_argument(
        "--wtq_dataset_path",
        default=DEFAULT_WTQ_PATH,
        help="WTQ dataset root.",
    )
    parser.add_argument(
        "--crt_dataset_path",
        default=DEFAULT_CRT_PATH,
        help="CRT-QA dataset root.",
    )
    parser.add_argument(
        "--scitab_dataset_path",
        default=DEFAULT_SCITAB_PATH,
        help="SciTab dataset root.",
    )
    parser.add_argument(
        "--wtq_exclude_ids_path",
        default="output/wtq_test_random_50.jsonl",
        help="Existing WTQ JSONL whose ids should be excluded.",
    )
    return parser.parse_args()


def sample_task_examples(
    *,
    task: str,
    dataset_path: Path,
    split: str,
    limit: int,
    seed: int,
    exclude_ids: set[str],
) -> list[dict]:
    load_args = argparse.Namespace(
        task=task,
        dataset_path=str(dataset_path),
        split=split,
        limit=None,
        shuffle=True,
    )
    examples = load_examples(load_args)
    return prepare_examples(
        examples=examples,
        limit=limit,
        shuffle=True,
        seed=seed,
        include_metadata=True,
        task=task,
        include_task=True,
        exclude_ids=exclude_ids,
    )


def main() -> None:
    args = parse_args()
    if args.per_task_limit <= 0:
        raise ValueError("--per_task_limit must be a positive integer")

    output_path = resolve_project_path(args.output_path)
    manifest_path = resolve_project_path(args.manifest_path)
    wtq_exclude_path = resolve_project_path(args.wtq_exclude_ids_path)

    specs = [
        {
            "task": "wtq",
            "dataset_path": Path(args.wtq_dataset_path).expanduser().resolve(),
            "split": "test",
            "exclude_ids_paths": [wtq_exclude_path],
        },
        {
            "task": "crt",
            "dataset_path": Path(args.crt_dataset_path).expanduser().resolve(),
            "split": "dataset",
            "exclude_ids_paths": [],
        },
        {
            "task": "scitab",
            "dataset_path": Path(args.scitab_dataset_path).expanduser().resolve(),
            "split": "all",
            "exclude_ids_paths": [],
        },
    ]

    mixed_examples = []
    source_manifest = {}
    for spec in specs:
        exclude_ids = load_excluded_ids(
            [str(path) for path in spec["exclude_ids_paths"]]
        )
        task_examples = sample_task_examples(
            task=spec["task"],
            dataset_path=spec["dataset_path"],
            split=spec["split"],
            limit=args.per_task_limit,
            seed=args.seed,
            exclude_ids=exclude_ids,
        )
        mixed_examples.extend(task_examples)
        source_manifest[spec["task"]] = {
            "dataset_path": str(spec["dataset_path"]),
            "split": spec["split"],
            "count": len(task_examples),
            "ids": [item["id"] for item in task_examples],
            "exclude_ids_paths": [str(path) for path in spec["exclude_ids_paths"]],
            "excluded_id_count": len(exclude_ids),
        }

    random.Random(args.seed).shuffle(mixed_examples)
    write_jsonl(mixed_examples, output_path)

    task_counts = Counter(item["task"] for item in mixed_examples)
    manifest = {
        "seed": args.seed,
        "per_task_limit": args.per_task_limit,
        "total_count": len(mixed_examples),
        "task_counts": dict(sorted(task_counts.items())),
        "output_path": str(output_path),
        "sources": source_manifest,
        "output_order": [
            {
                "task": item["task"],
                "id": item["id"],
                "dataset": item.get("dataset"),
                "split": item.get("split"),
                "table_id": item.get("table_id"),
            }
            for item in mixed_examples
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print(f"Wrote {len(mixed_examples)} examples to {output_path}")
    print(f"Task counts: {dict(sorted(task_counts.items()))}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
