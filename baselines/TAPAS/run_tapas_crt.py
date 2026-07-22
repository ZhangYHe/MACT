#!/usr/bin/env python3
"""Run a local TAPAS checkpoint on CRT-QA data in MACT JSONL format."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import transformers
from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer
from transformers.models.tapas.tokenization_tapas import parse_text


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL_PATH = Path("/home/zhangyunhe/nas/model/tapas-large-finetuned-wtq")
DEFAULT_INPUT_PATH = PROJECT_ROOT / "output" / "crt_answerable.jsonl"
TAPAS_MODEL_MAX_LENGTH = 1024
REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
)
EXPECTED_AGGREGATIONS = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local TAPAS Large WTQ checkpoint on CRT-QA and write "
            "MACT-compatible result JSONL. This script performs inference only."
        )
    )
    parser.add_argument(
        "--model_name_or_path",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Local TAPAS checkpoint directory. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--input_path",
        default=str(DEFAULT_INPUT_PATH),
        help=f"CRT-QA MACT JSONL input. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output_path",
        default="",
        help="Destination results.jsonl. Required unless --validate_only is used.",
    )
    parser.add_argument(
        "--run_config_path",
        default="",
        help="Run metadata JSON. Defaults to run_config.json beside --output_path.",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument(
        "--cell_classification_threshold",
        type=float,
        default=0.5,
        help="Cell selection probability threshold. Default: 0.5.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device inside CUDA_VISIBLE_DEVICES. Default: cuda:0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N examples. Intended for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing output that contains an exact prefix of input IDs.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help=(
            "Validate local files, input schema, and all tokenization without "
            "loading model weights or using CUDA."
        ),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    temporary_path.replace(path)


def validate_model_directory(model_path: Path) -> dict[str, Any]:
    if not model_path.is_dir():
        raise FileNotFoundError(f"TAPAS model directory does not exist: {model_path}")

    missing_files = [name for name in REQUIRED_MODEL_FILES if not (model_path / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            "TAPAS model download is incomplete. Missing required files: "
            + ", ".join(missing_files)
            + ". Only model.safetensors is required; pytorch_model.bin and tf_model.h5 are not used."
        )

    config_path = model_path / "config.json"
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if config.get("model_type") != "tapas":
        raise ValueError(f"Expected a TAPAS config in {config_path}, got {config.get('model_type')!r}.")
    raw_aggregations = config.get("aggregation_labels") or {}
    aggregations = {int(key): str(value) for key, value in raw_aggregations.items()}
    if aggregations != EXPECTED_AGGREGATIONS:
        raise ValueError(
            f"Unexpected TAPAS aggregation labels in {config_path}: {aggregations}. "
            f"Expected {EXPECTED_AGGREGATIONS}."
        )

    weight_path = model_path / "model.safetensors"
    weight_stat = weight_path.stat()
    return {
        "path": str(model_path),
        "weights_file": weight_path.name,
        "weights_size_bytes": weight_stat.st_size,
        "weights_mtime_ns": weight_stat.st_mtime_ns,
        "local_files_only": True,
        "use_safetensors": True,
        "dtype_policy": "checkpoint_and_pytorch_default",
        "aggregation_labels": aggregations,
        "max_position_embeddings": config.get("max_position_embeddings"),
        "max_num_rows": config.get("max_num_rows"),
        "max_num_columns": config.get("max_num_columns"),
    }


def normalize_cell(value: object) -> str:
    return "" if value is None else str(value)


def validate_example(item: object, line_number: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"Input line {line_number} must be a JSON object.")

    example_id = item.get("id")
    if not isinstance(example_id, str) or not example_id.strip():
        raise ValueError(f"Input line {line_number} has a missing or invalid id.")

    statement = item.get("statement")
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError(f"Input line {line_number} ({example_id}) has an invalid statement.")

    table = item.get("table_text")
    if not isinstance(table, list) or not table:
        raise ValueError(f"Input line {line_number} ({example_id}) has no table rows.")
    if not isinstance(table[0], list) or not table[0]:
        raise ValueError(f"Input line {line_number} ({example_id}) has an invalid header row.")

    column_count = len(table[0])
    for row_index, row in enumerate(table):
        if not isinstance(row, list):
            raise ValueError(
                f"Input line {line_number} ({example_id}) table row {row_index} is not a list."
            )
        if len(row) != column_count:
            raise ValueError(
                f"Input line {line_number} ({example_id}) table row {row_index} has "
                f"{len(row)} cells; expected {column_count}."
            )
    return item


def load_examples(input_path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"CRT input JSONL does not exist: {input_path}")

    examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                raw_item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on input line {line_number}: {error}") from error
            item = validate_example(raw_item, line_number)
            example_id = item["id"]
            if example_id in seen_ids:
                raise ValueError(f"Duplicate input id on line {line_number}: {example_id}")
            seen_ids.add(example_id)
            examples.append(item)
            if limit is not None and len(examples) >= limit:
                break

    if not examples:
        raise ValueError(f"CRT input JSONL contains no examples: {input_path}")
    return examples


def example_to_dataframe(item: dict[str, Any]) -> pd.DataFrame:
    table = item["table_text"]
    header = [normalize_cell(cell) for cell in table[0]]
    rows = [[normalize_cell(cell) for cell in row] for row in table[1:]]
    return pd.DataFrame(rows, columns=header, dtype=str)


def load_existing_results(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    if not output_path.is_file():
        raise ValueError(f"Output path exists but is not a file: {output_path}")

    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with output_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in existing output line {line_number}: {error}"
                ) from error
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise ValueError(f"Existing output line {line_number} has no valid id.")
            if item["id"] in seen_ids:
                raise ValueError(f"Duplicate id in existing output: {item['id']}")
            seen_ids.add(item["id"])
            results.append(item)
    return results


def validate_resume_prefix(
    examples: list[dict[str, Any]], existing_results: list[dict[str, Any]]
) -> None:
    if len(existing_results) > len(examples):
        raise ValueError(
            f"Existing output has {len(existing_results)} rows, but this run has only "
            f"{len(examples)} input rows."
        )
    expected_ids = [item["id"] for item in examples[: len(existing_results)]]
    result_ids = [item["id"] for item in existing_results]
    if result_ids != expected_ids:
        mismatch_index = next(
            index
            for index, (result_id, expected_id) in enumerate(zip(result_ids, expected_ids))
            if result_id != expected_id
        )
        raise ValueError(
            "Existing output is not an exact prefix of the current input at index "
            f"{mismatch_index}: output={result_ids[mismatch_index]!r}, "
            f"input={expected_ids[mismatch_index]!r}."
        )
    invalid_status_ids = [
        item["id"] for item in existing_results if item.get("run_status") != "completed"
    ]
    if invalid_status_ids:
        raise ValueError(
            "Existing output contains non-completed rows and cannot be resumed safely: "
            + ", ".join(invalid_status_ids[:5])
        )


def batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def encode_batch(
    batch: list[dict[str, Any]],
    tokenizer: Any,
    max_source_length: int,
) -> tuple[Any, list[pd.DataFrame], list[int], list[int]]:
    encoded_examples = []
    tables = []
    input_lengths = []
    encoded_rows = []

    for item in batch:
        table = example_to_dataframe(item)
        encoded = tokenizer(
            table=table,
            queries=item["statement"],
            truncation="drop_rows_to_fit",
            max_length=max_source_length,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        length = int(sum(encoded["attention_mask"]))
        if length > max_source_length:
            raise RuntimeError(
                f"TAPAS tokenization produced {length} tokens for {item['id']}, "
                f"exceeding max_source_length={max_source_length}."
            )
        row_count = max((token_type[2] for token_type in encoded["token_type_ids"]), default=0)
        encoded_examples.append(encoded)
        tables.append(table)
        input_lengths.append(length)
        encoded_rows.append(int(row_count))

    padded = tokenizer.pad(
        encoded_examples,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return padded, tables, input_lengths, encoded_rows


def extract_numeric_value(cell: str) -> float | None:
    numeric_values = []
    for numeric_span in parse_text(cell):
        for numeric_value in numeric_span.values:
            if numeric_value.float_value is not None:
                numeric_values.append(float(numeric_value.float_value))
    if len(numeric_values) != 1 or not math.isfinite(numeric_values[0]):
        return None
    return numeric_values[0]


def format_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"Cannot format a non-finite aggregation result: {value}")
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    formatted = format(value, ".15g")
    return "0" if formatted in {"-0", "-0.0"} else formatted


def postprocess_prediction(
    aggregation: str,
    cells: list[str],
) -> tuple[object, str, list[float | None]]:
    if not cells:
        return "", "empty_selection", []
    if aggregation == "NONE":
        prediction: object = cells[0] if len(cells) == 1 else list(cells)
        return prediction, "ok", []
    if aggregation == "COUNT":
        return str(len(cells)), "ok", []
    if aggregation not in {"SUM", "AVERAGE"}:
        return "", "unsupported_aggregation", []

    numeric_values = [extract_numeric_value(cell) for cell in cells]
    if any(value is None for value in numeric_values):
        return "", "non_numeric_aggregation", numeric_values
    values = [float(value) for value in numeric_values if value is not None]
    aggregate = sum(values)
    if aggregation == "AVERAGE":
        aggregate /= len(values)
    return format_number(aggregate), "ok", numeric_values


def is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def run_batch(
    batch: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    model_path: Path,
    max_source_length: int,
    cell_classification_threshold: float,
) -> list[dict[str, Any]]:
    encoding, tables, input_lengths, encoded_rows = encode_batch(
        batch=batch,
        tokenizer=tokenizer,
        max_source_length=max_source_length,
    )
    model_inputs = {name: tensor.to(device) for name, tensor in encoding.items()}

    try:
        with torch.inference_mode():
            outputs = model(**model_inputs)
    except RuntimeError as error:
        if is_cuda_oom(error):
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA ran out of memory while processing a batch of "
                f"{len(batch)} examples at the checkpoint's default precision. Reduce "
                "BATCH_SIZE and rerun the same RUN_DIR with --resume; completed batches "
                "are already saved."
            ) from error
        raise

    coordinates_batch, aggregation_indices = tokenizer.convert_logits_to_predictions(
        encoding,
        outputs.logits.detach().cpu(),
        outputs.logits_aggregation.detach().cpu(),
        cell_classification_threshold=cell_classification_threshold,
    )
    if len(coordinates_batch) != len(batch) or len(aggregation_indices) != len(batch):
        raise RuntimeError(
            "TAPAS postprocessing returned an unexpected batch size: "
            f"coordinates={len(coordinates_batch)}, aggregations={len(aggregation_indices)}, "
            f"examples={len(batch)}."
        )

    results = []
    for item, table, coordinates, aggregation_index, input_tokens, kept_rows in zip(
        batch,
        tables,
        coordinates_batch,
        aggregation_indices,
        input_lengths,
        encoded_rows,
    ):
        aggregation = model.config.aggregation_labels[int(aggregation_index)]
        try:
            cells = [str(table.iat[row_index, column_index]) for row_index, column_index in coordinates]
        except IndexError as error:
            raise RuntimeError(
                f"TAPAS returned an out-of-range coordinate for {item['id']}: {coordinates}"
            ) from error
        prediction, postprocess_status, numeric_values = postprocess_prediction(
            aggregation, cells
        )
        raw_answer = ", ".join(cells)
        if aggregation != "NONE":
            raw_answer = f"{aggregation} > {raw_answer}"

        result = dict(item)
        result.update(
            {
                "pred_answer": prediction,
                "pred_answer_all": [prediction],
                "run_status": "completed",
                "tapas_metadata": {
                    "model_name_or_path": str(model_path),
                    "selected_coordinates": [list(coordinate) for coordinate in coordinates],
                    "selected_cells": cells,
                    "aggregation": aggregation,
                    "aggregation_index": int(aggregation_index),
                    "raw_answer": raw_answer,
                    "numeric_values": numeric_values,
                    "postprocess_status": postprocess_status,
                    "input_tokens": input_tokens,
                    "max_source_length": max_source_length,
                    "truncation": "drop_rows_to_fit",
                    "table_rows": len(table),
                    "encoded_rows": kept_rows,
                    "truncated_rows": len(table) - kept_rows,
                    "cell_classification_threshold": cell_classification_threshold,
                },
            }
        )
        results.append(result)
    return results


def append_results(output_file: Any, results: list[dict[str, Any]]) -> None:
    for result in results:
        output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    output_file.flush()
    os.fsync(output_file.fileno())


def validate_final_output(output_path: Path, examples: list[dict[str, Any]]) -> None:
    results = load_existing_results(output_path)
    expected_ids = [item["id"] for item in examples]
    result_ids = [item["id"] for item in results]
    if result_ids != expected_ids:
        raise RuntimeError(
            f"Final output IDs/order do not match input: {len(result_ids)} results for "
            f"{len(expected_ids)} examples."
        )
    missing_predictions = [
        item["id"]
        for item in results
        if "pred_answer" not in item or item.get("run_status") != "completed"
    ]
    if missing_predictions:
        raise RuntimeError(
            "Final output contains incomplete predictions: " + ", ".join(missing_predictions[:5])
        )


def validate_numeric_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be positive, got {args.batch_size}")
    if args.max_source_length <= 0 or args.max_source_length > TAPAS_MODEL_MAX_LENGTH:
        raise ValueError(
            f"--max_source_length must be in [1, {TAPAS_MODEL_MAX_LENGTH}], "
            f"got {args.max_source_length}"
        )
    if not 0.0 <= args.cell_classification_threshold <= 1.0:
        raise ValueError(
            "--cell_classification_threshold must be between 0 and 1, got "
            f"{args.cell_classification_threshold}"
        )
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"--limit must be positive when provided, got {args.limit}")


def build_run_config(
    args: argparse.Namespace,
    model_metadata: dict[str, Any],
    input_path: Path,
    output_path: Path,
    total_examples: int,
) -> dict[str, Any]:
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "status": "running",
        "started_at": utc_now(),
        "model": model_metadata,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_examples": total_examples,
        "inference": {
            "batch_size": args.batch_size,
            "device": args.device,
            "dtype_policy": "checkpoint_and_pytorch_default",
            "max_source_length": args.max_source_length,
            "truncation": "drop_rows_to_fit",
            "cell_classification_threshold": args.cell_classification_threshold,
            "limit": args.limit,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pandas": pd.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_available": torch.cuda.is_available(),
            "visible_cuda_device_count": torch.cuda.device_count(),
            "visible_cuda_device_0": cuda_name,
        },
    }


def validate_tokenization(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    batch_size: int,
    max_source_length: int,
) -> dict[str, int]:
    input_lengths = []
    truncated_tables = 0
    max_rows = 0
    max_columns = 0
    for batch in batched(examples, batch_size):
        _, tables, lengths, encoded_rows = encode_batch(batch, tokenizer, max_source_length)
        input_lengths.extend(lengths)
        for table, kept_rows in zip(tables, encoded_rows):
            max_rows = max(max_rows, len(table))
            max_columns = max(max_columns, len(table.columns))
            if kept_rows < len(table):
                truncated_tables += 1
    return {
        "examples": len(examples),
        "max_input_tokens": max(input_lengths),
        "tables_truncated": truncated_tables,
        "max_data_rows": max_rows,
        "max_columns": max_columns,
    }


def main() -> None:
    args = parse_args()
    validate_numeric_args(args)

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"transformers\.models\.tapas\.tokenization_tapas",
    )

    model_path = Path(args.model_name_or_path).expanduser().resolve()
    input_path = Path(args.input_path).expanduser().resolve()
    model_metadata = validate_model_directory(model_path)
    examples = load_examples(input_path, args.limit)

    print(f"Loading local tokenizer: {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        clean_up_tokenization_spaces=True,
    )

    if args.validate_only:
        diagnostics = validate_tokenization(
            examples,
            tokenizer,
            args.batch_size,
            args.max_source_length,
        )
        print(f"Model files: OK ({model_path})")
        print(f"Input schema and tokenization: OK ({diagnostics})")
        print("Validation only: model weights were not loaded and CUDA was not used.")
        return

    if not args.output_path:
        raise ValueError("--output_path is required unless --validate_only is used.")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is not set. Choose a GPU before running, for example: "
            "CUDA_VISIBLE_DEVICES=2 bash run_crt.sh"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available inside the mact environment. Check the selected "
            "CUDA_VISIBLE_DEVICES value and NVIDIA driver before rerunning."
        )

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"This baseline requires CUDA inference; got --device {args.device!r}.")

    output_path = Path(args.output_path).expanduser().resolve()
    run_config_path = (
        Path(args.run_config_path).expanduser().resolve()
        if args.run_config_path
        else output_path.parent / "run_config.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_results = load_existing_results(output_path)
    if existing_results and not args.resume:
        raise FileExistsError(
            f"Output already contains {len(existing_results)} results: {output_path}. "
            "Use a new RUN_DIR or pass --resume."
        )
    if output_path.exists() and output_path.stat().st_size > 0 and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    validate_resume_prefix(examples, existing_results)

    run_config = build_run_config(
        args, model_metadata, input_path, output_path, len(examples)
    )
    run_config["resumed_from_results"] = len(existing_results)
    write_json(run_config_path, run_config)

    try:
        print(f"Loading local safetensors model at default precision: {model_path}", flush=True)
        model = AutoModelForTableQuestionAnswering.from_pretrained(
            model_path,
            local_files_only=True,
            use_safetensors=True,
        )
        model.to(device)
        model.eval()
        actual_dtype = str(next(model.parameters()).dtype)
        run_config["model"]["actual_dtype"] = actual_dtype
        write_json(run_config_path, run_config)
        print(f"Model dtype: {actual_dtype}", flush=True)

        pending_examples = examples[len(existing_results) :]
        print(
            f"Examples: total={len(examples)}, completed={len(existing_results)}, "
            f"pending={len(pending_examples)}, batch_size={args.batch_size}",
            flush=True,
        )

        output_mode = "a" if existing_results else "w"
        completed = len(existing_results)
        with output_path.open(output_mode, encoding="utf-8") as output_file:
            for batch in batched(pending_examples, args.batch_size):
                batch_results = run_batch(
                    batch=batch,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    model_path=model_path,
                    max_source_length=args.max_source_length,
                    cell_classification_threshold=args.cell_classification_threshold,
                )
                append_results(output_file, batch_results)
                completed += len(batch_results)
                print(f"Completed {completed}/{len(examples)}", flush=True)

        validate_final_output(output_path, examples)
    except BaseException as error:
        run_config["status"] = "failed"
        run_config["finished_at"] = utc_now()
        run_config["error"] = f"{type(error).__name__}: {error}"
        write_json(run_config_path, run_config)
        raise

    run_config["status"] = "completed"
    run_config["finished_at"] = utc_now()
    run_config["completed_results"] = len(examples)
    write_json(run_config_path, run_config)
    print(f"Saved {len(examples)} ordered predictions to {output_path}")


if __name__ == "__main__":
    main()
