#!/usr/bin/env python3
"""Run a local TAPEX checkpoint on CRT-QA data in MACT JSONL format."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.deprecated.tapex.tokenization_tapex import (
    TapexTruncationStrategy,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL_PATH = Path("/home/zhangyunhe/nas/model/tapex-large-finetuned-wtq")
DEFAULT_INPUT_PATH = PROJECT_ROOT / "output" / "crt_answerable.jsonl"
TAPEX_MODEL_MAX_LENGTH = 1024
REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "merges.txt",
    "tokenizer_config.json",
    "vocab.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local TAPEX Large checkpoint on CRT-QA and write MACT-compatible "
            "result JSONL. This script performs inference only."
        )
    )
    parser.add_argument(
        "--model_name_or_path",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Local TAPEX checkpoint directory. Default: {DEFAULT_MODEL_PATH}",
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
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
        help="Validate local files and input schema without loading the model or using CUDA.",
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
        raise FileNotFoundError(f"TAPEX model directory does not exist: {model_path}")

    missing_files = [name for name in REQUIRED_MODEL_FILES if not (model_path / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"TAPEX model directory is missing required files: {', '.join(missing_files)}"
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
    if limit is not None and limit < 0:
        raise ValueError(f"--limit must be non-negative, got {limit}")

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

    if not examples and limit != 0:
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


def is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def encode_batch(
    batch: list[dict[str, Any]],
    tokenizer: Any,
    max_source_length: int,
    seed: int,
) -> Any:
    """Apply TAPEX row truncation per table, then pad examples into one batch.

    Transformers 4.44.1 routes the TAPEX-specific ``drop_rows_to_fit`` value
    through the generic truncation enum in ``TapexTokenizer.__call__``. Calling
    ``prepare_table_query`` directly preserves TAPEX's intended row-level
    truncation, while ordinary ``prepare_for_model`` and ``pad`` handle the
    already-truncated token IDs.
    """

    encoded_examples = []
    special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
    row_truncation_budget = max_source_length - special_token_count
    if row_truncation_budget <= 0:
        raise ValueError(
            f"max_source_length={max_source_length} leaves no room after "
            f"{special_token_count} special tokens."
        )
    for item in batch:
        table = example_to_dataframe(item)
        question = item["statement"]

        # TAPEX may randomly remove unrelated rows. Make that choice stable per
        # example so changing batch size or resuming cannot change predictions.
        random.seed(f"{seed}:{item['id']}")
        joint_input = tokenizer.prepare_table_query(
            table,
            question,
            answer=None,
            truncation_strategy=TapexTruncationStrategy.DROP_ROWS_TO_FIT,
            # TAPEX's estimator counts content tokens but omits the BART
            # special tokens added below. Reserve their positions explicitly.
            max_length=row_truncation_budget,
        )
        if tokenizer.do_lower_case:
            joint_input = joint_input.lower()
        tokens = tokenizer.tokenize(joint_input)
        encoded = tokenizer.prepare_for_model(
            ids=tokenizer.convert_tokens_to_ids(tokens),
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        if len(encoded["input_ids"]) > max_source_length:
            raise RuntimeError(
                f"TAPEX row truncation produced {len(encoded['input_ids'])} tokens for "
                f"{item['id']}, exceeding max_source_length={max_source_length}."
            )
        encoded_examples.append(encoded)

    return tokenizer.pad(
        encoded_examples,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )


def run_batch(
    batch: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    model_path: Path,
    max_source_length: int,
    max_new_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    encoding = encode_batch(
        batch=batch,
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        seed=seed,
    )
    input_lengths = encoding["attention_mask"].sum(dim=1).tolist()
    model_inputs = {name: tensor.to(device) for name, tensor in encoding.items()}

    try:
        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
            )
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

    predictions = [
        answer.strip()
        for answer in tokenizer.batch_decode(generated, skip_special_tokens=True)
    ]
    if len(predictions) != len(batch):
        raise RuntimeError(
            f"Model returned {len(predictions)} predictions for a batch of {len(batch)} examples."
        )

    results = []
    for item, prediction, input_tokens in zip(batch, predictions, input_lengths):
        result = dict(item)
        result.update(
            {
                "pred_answer": prediction,
                "pred_answer_all": [prediction],
                "run_status": "completed",
                "tapex_metadata": {
                    "model_name_or_path": str(model_path),
                    "input_tokens": int(input_tokens),
                    "max_source_length": max_source_length,
                    "truncation": "drop_rows_to_fit",
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "num_beams": 1,
                    "seed": seed,
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


def build_run_config(
    args: argparse.Namespace,
    model_metadata: dict[str, Any],
    input_path: Path,
    output_path: Path,
    total_examples: int,
) -> dict[str, Any]:
    cuda_name = None
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(0)
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
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "seed": args.seed,
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


def validate_numeric_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "max_source_length", "max_new_tokens"):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name} must be positive, got {value}")
    if args.max_source_length > TAPEX_MODEL_MAX_LENGTH:
        raise ValueError(
            f"--max_source_length cannot exceed TAPEX's {TAPEX_MODEL_MAX_LENGTH}-token "
            f"position limit, got {args.max_source_length}"
        )
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"--limit must be positive when provided, got {args.limit}")


def main() -> None:
    args = parse_args()
    validate_numeric_args(args)

    model_path = Path(args.model_name_or_path).expanduser().resolve()
    input_path = Path(args.input_path).expanduser().resolve()
    model_metadata = validate_model_directory(model_path)
    examples = load_examples(input_path, args.limit)

    if args.validate_only:
        first_table = example_to_dataframe(examples[0]) if examples else pd.DataFrame()
        print(f"Model files: OK ({model_path})")
        print(f"Input schema: OK ({len(examples)} examples)")
        if examples:
            print(
                f"First table: {first_table.shape[0]} data rows x "
                f"{first_table.shape[1]} columns"
            )
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
        print(f"Loading local tokenizer: {model_path}", flush=True)
        # TAPEX in Transformers 4.44.1 emits repetitive compatibility warnings
        # from its internal row-length estimator. They do not affect encoding.
        transformers.logging.set_verbosity_error()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            clean_up_tokenization_spaces=True,
        )
        print(
            f"Loading local safetensors model at checkpoint/PyTorch default precision: "
            f"{model_path}",
            flush=True,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
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
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
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
