#!/usr/bin/env python3
"""Run the local OmniTab WTQ checkpoint on CRT-QA without training."""

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
DEFAULT_MODEL_PATH = Path("/home/zhangyunhe/nas/model/omnitab-large-finetuned-wtq")
DEFAULT_INPUT_PATH = PROJECT_ROOT / "output" / "crt_answerable.jsonl"
MODEL_MAX_LENGTH = 1024
REQUIRED_MODEL_FILES = (
    "config.json",
    "pytorch_model.bin",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local OmniTab Large WTQ inference on CRT-QA in MACT format."
    )
    parser.add_argument("--model_name_or_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--input_path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output_path", default="")
    parser.add_argument("--run_config_path", default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate files, schema, and tokenization without loading weights or CUDA.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    temporary.replace(path)


def validate_model_directory(model_path: Path) -> dict[str, Any]:
    if not model_path.is_dir():
        raise FileNotFoundError(f"OmniTab model directory does not exist: {model_path}")
    missing = [name for name in REQUIRED_MODEL_FILES if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "OmniTab download is incomplete. Missing: "
            + ", ".join(missing)
            + ". The official checkpoint uses pytorch_model.bin, not safetensors."
        )
    with (model_path / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if config.get("model_type") != "bart":
        raise ValueError(f"Expected a BART OmniTab config, got {config.get('model_type')!r}.")
    weight = model_path / "pytorch_model.bin"
    stat = weight.stat()
    return {
        "path": str(model_path),
        "weights_file": weight.name,
        "weights_size_bytes": stat.st_size,
        "weights_mtime_ns": stat.st_mtime_ns,
        "local_files_only": True,
        "use_safetensors": False,
        "dtype_policy": "checkpoint_and_pytorch_default",
        "config_torch_dtype": config.get("torch_dtype"),
    }


def normalize_cell(value: object) -> str:
    return "" if value is None else str(value)


def validate_example(raw: object, line_number: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Input line {line_number} must be a JSON object.")
    example_id = raw.get("id")
    if not isinstance(example_id, str) or not example_id.strip():
        raise ValueError(f"Input line {line_number} has an invalid id.")
    if not isinstance(raw.get("statement"), str) or not raw["statement"].strip():
        raise ValueError(f"Input line {line_number} ({example_id}) has an invalid statement.")
    table = raw.get("table_text")
    if not isinstance(table, list) or len(table) < 2 or not isinstance(table[0], list):
        raise ValueError(f"Input line {line_number} ({example_id}) has an invalid table.")
    width = len(table[0])
    if width == 0:
        raise ValueError(f"Input line {line_number} ({example_id}) has an empty header.")
    for row_index, row in enumerate(table):
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(
                f"Input line {line_number} ({example_id}) row {row_index} does not "
                f"match the {width}-column header."
            )
    return raw


def load_examples(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"CRT input JSONL does not exist: {path}")
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                item = validate_example(json.loads(line), line_number)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on input line {line_number}: {error}") from error
            if item["id"] in seen:
                raise ValueError(f"Duplicate input id: {item['id']}")
            seen.add(item["id"])
            examples.append(item)
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def example_to_dataframe(item: dict[str, Any]) -> pd.DataFrame:
    table = item["table_text"]
    header = [normalize_cell(cell) for cell in table[0]]
    rows = [[normalize_cell(cell) for cell in row] for row in table[1:]]
    return pd.DataFrame(rows, columns=header, dtype=str)


def batched(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def encode_batch(
    batch: list[dict[str, Any]], tokenizer: Any, max_source_length: int, seed: int
) -> tuple[Any, list[int]]:
    encoded_examples = []
    input_lengths = []
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    content_budget = max_source_length - special_tokens
    if content_budget <= 0:
        raise ValueError("max_source_length leaves no space for table content.")
    for item in batch:
        random.seed(f"{seed}:{item['id']}")
        joint_input = tokenizer.prepare_table_query(
            example_to_dataframe(item),
            item["statement"],
            answer=None,
            truncation_strategy=TapexTruncationStrategy.DROP_ROWS_TO_FIT,
            max_length=content_budget,
        )
        if tokenizer.do_lower_case:
            joint_input = joint_input.lower()
        tokens = tokenizer.tokenize(joint_input)
        encoded = tokenizer.prepare_for_model(
            tokenizer.convert_tokens_to_ids(tokens),
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        length = len(encoded["input_ids"])
        if length > max_source_length:
            raise RuntimeError(
                f"OmniTab tokenization produced {length} tokens for {item['id']}; "
                f"limit is {max_source_length}."
            )
        encoded_examples.append(encoded)
        input_lengths.append(length)
    padded = tokenizer.pad(
        encoded_examples, padding=True, return_attention_mask=True, return_tensors="pt"
    )
    return padded, input_lengths


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
    encoding, input_lengths = encode_batch(batch, tokenizer, max_source_length, seed)
    inputs = {name: tensor.to(device) for name, tensor in encoding.items()}
    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens
            )
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM at OmniTab's default precision. Lower BATCH_SIZE and resume "
                "the same RUN_DIR; completed batches are already saved."
            ) from error
        raise
    predictions = [
        text.strip() for text in tokenizer.batch_decode(generated, skip_special_tokens=True)
    ]
    if len(predictions) != len(batch):
        raise RuntimeError("OmniTab returned a different number of predictions than inputs.")
    results = []
    for item, prediction, input_tokens in zip(batch, predictions, input_lengths):
        result = dict(item)
        result.update(
            {
                "pred_answer": prediction,
                "pred_answer_all": [prediction],
                "run_status": "completed",
                "omnitab_metadata": {
                    "model_name_or_path": str(model_path),
                    "input_tokens": input_tokens,
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


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    results = []
    seen = set()
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid existing output line {line_number}: {error}") from error
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise ValueError(f"Existing output line {line_number} has no valid id.")
            if item["id"] in seen:
                raise ValueError(f"Duplicate existing output id: {item['id']}")
            seen.add(item["id"])
            results.append(item)
    return results


def validate_resume_prefix(
    examples: list[dict[str, Any]], existing: list[dict[str, Any]]
) -> None:
    if len(existing) > len(examples):
        raise ValueError("Existing output is longer than the selected input.")
    expected = [item["id"] for item in examples[: len(existing)]]
    actual = [item["id"] for item in existing]
    if actual != expected:
        raise ValueError("Existing output IDs are not an exact input prefix.")
    if any(item.get("run_status") != "completed" for item in existing):
        raise ValueError("Existing output contains non-completed rows.")


def append_results(output_file: Any, results: list[dict[str, Any]]) -> None:
    for result in results:
        output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    output_file.flush()
    os.fsync(output_file.fileno())


def validate_final_output(path: Path, examples: list[dict[str, Any]]) -> None:
    results = load_existing_results(path)
    if [item["id"] for item in results] != [item["id"] for item in examples]:
        raise RuntimeError("Final result count, IDs, or order do not match the input.")
    if any("pred_answer" not in item or item.get("run_status") != "completed" for item in results):
        raise RuntimeError("Final output contains incomplete results.")


def validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "max_source_length", "max_new_tokens"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive.")
    if args.max_source_length > MODEL_MAX_LENGTH:
        raise ValueError(f"--max_source_length cannot exceed {MODEL_MAX_LENGTH}.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    model_path = Path(args.model_name_or_path).expanduser().resolve()
    input_path = Path(args.input_path).expanduser().resolve()
    model_metadata = validate_model_directory(model_path)
    examples = load_examples(input_path, args.limit)

    transformers.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, clean_up_tokenization_spaces=True
    )
    if args.validate_only:
        lengths = []
        for batch in batched(examples, args.batch_size):
            _, batch_lengths = encode_batch(batch, tokenizer, args.max_source_length, args.seed)
            lengths.extend(batch_lengths)
        print(f"Model files: OK ({model_path})")
        print(f"Input/tokenization: OK ({len(examples)} examples, max={max(lengths)} tokens)")
        print("Validation only: model weights were not loaded and CUDA was not used.")
        return

    if not args.output_path:
        raise ValueError("--output_path is required unless --validate_only is used.")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError("CUDA_VISIBLE_DEVICES is not set; choose a GPU before running.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the mact environment.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("OmniTab baseline requires a CUDA device.")

    output_path = Path(args.output_path).expanduser().resolve()
    config_path = (
        Path(args.run_config_path).expanduser().resolve()
        if args.run_config_path
        else output_path.parent / "run_config.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_results(output_path)
    if output_path.exists() and output_path.stat().st_size and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_path}; use --resume or a new RUN_DIR.")
    validate_resume_prefix(examples, existing)

    run_config = {
        "status": "running",
        "started_at": utc_now(),
        "model": model_metadata,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_examples": len(examples),
        "resumed_from_results": len(existing),
        "inference": {
            "batch_size": args.batch_size,
            "device": args.device,
            "dtype_policy": "checkpoint_and_pytorch_default",
            "max_source_length": args.max_source_length,
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
            "visible_cuda_device_count": torch.cuda.device_count(),
            "visible_cuda_device_0": torch.cuda.get_device_name(0),
        },
    }
    write_json(config_path, run_config)
    try:
        print(f"Loading local OmniTab at default precision: {model_path}", flush=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, local_files_only=True, use_safetensors=False
        )
        model.to(device)
        model.eval()
        actual_dtype = str(next(model.parameters()).dtype)
        run_config["model"]["actual_dtype"] = actual_dtype
        write_json(config_path, run_config)
        print(f"Model dtype: {actual_dtype}", flush=True)

        pending = examples[len(existing) :]
        completed = len(existing)
        mode = "a" if existing else "w"
        with output_path.open(mode, encoding="utf-8") as output_file:
            for batch in batched(pending, args.batch_size):
                results = run_batch(
                    batch,
                    tokenizer,
                    model,
                    device,
                    model_path,
                    args.max_source_length,
                    args.max_new_tokens,
                    args.seed,
                )
                append_results(output_file, results)
                completed += len(results)
                print(f"Completed {completed}/{len(examples)}", flush=True)
        validate_final_output(output_path, examples)
    except BaseException as error:
        run_config.update(
            status="failed",
            finished_at=utc_now(),
            error=f"{type(error).__name__}: {error}",
        )
        write_json(config_path, run_config)
        raise
    run_config.update(
        status="completed", finished_at=utc_now(), completed_results=len(examples)
    )
    write_json(config_path, run_config)
    print(f"Saved {len(examples)} ordered predictions to {output_path}")


if __name__ == "__main__":
    main()
