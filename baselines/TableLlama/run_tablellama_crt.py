#!/usr/bin/env python3
"""Run the local TableLlama checkpoint on CRT-QA without training."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import transformers
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL_PATH = Path("/home/zhangyunhe/nas/model/TableLlama")
DEFAULT_INPUT_PATH = PROJECT_ROOT / "output" / "crt_answerable.jsonl"
DEFAULT_MODEL_MAX_LENGTH = 4096
DEFAULT_MAX_INPUT_TOKENS = 3968
DEFAULT_MAX_NEW_TOKENS = 128
EXPECTED_MODEL_TYPE = "llama"
REQUIRED_MODEL_FILES = (
    "config.json",
    "pytorch_model.bin.index.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer_config.json",
)

PROMPT_PREFIX = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that
appropriately completes the request.

### Instruction:
This is a table QA task. The goal of this task is to answer the question given the table.

### Input:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local TableLlama-7B inference on CRT-QA in MACT format."
    )
    parser.add_argument("--model_name_or_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--input_path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output_path", default="")
    parser.add_argument("--run_config_path", default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_model_length", type=int, default=DEFAULT_MODEL_MAX_LENGTH)
    parser.add_argument("--max_input_tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate model files, input schema, and prompts without loading weights or CUDA.",
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
        raise FileNotFoundError(f"TableLlama model directory does not exist: {model_path}")
    missing = [name for name in REQUIRED_MODEL_FILES if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "TableLlama download is incomplete. Missing: " + ", ".join(missing)
        )

    with (model_path / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError(
            f"Expected a Llama TableLlama config, got {config.get('model_type')!r}."
        )

    index_path = model_path / "pytorch_model.bin.index.json"
    with index_path.open(encoding="utf-8") as index_file:
        index = json.load(index_file)
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names:
        raise ValueError("TableLlama weight index contains no shards.")
    missing_shards = [name for name in shard_names if not (model_path / name).is_file()]
    if missing_shards:
        raise FileNotFoundError(
            "TableLlama download is incomplete. Missing weight shards: "
            + ", ".join(missing_shards)
        )
    shard_sizes = {name: (model_path / name).stat().st_size for name in shard_names}
    indexed_size = index.get("metadata", {}).get("total_size")
    if isinstance(indexed_size, int) and indexed_size <= 0:
        raise ValueError("TableLlama weight index reports an invalid total size.")

    return {
        "path": str(model_path),
        "weights_index": index_path.name,
        "weight_shards": shard_sizes,
        "weights_size_bytes": sum(shard_sizes.values()),
        "indexed_parameter_bytes": indexed_size,
        "local_files_only": True,
        "engine": "vllm",
        "load_format": "auto",
        "dtype_policy": "vllm_auto_from_checkpoint_config",
        "config_torch_dtype": config.get("torch_dtype"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "rope_scaling": config.get("rope_scaling"),
    }


def normalize_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")


def validate_example(raw: object, line_number: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Input line {line_number} must be a JSON object.")
    example_id = raw.get("id")
    if not isinstance(example_id, str) or not example_id.strip():
        raise ValueError(f"Input line {line_number} has an invalid id.")
    statement = raw.get("statement")
    if not isinstance(statement, str) or not statement.strip():
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


def serialize_table(table: list[list[object]], rows_to_keep: int | None = None) -> str:
    header = table[0]
    rows = table[1:] if rows_to_keep is None else table[1 : rows_to_keep + 1]
    pieces = ["[TAB] col: | ", " | ".join(normalize_cell(cell) for cell in header)]
    pieces.append(" | [SEP]")
    for row in rows:
        pieces.extend((" | ", " | ".join(normalize_cell(cell) for cell in row), " | [SEP]"))
    return "".join(pieces)


def build_prompt(item: dict[str, Any], rows_to_keep: int | None = None) -> str:
    table_input = serialize_table(item["table_text"], rows_to_keep)
    return (
        f"{PROMPT_PREFIX}{table_input}\n\n"
        f"### Question:\n{item['statement']}\n\n"
        "### Response:\n"
    )


def token_count(tokenizer: Any, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=True))


def truncate_prompt_to_fit(
    item: dict[str, Any], tokenizer: Any, max_input_tokens: int
) -> tuple[str, int, int, int]:
    total_rows = len(item["table_text"]) - 1
    full_prompt = build_prompt(item, total_rows)
    full_length = token_count(tokenizer, full_prompt)
    if full_length <= max_input_tokens:
        return full_prompt, full_length, total_rows, 0

    low = 0
    high = total_rows
    best: tuple[str, int, int] | None = None
    while low <= high:
        kept_rows = (low + high) // 2
        prompt = build_prompt(item, kept_rows)
        length = token_count(tokenizer, prompt)
        if length <= max_input_tokens:
            best = (prompt, length, kept_rows)
            low = kept_rows + 1
        else:
            high = kept_rows - 1
    if best is None:
        raise ValueError(
            f"TableLlama prompt for {item['id']} exceeds {max_input_tokens} tokens "
            "even after all data rows are removed. Increase MAX_INPUT_TOKENS."
        )
    prompt, length, kept_rows = best
    return prompt, length, kept_rows, total_rows - kept_rows


def batched(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def is_cuda_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "cuda out of memory",
            "out of memory",
            "no available memory for the cache blocks",
            "no available memory for cache blocks",
        )
    )


def oom_error(error: BaseException) -> RuntimeError:
    return RuntimeError(
        "TableLlama CUDA OOM at the checkpoint's default precision. Lower BATCH_SIZE "
        "(try 8, then 4) and resume with the same RUN_DIR --resume. If model "
        "initialization fails on an otherwise-free 24 GB GPU, expose two GPUs and set "
        "TENSOR_PARALLEL_SIZE=2."
    )


def run_batch(
    batch: list[dict[str, Any]],
    tokenizer: Any,
    engine: Any,
    sampling_params: Any,
    model_path: Path,
    max_model_length: int,
    max_input_tokens: int,
    max_new_tokens: int,
    seed: int,
    actual_dtype: str,
) -> list[dict[str, Any]]:
    prepared = [
        truncate_prompt_to_fit(item, tokenizer, max_input_tokens) for item in batch
    ]
    prompts = [entry[0] for entry in prepared]
    try:
        generated = engine.generate(prompts, sampling_params, use_tqdm=False)
    except BaseException as error:
        if is_cuda_oom(error):
            raise oom_error(error) from error
        raise
    if len(generated) != len(batch):
        raise RuntimeError(
            f"TableLlama returned {len(generated)} outputs for {len(batch)} inputs."
        )

    results = []
    for item, request_output, prompt_info in zip(batch, generated, prepared):
        candidates = getattr(request_output, "outputs", None)
        if not candidates:
            raise RuntimeError(f"TableLlama returned no completion for {item['id']}.")
        raw_response = str(candidates[0].text)
        prediction = raw_response.strip()
        _, input_tokens, kept_rows, truncated_rows = prompt_info
        result = dict(item)
        result.update(
            {
                "pred_answer": prediction,
                "pred_answer_all": [prediction],
                "run_status": "completed",
                "tablellama_metadata": {
                    "model_name_or_path": str(model_path),
                    "actual_dtype": actual_dtype,
                    "raw_response": raw_response,
                    "input_tokens": input_tokens,
                    "total_data_rows": len(item["table_text"]) - 1,
                    "kept_data_rows": kept_rows,
                    "truncated_data_rows": truncated_rows,
                    "table_serialization": "tablellama_tab_sep",
                    "max_model_length": max_model_length,
                    "max_input_tokens": max_input_tokens,
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                    "top_p": 1.0,
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
    seen: set[str] = set()
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
    for name in (
        "batch_size",
        "max_model_length",
        "max_input_tokens",
        "max_new_tokens",
        "tensor_parallel_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive.")
    if args.max_input_tokens + args.max_new_tokens > args.max_model_length:
        raise ValueError("--max_input_tokens + --max_new_tokens cannot exceed --max_model_length.")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu_memory_utilization must be in (0, 1].")
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
        model_path, local_files_only=True, use_fast=False
    )
    if args.validate_only:
        lengths = []
        truncated_examples = 0
        truncated_rows = 0
        for item in examples:
            _, length, _, removed = truncate_prompt_to_fit(
                item, tokenizer, args.max_input_tokens
            )
            lengths.append(length)
            truncated_examples += bool(removed)
            truncated_rows += removed
        print(f"Model files: OK ({model_path})")
        print(
            f"Input/prompts: OK ({len(examples)} examples, max={max(lengths)} tokens, "
            f"truncated_examples={truncated_examples}, truncated_rows={truncated_rows})"
        )
        print("Validation only: model weights were not loaded and CUDA was not used.")
        return

    if not args.output_path:
        raise ValueError("--output_path is required unless --validate_only is used.")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError("CUDA_VISIBLE_DEVICES is not set; choose GPU(s) before running.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the mact environment.")
    if torch.cuda.device_count() < args.tensor_parallel_size:
        raise RuntimeError(
            f"tensor_parallel_size={args.tensor_parallel_size} needs at least that many "
            f"visible GPUs; found {torch.cuda.device_count()}."
        )

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

    import vllm
    from vllm import LLM, SamplingParams

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
            "max_model_length": args.max_model_length,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0,
            "top_p": 1.0,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "seed": args.seed,
            "limit": args.limit,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_cuda_device_count": torch.cuda.device_count(),
            "visible_cuda_devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        },
    }
    write_json(config_path, run_config)

    try:
        print(
            f"Loading TableLlama with vLLM dtype=auto, tensor_parallel_size="
            f"{args.tensor_parallel_size}: {model_path}",
            flush=True,
        )
        try:
            engine = LLM(
                model=str(model_path),
                tokenizer=str(model_path),
                tokenizer_mode="slow",
                trust_remote_code=False,
                dtype="auto",
                load_format="auto",
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_length,
                gpu_memory_utilization=args.gpu_memory_utilization,
                seed=args.seed,
                enforce_eager=args.enforce_eager,
                disable_log_stats=True,
            )
        except BaseException as error:
            if is_cuda_oom(error):
                raise oom_error(error) from error
            raise
        engine_dtype = getattr(getattr(engine, "llm_engine", None), "model_config", None)
        actual_dtype = str(
            getattr(engine_dtype, "dtype", model_metadata.get("config_torch_dtype", "auto"))
        )
        run_config["model"]["actual_dtype"] = actual_dtype
        write_json(config_path, run_config)

        sampling_params = SamplingParams(
            temperature=0, top_p=1.0, max_tokens=args.max_new_tokens, seed=args.seed
        )
        pending = examples[len(existing) :]
        completed = len(existing)
        mode = "a" if existing else "w"
        with output_path.open(mode, encoding="utf-8") as output_file:
            for batch in batched(pending, args.batch_size):
                results = run_batch(
                    batch,
                    tokenizer,
                    engine,
                    sampling_params,
                    model_path,
                    args.max_model_length,
                    args.max_input_tokens,
                    args.max_new_tokens,
                    args.seed,
                    actual_dtype,
                )
                append_results(output_file, results)
                completed += len(results)
                print(f"Completed {completed}/{len(examples)}", flush=True)
        validate_final_output(output_path, examples)
    except BaseException as error:
        run_config.update(
            status="failed", finished_at=utc_now(), error=f"{type(error).__name__}: {error}"
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
