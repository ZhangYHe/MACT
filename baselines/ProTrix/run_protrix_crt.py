#!/usr/bin/env python3
"""Run the local ProTrix model on CRT-QA with two-stage SQL reasoning."""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer

from protrix_sql import execute_queries, extract_sql_queries, format_execution_blocks


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL_PATH = Path("/home/zhangyunhe/nas/model/ProTrix")
DEFAULT_INPUT_PATH = PROJECT_ROOT / "output" / "crt_answerable.jsonl"
EXPECTED_MODEL_TYPE = "llama"
DEFAULT_MODEL_MAX_LENGTH = 4096
DEFAULT_MAX_INPUT_TOKENS = 3072
DEFAULT_MAX_NEW_TOKENS = 1024
REQUIRED_TOKENIZER_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
TASK_TEXT = """## Task
You will answering the question based on the given table.You should reach a short-form answer after reasoning.
You are asked to answer the question in three steps.
1. Analyze the question and the given context. Make up a plan to answer the question.
2. Write one or more SQL to query the table for neccessary information and output expected result
3. Reason step-by-step based on execution result to reach the final answer.

## Answer:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local ProTrix 7B on CRT-QA with Plan-then-Reason SQL inference."
    )
    parser.add_argument("--model_name_or_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--input_path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output_path", default="")
    parser.add_argument("--run_config_path", default="")
    parser.add_argument("--batch_size", type=int, default=8)
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
        help="Validate model files, input, and prompts without loading weights or CUDA.",
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
        raise FileNotFoundError(f"ProTrix model directory does not exist: {model_path}")
    missing = [name for name in REQUIRED_TOKENIZER_FILES if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError("ProTrix download is incomplete. Missing: " + ", ".join(missing))
    with (model_path / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError(f"Expected a Llama ProTrix config, got {config.get('model_type')!r}.")
    with (model_path / "model.safetensors.index.json").open(encoding="utf-8") as index_file:
        index = json.load(index_file)
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names:
        raise ValueError("ProTrix safetensors index has no weight shards.")
    missing_shards = [name for name in shard_names if not (model_path / name).is_file()]
    if missing_shards:
        raise FileNotFoundError(
            "ProTrix download is incomplete. Missing weight shards: "
            + ", ".join(missing_shards)
        )
    shard_sizes = {name: (model_path / name).stat().st_size for name in shard_names}
    return {
        "path": str(model_path),
        "weights_index": "model.safetensors.index.json",
        "weight_shards": shard_sizes,
        "weights_size_bytes": sum(shard_sizes.values()),
        "local_files_only": True,
        "engine": "vllm",
        "dtype_policy": "vllm_auto_from_checkpoint_config",
        "config_torch_dtype": config.get("torch_dtype"),
        "max_position_embeddings": config.get("max_position_embeddings"),
    }


def validate_example(raw: object, line_number: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Input line {line_number} must be an object.")
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
    examples = []
    seen = set()
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
        raise ValueError(f"No input examples found in {path}")
    return examples


def normalize_cell(value: object) -> str:
    return "" if value is None else str(value)


def serialize_table(table: list[list[object]]) -> str:
    return "\n".join(" | ".join(normalize_cell(cell) for cell in row) for row in table)


def build_base_prompt(item: dict[str, Any], table: list[list[object]]) -> str:
    return (
        f"## Question\n{item['statement'].strip()}\n\n"
        f"## Table\n{serialize_table(table)}\n\n{TASK_TEXT}"
    )


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def truncate_stage_one(
    item: dict[str, Any], tokenizer: Any, max_input_tokens: int
) -> tuple[str, list[list[object]], int, int]:
    table = [list(row) for row in item["table_text"]]
    original_rows = len(table) - 1
    while True:
        prompt = build_base_prompt(item, table)
        length = token_count(tokenizer, prompt)
        if length <= max_input_tokens:
            return prompt, table, length, original_rows - (len(table) - 1)
        if len(table) <= 2:
            raise ValueError(
                f"Prompt for {item['id']} exceeds {max_input_tokens} tokens even with one row."
            )
        table.pop()


def _analysis_before_sql(first_response: str) -> str:
    match = re.search(r"```\s*sql", first_response, flags=re.IGNORECASE)
    return first_response[: match.start()].rstrip() if match else first_response.rstrip()


def build_second_prompt(
    item: dict[str, Any],
    stage_one_table: list[list[object]],
    first_response: str,
    executions: list[dict[str, Any]],
    tokenizer: Any,
    max_input_tokens: int,
) -> tuple[str, int, int, bool]:
    table = [list(row) for row in stage_one_table]
    original_rows = len(table) - 1
    analysis = _analysis_before_sql(first_response)
    execution_text = format_execution_blocks(executions)
    suffix = f"{analysis}\n{execution_text}\n3.Step-by-step Answer prediction\n"
    while True:
        prompt = build_base_prompt(item, table) + suffix
        length = token_count(tokenizer, prompt)
        if length <= max_input_tokens:
            return prompt, length, original_rows - (len(table) - 1), False
        if len(table) <= 2:
            break
        table.pop()

    fixed_suffix = f"\n{execution_text}\n3.Step-by-step Answer prediction\n"
    fixed = build_base_prompt(item, table) + fixed_suffix
    fixed_length = token_count(tokenizer, fixed)
    if fixed_length > max_input_tokens:
        raise ValueError(
            f"Second-stage SQL context for {item['id']} exceeds {max_input_tokens} tokens."
        )
    analysis_ids = tokenizer.encode(analysis, add_special_tokens=False)
    budget = max_input_tokens - fixed_length
    shortened = tokenizer.decode(analysis_ids[:budget], skip_special_tokens=True).rstrip()
    prompt = build_base_prompt(item, table) + f"{shortened}{fixed_suffix}"
    while token_count(tokenizer, prompt) > max_input_tokens and shortened:
        analysis_ids = analysis_ids[:-8]
        shortened = tokenizer.decode(analysis_ids, skip_special_tokens=True).rstrip()
        prompt = build_base_prompt(item, table) + f"{shortened}{fixed_suffix}"
    return prompt, token_count(tokenizer, prompt), original_rows - (len(table) - 1), True


def _clean_answer(text: str) -> object:
    value = text.strip()
    value = re.sub(r"^[\s#>*\-]+", "", value)
    value = value.strip("`*_ \t")
    if value.endswith("."):
        value = value[:-1].rstrip()
    value = value.strip().strip('"').strip("'")
    if value.startswith("[") and value.endswith("]"):
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(value)
            except Exception:
                continue
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
    return value


def extract_short_answer(response: str) -> tuple[object, str]:
    line_pattern = re.compile(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(.+?)\s*$"
    )
    matches = line_pattern.findall(response)
    if matches:
        return _clean_answer(matches[-1]), "answer_marker"
    inline = re.findall(
        r"(?i)\b(?:the\s+)?(?:final\s+)?answer\s+is\s+([^\n]+)", response
    )
    if inline:
        return _clean_answer(inline[-1]), "inline_answer_is"
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if lines:
        return _clean_answer(lines[-1]), "last_nonempty_line"
    return "", "empty_response"


def batched(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def generate_texts(engine: Any, prompts: list[str], sampling_params: Any) -> list[str]:
    if not prompts:
        return []
    outputs = engine.generate(prompts, sampling_params, use_tqdm=False)
    texts = [output.outputs[0].text for output in outputs]
    if len(texts) != len(prompts):
        raise RuntimeError("vLLM returned a different number of outputs than prompts.")
    return texts


def run_batch(
    batch: list[dict[str, Any]],
    tokenizer: Any,
    engine: Any,
    sampling_params: Any,
    model_path: Path,
    max_input_tokens: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    stage_one = [truncate_stage_one(item, tokenizer, max_input_tokens) for item in batch]
    first_responses = generate_texts(engine, [entry[0] for entry in stage_one], sampling_params)
    execution_batches = []
    second_prompts = []
    second_indices = []
    second_metadata = {}
    for index, (item, stage, first_response) in enumerate(zip(batch, stage_one, first_responses)):
        queries = extract_sql_queries(first_response)
        executions = execute_queries(stage[1], queries) if queries else []
        execution_batches.append(executions)
        if any(execution["status"] == "ok" for execution in executions):
            prompt, length, truncated_rows, analysis_truncated = build_second_prompt(
                item,
                stage[1],
                first_response,
                executions,
                tokenizer,
                max_input_tokens,
            )
            second_indices.append(index)
            second_prompts.append(prompt)
            second_metadata[index] = (length, truncated_rows, analysis_truncated)
    second_responses = generate_texts(engine, second_prompts, sampling_params)
    second_by_index = dict(zip(second_indices, second_responses))

    results = []
    for index, (item, stage, first_response, executions) in enumerate(
        zip(batch, stage_one, first_responses, execution_batches)
    ):
        used_second = index in second_by_index
        final_response = second_by_index.get(index, first_response)
        prediction, extraction_method = extract_short_answer(final_response)
        second_info = second_metadata.get(index)
        result = dict(item)
        result.update(
            {
                "pred_answer": prediction,
                "pred_answer_all": [prediction],
                "run_status": "completed",
                "protrix_metadata": {
                    "model_name_or_path": str(model_path),
                    "first_response": first_response,
                    "second_response": second_by_index.get(index),
                    "used_second_pass": used_second,
                    "sql_queries": extract_sql_queries(first_response),
                    "sql_executions": executions,
                    "answer_extraction_method": extraction_method,
                    "stage_one_input_tokens": stage[2],
                    "stage_one_truncated_rows": stage[3],
                    "stage_two_input_tokens": second_info[0] if second_info else None,
                    "stage_two_additional_truncated_rows": second_info[1] if second_info else None,
                    "stage_two_analysis_truncated": second_info[2] if second_info else None,
                    "max_input_tokens": max_input_tokens,
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
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
                raise ValueError(f"Duplicate output id: {item['id']}")
            seen.add(item["id"])
            results.append(item)
    return results


def validate_resume_prefix(examples: list[dict[str, Any]], existing: list[dict[str, Any]]) -> None:
    if len(existing) > len(examples):
        raise ValueError("Existing output is longer than the selected input.")
    if [x["id"] for x in existing] != [x["id"] for x in examples[: len(existing)]]:
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
    if [x["id"] for x in results] != [x["id"] for x in examples]:
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
        raise ValueError("max_input_tokens + max_new_tokens cannot exceed max_model_length.")
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
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)

    if args.validate_only:
        lengths = []
        truncated = 0
        for item in examples:
            _, _, length, removed = truncate_stage_one(item, tokenizer, args.max_input_tokens)
            lengths.append(length)
            truncated += bool(removed)
        print(f"Model files: OK ({model_path})")
        print(
            f"Input/prompts: OK ({len(examples)} examples, max={max(lengths)} tokens, "
            f"truncated_tables={truncated})"
        )
        print("Validation only: model weights were not loaded and CUDA was not used.")
        return

    if not args.output_path:
        raise ValueError("--output_path is required unless --validate_only is used.")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError("CUDA_VISIBLE_DEVICES is not set; choose GPU(s) before running.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected environment.")
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
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "seed": args.seed,
            "two_stage_sql": True,
            "limit": args.limit,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
            "pandas": pd.__version__,
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
            f"Loading ProTrix with vLLM dtype=auto, tensor_parallel_size="
            f"{args.tensor_parallel_size}: {model_path}",
            flush=True,
        )
        engine = LLM(
            model=str(model_path),
            tokenizer=str(model_path),
            tokenizer_mode="slow",
            trust_remote_code=False,
            dtype="auto",
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_length,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
            enforce_eager=args.enforce_eager,
            disable_log_stats=True,
        )
        sampling_params = SamplingParams(
            temperature=0, top_p=1.0, max_tokens=args.max_new_tokens
        )
        run_config["model"]["actual_dtype"] = model_metadata.get("config_torch_dtype")
        write_json(config_path, run_config)

        pending = examples[len(existing) :]
        completed = len(existing)
        mode = "a" if existing else "w"
        with output_path.open(mode, encoding="utf-8") as output_file:
            for batch in batched(pending, args.batch_size):
                try:
                    results = run_batch(
                        batch,
                        tokenizer,
                        engine,
                        sampling_params,
                        model_path,
                        args.max_input_tokens,
                        args.max_new_tokens,
                    )
                except RuntimeError as error:
                    if "out of memory" in str(error).lower():
                        raise RuntimeError(
                            "ProTrix CUDA OOM. First retry with lower BATCH_SIZE and the same "
                            "RUN_DIR --resume. If model initialization itself fails on one 3090, "
                            "select two GPUs and set TENSOR_PARALLEL_SIZE=2."
                        ) from error
                    raise
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
