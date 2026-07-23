#!/usr/bin/env python3
"""Low-call GPT reasoning baselines for MACT table-QA JSONL files."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DIRECT_LLM_DIR = SCRIPT_DIR.parent / "DirectLLM"
if str(DIRECT_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(DIRECT_LLM_DIR))

from direct_llm_baseline import (  # noqa: E402
    RequestStartLimiter,
    ensure_local_base_url_bypasses_proxy,
    load_env_file,
    load_model_config,
    read_jsonl,
    read_question,
    read_table,
    resolve_path,
)


DEFAULT_DATASET_PATH = PROJECT_ROOT / "output" / "crt_answerable.jsonl"
DEFAULT_MODEL_CONFIG = DIRECT_LLM_DIR / "gpt_5.yaml"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "reasoning_llm_results"
PROMPT_VERSION = "reasoning_llm_short_v1"
SC_SAMPLE_COUNT = 3

COT_INSTRUCTION = (
    "Reason in ≤3 short steps. End with `FINAL_ANSWER: <answer>`."
)
SELF_CORRECTION_INSTRUCTION = (
    "Check the answer against the table and fix it if needed.\n"
    "Give one short check, then `FINAL_ANSWER: <answer>`."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Zero-shot CoT, Self-Consistency-3, and single-pass "
            "Self-Correction-1 with shared GPT generations."
        )
    )
    parser.add_argument("--dataset_path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model_config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--env_file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument(
        "--multi_choice_mode",
        required=True,
        choices=["n", "separate"],
        help=(
            "'n' sends one n=3 request; 'separate' sends three n=1 requests. "
            "Choose explicitly after running test_n3_request.py."
        ),
    )
    parser.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "4")))
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("TEMPERATURE", "0.6")),
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=float(os.getenv("TOP_P", "0.95")),
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=int(os.getenv("MAX_TOKENS", "2000")),
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=float(os.getenv("FREQUENCY_PENALTY", "0")),
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=float(os.getenv("PRESENCE_PENALTY", "0")),
    )
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--request_interval_s",
        type=float,
        default=float(os.getenv("REQUEST_INTERVAL_S", "0")),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful phases from output_dir/checkpoint.jsonl.",
    )
    return parser.parse_args()


def build_cot_prompt(question: str, table_text: str) -> str:
    """Build the deliberately short shared prompt for CoT and SC-3."""
    return f"TABLE\n{table_text}\n\nQ: {question}\n{COT_INSTRUCTION}"


def build_self_correction_messages(
    cot_prompt: str, initial_response: str
) -> list[dict[str, str]]:
    """Continue the CoT conversation with one compact correction request."""
    return [
        {"role": "user", "content": cot_prompt},
        {"role": "assistant", "content": initial_response},
        {"role": "user", "content": SELF_CORRECTION_INSTRUCTION},
    ]


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _parse_chat_response(response_body: bytes, http_status: int) -> dict[str, Any]:
    data = json.loads(response_body.decode("utf-8"))
    raw_choices = data.get("choices")
    if not isinstance(raw_choices, list) or not raw_choices:
        raise ValueError(f"LLM response has no choices: {data}")

    choices: list[dict[str, Any]] = []
    for fallback_index, raw_choice in enumerate(raw_choices):
        if not isinstance(raw_choice, dict):
            raise ValueError(f"Invalid choice at index {fallback_index}: {raw_choice!r}")
        message = raw_choice.get("message") or {}
        content = message.get("content")
        if content is None:
            raise ValueError(
                f"LLM choice {fallback_index} has no message content: {raw_choice}"
            )
        choices.append(
            {
                "index": int(raw_choice.get("index", fallback_index)),
                "content": str(content).strip(),
                "finish_reason": raw_choice.get("finish_reason"),
            }
        )
    choices.sort(key=lambda choice: choice["index"])

    usage = data.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return {
        "http_status": http_status,
        "model": str(data.get("model", "")),
        "choices": choices,
        "usage": usage,
    }


def send_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    frequency_penalty: float,
    presence_penalty: float,
    request_timeout: float,
    max_retries: int,
    request_limiter: RequestStartLimiter | None,
) -> dict[str, Any]:
    """Send one Chat Completions request without automatic n-mode fallback."""
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "n": n,
        "stop": None,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{normalize_base_url(base_url)}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        if request_limiter is not None:
            request_limiter.wait()
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                status = int(getattr(response, "status", response.getcode()))
                return _parse_chat_response(response.read(), status)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {error_body}")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(min(2**attempt, 30))

    raise RuntimeError(
        f"LLM request failed after {max_retries + 1} attempts: {last_error}"
    )


def _strip_answer_markup(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^[\s`*_#>-]+", "", value)
    value = re.sub(r"[\s`*_]+$", "", value)
    return value.strip()


def extract_final_answer(response: str) -> tuple[str, str]:
    """Extract the last explicit final marker, otherwise the last non-empty line."""
    marker_matches = list(
        re.finditer(
            r"(?im)^\s*`?FINAL_ANSWER\s*:\s*(.*?)`?\s*$",
            str(response),
        )
    )
    for match in reversed(marker_matches):
        answer = _strip_answer_markup(match.group(1))
        if answer:
            return answer, "marker"

    non_empty_lines = [
        _strip_answer_markup(line)
        for line in str(response).splitlines()
        if _strip_answer_markup(line)
    ]
    if non_empty_lines:
        return non_empty_lines[-1], "last_nonempty_line"
    return "", "empty"


def _prediction_items(answer: str) -> list[str]:
    text = str(answer).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed]
    if "|" in text:
        return [part.replace(r"\p", "|").strip() for part in text.split("|")]
    return [text]


def _normalize_vote_item(value: str) -> tuple[str, str]:
    text = "".join(
        char
        for char in unicodedata.normalize("NFKD", str(value))
        if unicodedata.category(char) != "Mn"
    )
    text = re.sub(r"[‘’´`]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.endswith("."):
        text = text[:-1].rstrip()

    numeric_text = text.replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", numeric_text):
        amount = float(numeric_text)
        if abs(amount - round(amount)) < 1e-9:
            return "number", str(int(round(amount)))
        return "number", format(amount, ".12g")
    return "string", text.casefold()


def canonical_vote_key(answer: str) -> str:
    """Create a gold-independent, evaluator-like key for answer voting."""
    normalized_items = sorted(
        _normalize_vote_item(item) for item in _prediction_items(answer)
    )
    return json.dumps(normalized_items, ensure_ascii=False, separators=(",", ":"))


def select_self_consistent_answer(
    choices: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(choices) != SC_SAMPLE_COUNT:
        raise ValueError(
            f"Self-Consistency-{SC_SAMPLE_COUNT} requires exactly "
            f"{SC_SAMPLE_COUNT} choices, got {len(choices)}"
        )

    keys = [canonical_vote_key(str(choice.get("pred_answer", ""))) for choice in choices]
    counts = Counter(keys)
    highest_count = max(counts.values())
    winning_keys = {key for key, count in counts.items() if count == highest_count}
    all_different = highest_count == 1
    selected_index = 0
    if not all_different:
        selected_index = next(
            index for index, key in enumerate(keys) if key in winning_keys
        )

    return {
        "pred_answer": str(choices[selected_index].get("pred_answer", "")),
        "selected_choice_index": selected_index,
        "all_different_tie": all_different,
        "vote_counts": [
            {"key": key, "count": count}
            for key, count in sorted(counts.items())
        ],
    }


def sum_usage(usages: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Sum numeric usage fields, including nested token-detail mappings."""
    total: dict[str, Any] = {}
    for usage in usages:
        for key, value in usage.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                total[key] = total.get(key, 0) + value
            elif isinstance(value, dict):
                nested = total.setdefault(key, {})
                if not isinstance(nested, dict):
                    continue
                nested_sum = sum_usage([value])
                for nested_key, nested_value in nested_sum.items():
                    if isinstance(nested_value, (int, float)):
                        nested[nested_key] = nested.get(nested_key, 0) + nested_value
    return total


class CheckpointWriter:
    """Append complete state snapshots safely from concurrent workers."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self._lock = threading.Lock()

    def append(self, state: dict[str, Any]) -> None:
        snapshot = dict(state)
        snapshot["updated_at"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(snapshot, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            with self.checkpoint_path.open("a", encoding="utf-8") as output_file:
                output_file.write(line)
                output_file.flush()
                os.fsync(output_file.fileno())


def load_checkpoint(checkpoint_path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not checkpoint_path.is_file():
        return latest
    with checkpoint_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            state = json.loads(line)
            example_id = str(state.get("id", "")).strip()
            if not example_id:
                raise ValueError(
                    f"Checkpoint row {line_number} is missing a non-empty id"
                )
            latest[example_id] = state
    return latest


def _request_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "api_key": config["api_key"],
        "base_url": config["base_url"],
        "model_name": config["model_name"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens": config["max_tokens"],
        "frequency_penalty": config["frequency_penalty"],
        "presence_penalty": config["presence_penalty"],
        "request_timeout": config["request_timeout"],
        "max_retries": config["max_retries"],
        "request_limiter": config["request_limiter"],
    }


def _generate_cot_choices(
    cot_prompt: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], int, list[str], list[int]]:
    messages = [{"role": "user", "content": cot_prompt}]
    request_kwargs = _request_kwargs(config)
    responses: list[dict[str, Any]]
    if config["multi_choice_mode"] == "n":
        result = send_chat_completion(
            messages=messages,
            n=SC_SAMPLE_COUNT,
            **request_kwargs,
        )
        if len(result["choices"]) != SC_SAMPLE_COUNT:
            raise RuntimeError(
                f"n={SC_SAMPLE_COUNT} request returned "
                f"{len(result['choices'])} choices; no automatic fallback was attempted"
            )
        responses = [result]
    else:
        responses = [
            send_chat_completion(messages=messages, n=1, **request_kwargs)
            for _ in range(SC_SAMPLE_COUNT)
        ]

    choices: list[dict[str, Any]] = []
    returned_models: list[str] = []
    http_statuses: list[int] = []
    for response in responses:
        returned_models.append(str(response.get("model", "")))
        http_statuses.append(int(response["http_status"]))
        for raw_choice in response["choices"]:
            answer, parse_status = extract_final_answer(raw_choice["content"])
            choices.append(
                {
                    "index": len(choices),
                    "response": raw_choice["content"],
                    "pred_answer": answer,
                    "parse_status": parse_status,
                    "finish_reason": raw_choice.get("finish_reason"),
                }
            )
    if len(choices) != SC_SAMPLE_COUNT:
        raise RuntimeError(
            f"Expected {SC_SAMPLE_COUNT} total CoT choices, got {len(choices)}"
        )
    return (
        choices,
        sum_usage(response.get("usage", {}) for response in responses),
        len(responses),
        returned_models,
        http_statuses,
    )


def _prepare_run_metadata(
    item: dict[str, Any], config: dict[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    table_text, table_truncated = read_table(item, max_table_chars=0)
    if table_truncated:
        raise AssertionError("Reasoning baselines must always use the complete table")
    cot_prompt = build_cot_prompt(read_question(item), table_text)
    prompt_sha256 = hashlib.sha256(cot_prompt.encode("utf-8")).hexdigest()
    generation_config = {
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens": config["max_tokens"],
        "frequency_penalty": config["frequency_penalty"],
        "presence_penalty": config["presence_penalty"],
        "sc_samples": SC_SAMPLE_COUNT,
    }
    return cot_prompt, prompt_sha256, generation_config


def validate_resume_compatibility(
    item: dict[str, Any],
    existing_state: dict[str, Any],
    config: dict[str, Any],
) -> None:
    _, prompt_sha256, generation_config = _prepare_run_metadata(item, config)
    resume_fields = {
        "prompt_version": PROMPT_VERSION,
        "model_name": config["model_name"],
        "multi_choice_mode": config["multi_choice_mode"],
        "cot_prompt_sha256": prompt_sha256,
        "generation_config": generation_config,
    }
    mismatches = [
        field
        for field, expected_value in resume_fields.items()
        if existing_state.get(field) != expected_value
    ]
    if mismatches:
        raise ValueError(
            f"Cannot resume {item['id']}: checkpoint differs in "
            + ", ".join(mismatches)
        )


def run_one(
    item: dict[str, Any],
    existing_state: dict[str, Any] | None,
    config: dict[str, Any],
    checkpoint_writer: CheckpointWriter,
) -> dict[str, Any]:
    example_id = str(item["id"])
    cot_prompt, prompt_sha256, generation_config = _prepare_run_metadata(
        item, config
    )
    if existing_state:
        validate_resume_compatibility(item, existing_state, config)

    state = dict(existing_state or {})
    state.update(
        {
            "schema_version": 1,
            "prompt_version": PROMPT_VERSION,
            "id": example_id,
            "item": item,
            "model_name": config["model_name"],
            "multi_choice_mode": config["multi_choice_mode"],
            "generation_config": generation_config,
            "cot_prompt_sha256": prompt_sha256,
        }
    )

    if state.get("cot_status") != "success":
        try:
            (
                cot_choices,
                cot_usage,
                request_count,
                returned_models,
                http_statuses,
            ) = _generate_cot_choices(cot_prompt, config)
            state.update(
                {
                    "cot_status": "success",
                    "cot_error": "",
                    "cot_choices": cot_choices,
                    "cot_usage": cot_usage,
                    "cot_request_count": request_count,
                    "cot_returned_models": returned_models,
                    "cot_http_statuses": http_statuses,
                    "self_consistency": select_self_consistent_answer(cot_choices),
                }
            )
        except Exception as exc:
            state.update({"cot_status": "fail", "cot_error": str(exc)})
            checkpoint_writer.append(state)
            return state
        checkpoint_writer.append(state)

    cot_choices = state.get("cot_choices") or []
    if len(cot_choices) != SC_SAMPLE_COUNT:
        state.update(
            {
                "cot_status": "fail",
                "cot_error": (
                    f"Checkpoint has {len(cot_choices)} CoT choices; "
                    f"expected {SC_SAMPLE_COUNT}"
                ),
            }
        )
        checkpoint_writer.append(state)
        return state

    if not state.get("self_consistency"):
        state["self_consistency"] = select_self_consistent_answer(cot_choices)
        checkpoint_writer.append(state)

    if state.get("self_correction_status") != "success":
        try:
            initial_response = str(cot_choices[0]["response"])
            correction_result = send_chat_completion(
                messages=build_self_correction_messages(cot_prompt, initial_response),
                n=1,
                **_request_kwargs(config),
            )
            if len(correction_result["choices"]) != 1:
                raise RuntimeError(
                    "Self-correction request returned "
                    f"{len(correction_result['choices'])} choices; expected 1"
                )
            raw_choice = correction_result["choices"][0]
            answer, parse_status = extract_final_answer(raw_choice["content"])
            state.update(
                {
                    "self_correction_status": "success",
                    "self_correction_error": "",
                    "self_correction_response": raw_choice["content"],
                    "self_correction_pred_answer": answer,
                    "self_correction_parse_status": parse_status,
                    "self_correction_finish_reason": raw_choice.get("finish_reason"),
                    "self_correction_usage": correction_result.get("usage", {}),
                    "self_correction_request_count": 1,
                    "self_correction_returned_model": correction_result.get("model", ""),
                    "self_correction_http_status": correction_result["http_status"],
                }
            )
        except Exception as exc:
            state.update(
                {
                    "self_correction_status": "fail",
                    "self_correction_error": str(exc),
                }
            )
        state["total_usage"] = sum_usage(
            [
                state.get("cot_usage", {}),
                state.get("self_correction_usage", {}),
            ]
        )
        checkpoint_writer.append(state)

    if "total_usage" not in state:
        state["total_usage"] = sum_usage(
            [
                state.get("cot_usage", {}),
                state.get("self_correction_usage", {}),
            ]
        )
    return state


def validate_dataset_rows(rows: list[dict[str, Any]]) -> None:
    seen_ids: set[str] = set()
    for row_index, item in enumerate(rows):
        example_id = str(item.get("id", "")).strip()
        if not example_id:
            raise ValueError(f"Dataset row {row_index} is missing a non-empty id")
        if example_id in seen_ids:
            raise ValueError(f"Duplicate dataset id: {example_id}")
        seen_ids.add(example_id)


def _base_result(
    item: dict[str, Any],
    state: dict[str, Any] | None,
    method: str,
) -> dict[str, Any]:
    result = dict(item)
    result.update(
        {
            "method": method,
            "model_name": (state or {}).get("model_name", ""),
            "prompt_version": (state or {}).get("prompt_version", PROMPT_VERSION),
            "temperature": (state or {}).get("generation_config", {}).get(
                "temperature", 0.6
            ),
            "generation_config": (state or {}).get("generation_config", {}),
            "multi_choice_mode": (state or {}).get("multi_choice_mode", ""),
            "table_truncated": False,
        }
    )
    return result


def materialize_results(
    rows: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    method_paths = {
        "zero_shot_cot": output_dir / "zero_shot_cot" / "results.jsonl",
        "self_consistency_3": output_dir
        / "self_consistency_3"
        / "results.jsonl",
        "single_pass_self_correction_1": output_dir
        / "single_pass_self_correction_1"
        / "results.jsonl",
    }
    for path in method_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    output_files = {
        method: path.open("w", encoding="utf-8")
        for method, path in method_paths.items()
    }
    try:
        for item in rows:
            example_id = str(item["id"])
            state = states.get(example_id)

            cot_result = _base_result(item, state, "zero_shot_cot")
            cot_choices = (state or {}).get("cot_choices") or []
            cot_success = (
                (state or {}).get("cot_status") == "success"
                and len(cot_choices) == SC_SAMPLE_COUNT
            )
            cot_result.update(
                {
                    "response": cot_choices[0]["response"] if cot_success else "",
                    "pred_answer": (
                        cot_choices[0]["pred_answer"] if cot_success else ""
                    ),
                    "execute_status": "success" if cot_success else "fail",
                    "parse_status": (
                        cot_choices[0]["parse_status"] if cot_success else ""
                    ),
                    "finish_reason": (
                        cot_choices[0].get("finish_reason") if cot_success else None
                    ),
                    "error": "" if cot_success else (state or {}).get(
                        "cot_error", "not_started"
                    ),
                    "usage": (state or {}).get("cot_usage", {}),
                }
            )

            sc_result = _base_result(item, state, "self_consistency_3")
            sc_info = (state or {}).get("self_consistency") or {}
            sc_result.update(
                {
                    "response": (
                        [choice["response"] for choice in cot_choices]
                        if cot_success
                        else []
                    ),
                    "pred_answer": sc_info.get("pred_answer", ""),
                    "execute_status": "success" if cot_success else "fail",
                    "reasoning_samples": cot_choices if cot_success else [],
                    "selected_choice_index": sc_info.get(
                        "selected_choice_index"
                    ),
                    "all_different_tie": sc_info.get("all_different_tie"),
                    "vote_counts": sc_info.get("vote_counts", []),
                    "error": "" if cot_success else (state or {}).get(
                        "cot_error", "not_started"
                    ),
                    "usage": (state or {}).get("cot_usage", {}),
                }
            )

            correction_result = _base_result(
                item, state, "single_pass_self_correction_1"
            )
            correction_success = (
                (state or {}).get("self_correction_status") == "success"
            )
            correction_result.update(
                {
                    "response": (
                        (state or {}).get("self_correction_response", "")
                        if correction_success
                        else ""
                    ),
                    "pred_answer": (
                        (state or {}).get("self_correction_pred_answer", "")
                        if correction_success
                        else ""
                    ),
                    "execute_status": (
                        "success" if correction_success else "fail"
                    ),
                    "parse_status": (
                        (state or {}).get("self_correction_parse_status", "")
                        if correction_success
                        else ""
                    ),
                    "finish_reason": (state or {}).get(
                        "self_correction_finish_reason"
                    ),
                    "initial_cot_response": (
                        cot_choices[0]["response"] if cot_success else ""
                    ),
                    "initial_cot_answer": (
                        cot_choices[0]["pred_answer"] if cot_success else ""
                    ),
                    "error": (
                        ""
                        if correction_success
                        else (state or {}).get(
                            "self_correction_error", "not_started"
                        )
                    ),
                    "usage": (state or {}).get(
                        "self_correction_usage", {}
                    ),
                }
            )

            for method, result in (
                ("zero_shot_cot", cot_result),
                ("self_consistency_3", sc_result),
                ("single_pass_self_correction_1", correction_result),
            ):
                output_files[method].write(
                    json.dumps(result, ensure_ascii=False, default=str) + "\n"
                )
    finally:
        for output_file in output_files.values():
            output_file.close()
    return method_paths


def _validate_args(args: argparse.Namespace) -> None:
    if args.workers <= 0:
        raise ValueError(f"--workers must be positive, got {args.workers}")
    if args.max_retries < 0:
        raise ValueError(
            f"--max_retries must be non-negative, got {args.max_retries}"
        )
    if args.request_interval_s < 0:
        raise ValueError(
            "--request_interval_s must be non-negative, got "
            f"{args.request_interval_s}"
        )
    if not 0 <= args.temperature <= 2:
        raise ValueError(
            f"--temperature must be between 0 and 2, got {args.temperature}"
        )
    if not 0 <= args.top_p <= 1:
        raise ValueError(f"--top_p must be between 0 and 1, got {args.top_p}")
    if args.max_tokens <= 0:
        raise ValueError(f"--max_tokens must be positive, got {args.max_tokens}")
    if not -2 <= args.frequency_penalty <= 2:
        raise ValueError("--frequency_penalty must be between -2 and 2")
    if not -2 <= args.presence_penalty <= 2:
        raise ValueError("--presence_penalty must be between -2 and 2")


def main() -> None:
    args = parse_args()
    _validate_args(args)

    env_file = Path(args.env_file).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = output_dir / "checkpoint.jsonl"

    load_env_file(env_file)
    model_config = load_model_config(resolve_path(args.model_config), env_file)
    ensure_local_base_url_bypasses_proxy(model_config["base_url"])

    rows = read_jsonl(dataset_path, args.limit)
    validate_dataset_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint_path}. "
            "Use --resume or choose another --output_dir."
        )

    existing_states = load_checkpoint(checkpoint_path) if args.resume else {}
    checkpoint_writer = CheckpointWriter(checkpoint_path)
    request_limiter = RequestStartLimiter(args.request_interval_s)
    config: dict[str, Any] = {
        **model_config,
        "multi_choice_mode": args.multi_choice_mode,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "request_timeout": args.request_timeout,
        "max_retries": args.max_retries,
        "request_limiter": request_limiter,
    }

    for item in rows:
        existing_state = existing_states.get(str(item["id"]))
        if existing_state:
            validate_resume_compatibility(item, existing_state, config)

    pending_rows = [
        item
        for item in rows
        if not (
            existing_states.get(str(item["id"]), {}).get("cot_status") == "success"
            and existing_states.get(str(item["id"]), {}).get(
                "self_correction_status"
            )
            == "success"
        )
    ]
    print(
        f"Reasoning baselines: examples={len(rows)} pending={len(pending_rows)} "
        f"model={model_config['model_name']} mode={args.multi_choice_mode} "
        f"workers={args.workers} temperature={args.temperature} "
        f"top_p={args.top_p} max_tokens={args.max_tokens}",
        flush=True,
    )

    completed = 0
    failed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = {
                executor.submit(
                    run_one,
                    item,
                    existing_states.get(str(item["id"])),
                    config,
                    checkpoint_writer,
                ): str(item["id"])
                for item in pending_rows
            }
            for future in concurrent.futures.as_completed(futures):
                state = future.result()
                completed += 1
                if not (
                    state.get("cot_status") == "success"
                    and state.get("self_correction_status") == "success"
                ):
                    failed += 1
                print(
                    f"[PROGRESS] {completed}/{len(pending_rows)} "
                    f"id={state['id']} cot={state.get('cot_status')} "
                    f"correction={state.get('self_correction_status')}",
                    flush=True,
                )
    finally:
        final_states = load_checkpoint(checkpoint_path)
        method_paths = materialize_results(rows, final_states, output_dir)
        for method, path in method_paths.items():
            print(f"{method}: {path}", flush=True)

    if failed:
        print(
            f"Completed with {failed} examples containing a failed phase. "
            "Run again with --resume to retry only failed phases.",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
