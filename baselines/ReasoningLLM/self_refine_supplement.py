#!/usr/bin/env python3
"""Repair technical generation failures and add a standard Self-Refine-1 run."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DIRECT_LLM_DIR = SCRIPT_DIR.parent / "DirectLLM"
if str(DIRECT_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(DIRECT_LLM_DIR))

import reasoning_llm_baseline as baseline  # noqa: E402
from direct_llm_baseline import (  # noqa: E402
    RequestStartLimiter,
    ensure_local_base_url_bypasses_proxy,
    load_env_file,
    load_model_config,
    resolve_path,
)


DEFAULT_MODEL_CONFIG = DIRECT_LLM_DIR / "gpt_5.yaml"
SUPPLEMENT_CHECKPOINT_NAME = "self_refine_supplement_checkpoint.jsonl"
SUPPLEMENT_SCHEMA_VERSION = 1
SUPPLEMENT_PROMPT_VERSION = "self_refine_short_v1"
REPAIR_MAX_TOKENS = 4000

FEEDBACK_INSTRUCTION = (
    "Check the answer against the table. Give brief feedback only."
)
REFINEMENT_INSTRUCTION = (
    "Revise using the feedback. Reason in ≤3 short steps. "
    "End with `FINAL_ANSWER: <answer>`."
)


class InvalidGenerationError(RuntimeError):
    """Carry auditable attempts when a response is structurally unusable."""

    def __init__(
        self, message: str, attempts: list[dict[str, Any]]
    ) -> None:
        super().__init__(message)
        self.attempts = attempts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair only technical failures in an existing ReasoningLLM run, "
            "then add a separate Feedback -> Refinement baseline."
        )
    )
    parser.add_argument("--source_run_dir", required=True)
    parser.add_argument(
        "--model_config",
        default=str(DEFAULT_MODEL_CONFIG),
    )
    parser.add_argument("--env_file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "4")))
    parser.add_argument(
        "--request_interval_s",
        type=float,
        default=float(os.getenv("REQUEST_INTERVAL_S", "0")),
    )
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed repair, feedback, and refinement phases.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.workers <= 0:
        raise ValueError(f"--workers must be positive, got {args.workers}")
    if args.request_interval_s < 0:
        raise ValueError(
            "--request_interval_s must be non-negative, got "
            f"{args.request_interval_s}"
        )
    if args.max_retries < 0:
        raise ValueError(
            f"--max_retries must be non-negative, got {args.max_retries}"
        )


def build_feedback_messages(
    cot_prompt: str, initial_response: str
) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": cot_prompt},
        {"role": "assistant", "content": initial_response},
        {"role": "user", "content": FEEDBACK_INSTRUCTION},
    ]


def build_refinement_messages(
    cot_prompt: str,
    initial_response: str,
    feedback_response: str,
) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": cot_prompt},
        {"role": "assistant", "content": initial_response},
        {"role": "user", "content": FEEDBACK_INSTRUCTION},
        {"role": "assistant", "content": feedback_response},
        {"role": "user", "content": REFINEMENT_INSTRUCTION},
    ]


def source_state_sha256(state: dict[str, Any]) -> str:
    serialized = json.dumps(
        state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def normalize_answer_choice(
    raw_choice: dict[str, Any], *, index: int
) -> dict[str, Any]:
    response = str(
        raw_choice.get("response", raw_choice.get("content", ""))
    ).strip()
    answer, parse_status = baseline.extract_final_answer(response)
    return {
        "index": index,
        "response": response,
        "pred_answer": answer,
        "parse_status": parse_status,
        "finish_reason": raw_choice.get("finish_reason"),
    }


def answer_choice_is_valid(choice: dict[str, Any]) -> bool:
    normalized = normalize_answer_choice(
        choice, index=int(choice.get("index", 0))
    )
    return (
        normalized["finish_reason"] == "stop"
        and bool(normalized["response"])
        and bool(normalized["pred_answer"])
    )


def feedback_choice_is_valid(choice: dict[str, Any]) -> bool:
    response = str(
        choice.get("response", choice.get("content", ""))
    ).strip()
    return choice.get("finish_reason") == "stop" and bool(response)


def _source_cot_choices(source_state: dict[str, Any]) -> list[dict[str, Any]]:
    raw_choices = source_state.get("cot_choices") or []
    if len(raw_choices) != baseline.SC_SAMPLE_COUNT:
        return []
    return [
        normalize_answer_choice(choice, index=index)
        for index, choice in enumerate(raw_choices)
    ]


def merged_cot_choices(
    source_state: dict[str, Any], supplement_state: dict[str, Any]
) -> list[dict[str, Any]]:
    choices = _source_cot_choices(source_state)
    if not choices:
        return []
    replacements = supplement_state.get("cot_choice_replacements") or {}
    for raw_index, replacement in replacements.items():
        index = int(raw_index)
        if not 0 <= index < len(choices):
            raise ValueError(
                f"Replacement choice index {index} is outside "
                f"0..{len(choices) - 1}"
            )
        choices[index] = normalize_answer_choice(replacement, index=index)
    return choices


def invalid_cot_indices(choices: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, choice in enumerate(choices)
        if not answer_choice_is_valid(choice)
    ]


def _usage_from_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    return baseline.sum_usage(
        attempt.get("usage", {})
        for attempt in attempts
        if isinstance(attempt, dict)
    )


def _attempt_record(
    result: dict[str, Any], *, max_tokens: int, n: int
) -> dict[str, Any]:
    raw_choices = [
        {
            "index": int(choice.get("index", index)),
            "response": str(choice.get("content", "")).strip(),
            "finish_reason": choice.get("finish_reason"),
        }
        for index, choice in enumerate(result.get("choices") or [])
    ]
    return {
        "max_tokens": max_tokens,
        "n": n,
        "http_status": result.get("http_status"),
        "returned_model": result.get("model", ""),
        "choices": raw_choices,
        "usage": result.get("usage", {}),
    }


def _request_kwargs(
    config: dict[str, Any], *, max_tokens: int
) -> dict[str, Any]:
    return {
        "api_key": config["api_key"],
        "base_url": config["base_url"],
        "model_name": config["model_name"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens": max_tokens,
        "frequency_penalty": config["frequency_penalty"],
        "presence_penalty": config["presence_penalty"],
        "request_timeout": config["request_timeout"],
        "max_retries": config["max_retries"],
        "request_limiter": config["request_limiter"],
    }


def _send_single_with_length_retry(
    *,
    messages: list[dict[str, str]],
    config: dict[str, Any],
    validator: Callable[[dict[str, Any]], bool],
    initial_max_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    result = baseline.send_chat_completion(
        messages=messages,
        n=1,
        **_request_kwargs(config, max_tokens=initial_max_tokens),
    )
    attempts.append(
        _attempt_record(result, max_tokens=initial_max_tokens, n=1)
    )
    if len(result.get("choices") or []) != 1:
        raise InvalidGenerationError(
            (
                "Expected exactly one choice, got "
                f"{len(result.get('choices') or [])}"
            ),
            attempts,
        )

    raw_choice = result["choices"][0]
    content = str(raw_choice.get("content", "")).strip()
    should_retry = (
        initial_max_tokens < REPAIR_MAX_TOKENS
        and not content
        and raw_choice.get("finish_reason") == "length"
    )
    if should_retry:
        result = baseline.send_chat_completion(
            messages=messages,
            n=1,
            **_request_kwargs(config, max_tokens=REPAIR_MAX_TOKENS),
        )
        attempts.append(
            _attempt_record(result, max_tokens=REPAIR_MAX_TOKENS, n=1)
        )
        if len(result.get("choices") or []) != 1:
            raise InvalidGenerationError(
                (
                    "Length retry expected exactly one choice, got "
                    f"{len(result.get('choices') or [])}"
                ),
                attempts,
            )
        raw_choice = result["choices"][0]

    normalized = {
        "response": str(raw_choice.get("content", "")).strip(),
        "content": str(raw_choice.get("content", "")).strip(),
        "finish_reason": raw_choice.get("finish_reason"),
    }
    if not validator(normalized):
        raise InvalidGenerationError(
            (
                "Generation remained invalid after the allowed request(s): "
                f"finish_reason={normalized['finish_reason']!r} "
                f"content_empty={not bool(normalized['content'])}"
            ),
            attempts,
        )
    return normalized, attempts


def _source_correction_choice(source_state: dict[str, Any]) -> dict[str, Any]:
    return normalize_answer_choice(
        {
            "response": source_state.get("self_correction_response", ""),
            "finish_reason": source_state.get(
                "self_correction_finish_reason"
            ),
        },
        index=0,
    )


def _new_state(
    source_state: dict[str, Any],
    source_checkpoint: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SUPPLEMENT_SCHEMA_VERSION,
        "prompt_version": SUPPLEMENT_PROMPT_VERSION,
        "id": str(source_state["id"]),
        "source_checkpoint": str(source_checkpoint),
        "source_state_sha256": source_state_sha256(source_state),
        "model_name": config["model_name"],
        "generation_config": {
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "max_tokens": config["max_tokens"],
            "length_retry_max_tokens": REPAIR_MAX_TOKENS,
            "frequency_penalty": config["frequency_penalty"],
            "presence_penalty": config["presence_penalty"],
        },
        "cot_choice_replacements": {},
        "cot_repair_attempts": [],
        "self_correction_repair_attempts": [],
        "feedback_attempts": [],
        "refinement_attempts": [],
    }


def validate_resume_state(
    source_state: dict[str, Any],
    supplement_state: dict[str, Any],
    source_checkpoint: Path,
    config: dict[str, Any],
) -> None:
    expected = _new_state(source_state, source_checkpoint, config)
    immutable_fields = (
        "schema_version",
        "prompt_version",
        "id",
        "source_checkpoint",
        "source_state_sha256",
        "model_name",
        "generation_config",
    )
    mismatches = [
        field
        for field in immutable_fields
        if supplement_state.get(field) != expected[field]
    ]
    if mismatches:
        raise ValueError(
            f"Cannot resume {source_state['id']}: supplement differs in "
            + ", ".join(mismatches)
        )


def _append_attempts(
    state: dict[str, Any], field: str, attempts: list[dict[str, Any]]
) -> None:
    existing = list(state.get(field) or [])
    existing.extend(attempts)
    state[field] = existing


def _mark_source_unavailable(
    state: dict[str, Any],
    source_state: dict[str, Any],
) -> dict[str, Any]:
    error = str(
        source_state.get("cot_error")
        or "Source checkpoint has no complete three-choice CoT generation"
    )
    state.update(
        {
            "source_status": "unavailable",
            "source_error": error,
            "cot_repair_status": "blocked",
            "cot_repair_error": error,
            "self_correction_repair_status": "blocked",
            "self_correction_repair_error": error,
            "feedback_status": "blocked",
            "feedback_error": error,
            "refinement_status": "blocked",
            "refinement_error": error,
        }
    )
    return state


def _repair_cot(
    *,
    cot_prompt: str,
    source_state: dict[str, Any],
    state: dict[str, Any],
    config: dict[str, Any],
    writer: baseline.CheckpointWriter,
) -> bool:
    choices = merged_cot_choices(source_state, state)
    invalid_indices = invalid_cot_indices(choices)
    if not invalid_indices:
        state.update(
            {
                "cot_repair_status": "success",
                "cot_repair_error": "",
                "cot_repaired_indices": sorted(
                    int(index)
                    for index in (
                        state.get("cot_choice_replacements") or {}
                    )
                ),
            }
        )
        writer.append(state)
        return True

    try:
        result = baseline.send_chat_completion(
            messages=[{"role": "user", "content": cot_prompt}],
            n=len(invalid_indices),
            **_request_kwargs(config, max_tokens=REPAIR_MAX_TOKENS),
        )
        attempt = _attempt_record(
            result,
            max_tokens=REPAIR_MAX_TOKENS,
            n=len(invalid_indices),
        )
        _append_attempts(state, "cot_repair_attempts", [attempt])
        raw_choices = result.get("choices") or []
        valid_replacements = [
            normalize_answer_choice(choice, index=0)
            for choice in raw_choices
            if answer_choice_is_valid(
                {
                    "index": 0,
                    "response": choice.get("content", ""),
                    "finish_reason": choice.get("finish_reason"),
                }
            )
        ]
        replacements = dict(state.get("cot_choice_replacements") or {})
        for target_index, replacement in zip(
            invalid_indices, valid_replacements
        ):
            replacement["index"] = target_index
            replacements[str(target_index)] = replacement
        state["cot_choice_replacements"] = replacements

        remaining = invalid_cot_indices(
            merged_cot_choices(source_state, state)
        )
        if remaining:
            raise RuntimeError(
                f"CoT repair returned {len(valid_replacements)} valid "
                f"replacement(s); invalid indices remain: {remaining}"
            )
        state.update(
            {
                "cot_repair_status": "success",
                "cot_repair_error": "",
                "cot_repaired_indices": sorted(
                    int(index) for index in replacements
                ),
            }
        )
    except Exception as exc:
        state.update(
            {
                "cot_repair_status": "fail",
                "cot_repair_error": str(exc),
            }
        )
    writer.append(state)
    return state.get("cot_repair_status") == "success"


def _repair_self_correction(
    *,
    cot_prompt: str,
    source_state: dict[str, Any],
    state: dict[str, Any],
    config: dict[str, Any],
    writer: baseline.CheckpointWriter,
) -> bool:
    repaired_choices = merged_cot_choices(source_state, state)
    initial_response = repaired_choices[0]["response"]
    choice_zero_replaced = "0" in (
        state.get("cot_choice_replacements") or {}
    )
    source_choice = _source_correction_choice(source_state)
    source_correction_valid = (
        source_state.get("self_correction_status") == "success"
        and answer_choice_is_valid(source_choice)
    )

    if not choice_zero_replaced and source_correction_valid:
        state.update(
            {
                "self_correction_repair_status": "success",
                "self_correction_repair_mode": "reused",
                "self_correction_repair_error": "",
            }
        )
        writer.append(state)
        return True

    if state.get("self_correction_repair_status") == "success":
        return True

    initial_max_tokens = config["max_tokens"]
    if (
        not choice_zero_replaced
        and not source_choice["response"]
        and source_choice.get("finish_reason") == "length"
    ):
        initial_max_tokens = REPAIR_MAX_TOKENS

    try:
        raw_choice, attempts = _send_single_with_length_retry(
            messages=baseline.build_self_correction_messages(
                cot_prompt, initial_response
            ),
            config=config,
            validator=answer_choice_is_valid,
            initial_max_tokens=initial_max_tokens,
        )
        _append_attempts(
            state, "self_correction_repair_attempts", attempts
        )
        normalized = normalize_answer_choice(raw_choice, index=0)
        state.update(
            {
                "self_correction_repair_status": "success",
                "self_correction_repair_mode": "regenerated",
                "self_correction_repair_error": "",
                "self_correction_response": normalized["response"],
                "self_correction_pred_answer": normalized["pred_answer"],
                "self_correction_parse_status": normalized["parse_status"],
                "self_correction_finish_reason": normalized[
                    "finish_reason"
                ],
            }
        )
    except Exception as exc:
        if isinstance(exc, InvalidGenerationError):
            _append_attempts(
                state,
                "self_correction_repair_attempts",
                exc.attempts,
            )
        state.update(
            {
                "self_correction_repair_status": "fail",
                "self_correction_repair_mode": "regenerated",
                "self_correction_repair_error": str(exc),
            }
        )
    writer.append(state)
    return state.get("self_correction_repair_status") == "success"


def _generate_feedback(
    *,
    cot_prompt: str,
    initial_response: str,
    state: dict[str, Any],
    config: dict[str, Any],
    writer: baseline.CheckpointWriter,
) -> bool:
    if state.get("feedback_status") == "success":
        return True
    try:
        raw_choice, attempts = _send_single_with_length_retry(
            messages=build_feedback_messages(cot_prompt, initial_response),
            config=config,
            validator=feedback_choice_is_valid,
            initial_max_tokens=config["max_tokens"],
        )
        _append_attempts(state, "feedback_attempts", attempts)
        state.update(
            {
                "feedback_status": "success",
                "feedback_error": "",
                "feedback_response": raw_choice["response"],
                "feedback_finish_reason": raw_choice["finish_reason"],
            }
        )
    except Exception as exc:
        if isinstance(exc, InvalidGenerationError):
            _append_attempts(state, "feedback_attempts", exc.attempts)
        state.update(
            {
                "feedback_status": "fail",
                "feedback_error": str(exc),
            }
        )
    writer.append(state)
    return state.get("feedback_status") == "success"


def _generate_refinement(
    *,
    cot_prompt: str,
    initial_response: str,
    state: dict[str, Any],
    config: dict[str, Any],
    writer: baseline.CheckpointWriter,
) -> bool:
    if state.get("refinement_status") == "success":
        return True
    try:
        raw_choice, attempts = _send_single_with_length_retry(
            messages=build_refinement_messages(
                cot_prompt,
                initial_response,
                str(state["feedback_response"]),
            ),
            config=config,
            validator=answer_choice_is_valid,
            initial_max_tokens=config["max_tokens"],
        )
        _append_attempts(state, "refinement_attempts", attempts)
        normalized = normalize_answer_choice(raw_choice, index=0)
        state.update(
            {
                "refinement_status": "success",
                "refinement_error": "",
                "refinement_response": normalized["response"],
                "refinement_pred_answer": normalized["pred_answer"],
                "refinement_parse_status": normalized["parse_status"],
                "refinement_finish_reason": normalized["finish_reason"],
            }
        )
    except Exception as exc:
        if isinstance(exc, InvalidGenerationError):
            _append_attempts(state, "refinement_attempts", exc.attempts)
        state.update(
            {
                "refinement_status": "fail",
                "refinement_error": str(exc),
            }
        )
    writer.append(state)
    return state.get("refinement_status") == "success"


def _update_usage(state: dict[str, Any]) -> None:
    for phase in (
        "cot_repair",
        "self_correction_repair",
        "feedback",
        "refinement",
    ):
        state[f"{phase}_usage"] = _usage_from_attempts(
            list(state.get(f"{phase}_attempts") or [])
        )
        state[f"{phase}_request_count"] = len(
            state.get(f"{phase}_attempts") or []
        )
    state["total_supplement_usage"] = baseline.sum_usage(
        [
            state.get("cot_repair_usage", {}),
            state.get("self_correction_repair_usage", {}),
            state.get("feedback_usage", {}),
            state.get("refinement_usage", {}),
        ]
    )


def run_one(
    source_state: dict[str, Any],
    existing_state: dict[str, Any] | None,
    source_checkpoint: Path,
    config: dict[str, Any],
    writer: baseline.CheckpointWriter,
) -> dict[str, Any]:
    item = source_state.get("item")
    if not isinstance(item, dict):
        raise ValueError(f"Source state {source_state.get('id')} has no item")

    if existing_state:
        validate_resume_state(
            source_state,
            existing_state,
            source_checkpoint,
            config,
        )
        state = dict(existing_state)
    else:
        state = _new_state(source_state, source_checkpoint, config)

    source_choices = _source_cot_choices(source_state)
    if source_state.get("cot_status") != "success" or not source_choices:
        result = _mark_source_unavailable(
            state, source_state
        )
        _update_usage(result)
        writer.append(result)
        return result

    state.update({"source_status": "available", "source_error": ""})
    cot_prompt = baseline.build_cot_prompt(
        baseline.read_question(item),
        baseline.read_table(item, max_table_chars=0)[0],
    )
    expected_sha256 = hashlib.sha256(
        cot_prompt.encode("utf-8")
    ).hexdigest()
    if source_state.get("cot_prompt_sha256") != expected_sha256:
        raise ValueError(
            f"Source prompt hash mismatch for {source_state['id']}"
        )

    if state.get("cot_repair_status") != "success":
        if not _repair_cot(
            cot_prompt=cot_prompt,
            source_state=source_state,
            state=state,
            config=config,
            writer=writer,
        ):
            _update_usage(state)
            writer.append(state)
            return state

    repaired_choices = merged_cot_choices(source_state, state)
    initial_response = repaired_choices[0]["response"]

    if state.get("self_correction_repair_status") != "success":
        _repair_self_correction(
            cot_prompt=cot_prompt,
            source_state=source_state,
            state=state,
            config=config,
            writer=writer,
        )

    if not _generate_feedback(
        cot_prompt=cot_prompt,
        initial_response=initial_response,
        state=state,
        config=config,
        writer=writer,
    ):
        _update_usage(state)
        writer.append(state)
        return state

    _generate_refinement(
        cot_prompt=cot_prompt,
        initial_response=initial_response,
        state=state,
        config=config,
        writer=writer,
    )
    _update_usage(state)
    writer.append(state)
    return state


def state_is_terminal(state: dict[str, Any] | None) -> bool:
    if not state:
        return False
    if state.get("source_status") == "unavailable":
        return True
    return (
        state.get("cot_repair_status") == "success"
        and state.get("self_correction_repair_status") == "success"
        and state.get("feedback_status") == "success"
        and state.get("refinement_status") == "success"
    )


def _ordered_source_states(
    source_run_dir: Path,
    source_states: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    original_results = source_run_dir / "zero_shot_cot" / "results.jsonl"
    if not original_results.is_file():
        return list(source_states.values())

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    with original_results.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            example_id = str(json.loads(line).get("id", ""))
            if example_id in source_states and example_id not in seen:
                ordered.append(source_states[example_id])
                seen.add(example_id)
    if seen != set(source_states):
        missing = sorted(set(source_states) - seen)
        raise ValueError(
            "Original zero-shot results do not contain every checkpoint id; "
            f"missing {len(missing)} id(s)"
        )
    return ordered


def _common_result(
    item: dict[str, Any],
    source_state: dict[str, Any],
    supplement_state: dict[str, Any] | None,
    method: str,
    source_run_dir: Path,
) -> dict[str, Any]:
    result = dict(item)
    is_self_refine = method == "self_refine_1"
    result.update(
        {
            "method": method,
            "model_name": source_state.get("model_name", ""),
            "prompt_version": (
                SUPPLEMENT_PROMPT_VERSION
                if is_self_refine
                else source_state.get("prompt_version", "")
            ),
            "supplement_prompt_version": SUPPLEMENT_PROMPT_VERSION,
            "source_prompt_version": source_state.get("prompt_version", ""),
            "temperature": source_state.get("generation_config", {}).get(
                "temperature", 0.6
            ),
            "generation_config": source_state.get(
                "generation_config", {}
            ),
            "supplement_generation_config": (
                supplement_state or {}
            ).get("generation_config", {}),
            "source_run_dir": str(source_run_dir),
            "table_truncated": False,
            "technical_repair_only": not is_self_refine,
            "repair_selection_is_gold_independent": True,
            "source_status": (supplement_state or {}).get(
                "source_status", "not_started"
            ),
        }
    )
    return result


def _merged_correction(
    source_state: dict[str, Any], supplement_state: dict[str, Any]
) -> dict[str, Any]:
    if (
        supplement_state.get("self_correction_repair_mode")
        == "regenerated"
        and supplement_state.get("self_correction_repair_status")
        == "success"
    ):
        return {
            "response": supplement_state.get(
                "self_correction_response", ""
            ),
            "pred_answer": supplement_state.get(
                "self_correction_pred_answer", ""
            ),
            "parse_status": supplement_state.get(
                "self_correction_parse_status", ""
            ),
            "finish_reason": supplement_state.get(
                "self_correction_finish_reason"
            ),
        }
    return _source_correction_choice(source_state)


def materialize_results(
    ordered_source_states: list[dict[str, Any]],
    supplement_states: dict[str, dict[str, Any]],
    source_run_dir: Path,
) -> dict[str, Any]:
    method_paths = {
        "zero_shot_cot_repaired": source_run_dir
        / "repaired_baselines"
        / "zero_shot_cot"
        / "results.jsonl",
        "self_consistency_3_repaired": source_run_dir
        / "repaired_baselines"
        / "self_consistency_3"
        / "results.jsonl",
        "single_pass_self_correction_1_repaired": source_run_dir
        / "repaired_baselines"
        / "single_pass_self_correction_1"
        / "results.jsonl",
        "self_refine_1": source_run_dir
        / "self_refine_1"
        / "results.jsonl",
    }
    for path in method_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    output_files = {
        method: path.open("w", encoding="utf-8")
        for method, path in method_paths.items()
    }
    coverage = {
        method: {"success": 0, "fail": 0}
        for method in method_paths
    }
    try:
        for source_state in ordered_source_states:
            example_id = str(source_state["id"])
            item = source_state["item"]
            supplement_state = supplement_states.get(example_id) or {}
            choices = merged_cot_choices(source_state, supplement_state)
            cot_success = (
                supplement_state.get("cot_repair_status") == "success"
                and len(choices) == baseline.SC_SAMPLE_COUNT
                and not invalid_cot_indices(choices)
            )
            source_error = str(
                supplement_state.get("source_error")
                or supplement_state.get("cot_repair_error")
                or "supplement_not_completed"
            )

            zero = _common_result(
                item,
                source_state,
                supplement_state,
                "zero_shot_cot_repaired",
                source_run_dir,
            )
            zero.update(
                {
                    "response": choices[0]["response"] if cot_success else "",
                    "pred_answer": (
                        choices[0]["pred_answer"] if cot_success else ""
                    ),
                    "execute_status": "success" if cot_success else "fail",
                    "parse_status": (
                        choices[0]["parse_status"] if cot_success else ""
                    ),
                    "finish_reason": (
                        choices[0]["finish_reason"] if cot_success else None
                    ),
                    "repaired_choice_indices": supplement_state.get(
                        "cot_repaired_indices", []
                    ),
                    "error": "" if cot_success else source_error,
                    "source_usage": source_state.get("cot_usage", {}),
                    "repair_usage": supplement_state.get(
                        "cot_repair_usage", {}
                    ),
                    "usage": baseline.sum_usage(
                        [
                            source_state.get("cot_usage", {}),
                            supplement_state.get(
                                "cot_repair_usage", {}
                            ),
                        ]
                    ),
                }
            )

            sc = _common_result(
                item,
                source_state,
                supplement_state,
                "self_consistency_3_repaired",
                source_run_dir,
            )
            sc_info = (
                baseline.select_self_consistent_answer(choices)
                if cot_success
                else {}
            )
            sc.update(
                {
                    "response": (
                        [choice["response"] for choice in choices]
                        if cot_success
                        else []
                    ),
                    "pred_answer": sc_info.get("pred_answer", ""),
                    "execute_status": "success" if cot_success else "fail",
                    "reasoning_samples": choices if cot_success else [],
                    "selected_choice_index": sc_info.get(
                        "selected_choice_index"
                    ),
                    "all_different_tie": sc_info.get(
                        "all_different_tie"
                    ),
                    "vote_counts": sc_info.get("vote_counts", []),
                    "repaired_choice_indices": supplement_state.get(
                        "cot_repaired_indices", []
                    ),
                    "error": "" if cot_success else source_error,
                    "source_usage": source_state.get("cot_usage", {}),
                    "repair_usage": supplement_state.get(
                        "cot_repair_usage", {}
                    ),
                    "usage": baseline.sum_usage(
                        [
                            source_state.get("cot_usage", {}),
                            supplement_state.get(
                                "cot_repair_usage", {}
                            ),
                        ]
                    ),
                }
            )

            correction_success = (
                cot_success
                and supplement_state.get(
                    "self_correction_repair_status"
                )
                == "success"
            )
            correction = _common_result(
                item,
                source_state,
                supplement_state,
                "single_pass_self_correction_1_repaired",
                source_run_dir,
            )
            correction_choice = (
                _merged_correction(source_state, supplement_state)
                if correction_success
                else {}
            )
            correction.update(
                {
                    "response": correction_choice.get("response", ""),
                    "pred_answer": correction_choice.get(
                        "pred_answer", ""
                    ),
                    "execute_status": (
                        "success" if correction_success else "fail"
                    ),
                    "parse_status": correction_choice.get(
                        "parse_status", ""
                    ),
                    "finish_reason": correction_choice.get(
                        "finish_reason"
                    ),
                    "initial_cot_response": (
                        choices[0]["response"] if cot_success else ""
                    ),
                    "initial_cot_answer": (
                        choices[0]["pred_answer"] if cot_success else ""
                    ),
                    "repair_mode": supplement_state.get(
                        "self_correction_repair_mode", ""
                    ),
                    "error": (
                        ""
                        if correction_success
                        else str(
                            supplement_state.get(
                                "self_correction_repair_error"
                            )
                            or source_error
                        )
                    ),
                    "source_usage": source_state.get(
                        "self_correction_usage", {}
                    ),
                    "repair_usage": supplement_state.get(
                        "self_correction_repair_usage", {}
                    ),
                    "usage": baseline.sum_usage(
                        [
                            source_state.get(
                                "self_correction_usage", {}
                            ),
                            supplement_state.get(
                                "self_correction_repair_usage", {}
                            ),
                        ]
                    ),
                }
            )

            feedback_success = (
                cot_success
                and supplement_state.get("feedback_status") == "success"
            )
            refine_success = (
                feedback_success
                and supplement_state.get("refinement_status") == "success"
            )
            refine = _common_result(
                item,
                source_state,
                supplement_state,
                "self_refine_1",
                source_run_dir,
            )
            refine.update(
                {
                    "response": (
                        supplement_state.get("refinement_response", "")
                        if refine_success
                        else ""
                    ),
                    "pred_answer": (
                        supplement_state.get(
                            "refinement_pred_answer", ""
                        )
                        if refine_success
                        else ""
                    ),
                    "execute_status": (
                        "success" if refine_success else "fail"
                    ),
                    "parse_status": (
                        supplement_state.get(
                            "refinement_parse_status", ""
                        )
                        if refine_success
                        else ""
                    ),
                    "finish_reason": supplement_state.get(
                        "refinement_finish_reason"
                    ),
                    "initial_cot_response": (
                        choices[0]["response"] if cot_success else ""
                    ),
                    "initial_cot_answer": (
                        choices[0]["pred_answer"] if cot_success else ""
                    ),
                    "feedback_response": (
                        supplement_state.get("feedback_response", "")
                        if feedback_success
                        else ""
                    ),
                    "feedback_finish_reason": supplement_state.get(
                        "feedback_finish_reason"
                    ),
                    "error": (
                        ""
                        if refine_success
                        else str(
                            supplement_state.get("refinement_error")
                            or supplement_state.get("feedback_error")
                            or source_error
                        )
                    ),
                    "feedback_usage": supplement_state.get(
                        "feedback_usage", {}
                    ),
                    "refinement_usage": supplement_state.get(
                        "refinement_usage", {}
                    ),
                    "initial_cot_source_usage": source_state.get(
                        "cot_usage", {}
                    ),
                    "initial_cot_repair_usage": supplement_state.get(
                        "cot_repair_usage", {}
                    ),
                    "usage": baseline.sum_usage(
                        [
                            source_state.get("cot_usage", {}),
                            supplement_state.get(
                                "cot_repair_usage", {}
                            ),
                            supplement_state.get(
                                "feedback_usage", {}
                            ),
                            supplement_state.get(
                                "refinement_usage", {}
                            ),
                        ]
                    ),
                    "all_supplement_usage_for_example": supplement_state.get(
                        "total_supplement_usage", {}
                    ),
                }
            )

            results = {
                "zero_shot_cot_repaired": zero,
                "self_consistency_3_repaired": sc,
                "single_pass_self_correction_1_repaired": correction,
                "self_refine_1": refine,
            }
            for method, result in results.items():
                output_files[method].write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
                coverage[method][
                    "success"
                    if result["execute_status"] == "success"
                    else "fail"
                ] += 1
    finally:
        for output_file in output_files.values():
            output_file.close()

    summary = {
        "source_run_dir": str(source_run_dir),
        "total_examples": len(ordered_source_states),
        "coverage": coverage,
        "output_paths": {
            method: str(path) for method, path in method_paths.items()
        },
        "supplement_usage": baseline.sum_usage(
            state.get("total_supplement_usage", {})
            for state in supplement_states.values()
        ),
        "supplement_http_requests": sum(
            sum(
                int(state.get(f"{phase}_request_count", 0) or 0)
                for phase in (
                    "cot_repair",
                    "self_correction_repair",
                    "feedback",
                    "refinement",
                )
            )
            for state in supplement_states.values()
        ),
    }
    summary_path = source_run_dir / "self_refine_supplement_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    args = parse_args()
    _validate_args(args)

    source_run_dir = Path(args.source_run_dir).expanduser().resolve()
    source_checkpoint = source_run_dir / "checkpoint.jsonl"
    supplement_checkpoint = (
        source_run_dir / SUPPLEMENT_CHECKPOINT_NAME
    )
    if not source_checkpoint.is_file():
        raise FileNotFoundError(
            f"Source checkpoint does not exist: {source_checkpoint}"
        )
    if supplement_checkpoint.exists() and not args.resume:
        raise FileExistsError(
            f"Supplement checkpoint already exists: "
            f"{supplement_checkpoint}. Use --resume."
        )

    env_file = Path(args.env_file).expanduser().resolve()
    load_env_file(env_file)
    model_config = load_model_config(
        resolve_path(args.model_config), env_file
    )
    ensure_local_base_url_bypasses_proxy(model_config["base_url"])

    source_states = baseline.load_checkpoint(source_checkpoint)
    ordered_states = _ordered_source_states(
        source_run_dir, source_states
    )
    if not ordered_states:
        raise ValueError("Source checkpoint contains no examples")

    source_configs = {
        json.dumps(
            state.get("generation_config", {}),
            sort_keys=True,
        )
        for state in ordered_states
    }
    source_models = {
        str(state.get("model_name", "")) for state in ordered_states
    }
    if len(source_configs) != 1 or len(source_models) != 1:
        raise ValueError(
            "Source checkpoint mixes generation configurations or models"
        )
    source_generation_config = dict(
        ordered_states[0].get("generation_config") or {}
    )
    if model_config["model_name"] not in source_models:
        raise ValueError(
            f"Configured model {model_config['model_name']!r} does not "
            f"match source model {next(iter(source_models))!r}"
        )
    expected_config = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 2000,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }
    mismatches = [
        key
        for key, expected in expected_config.items()
        if source_generation_config.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "Source run does not use the required matched parameters: "
            + ", ".join(mismatches)
        )

    existing_states = (
        baseline.load_checkpoint(supplement_checkpoint)
        if args.resume
        else {}
    )
    request_limiter = RequestStartLimiter(args.request_interval_s)
    config: dict[str, Any] = {
        **model_config,
        **expected_config,
        "request_timeout": args.request_timeout,
        "max_retries": args.max_retries,
        "request_limiter": request_limiter,
    }
    for source_state in ordered_states:
        existing = existing_states.get(str(source_state["id"]))
        if existing:
            validate_resume_state(
                source_state,
                existing,
                source_checkpoint,
                config,
            )

    pending = [
        state
        for state in ordered_states
        if not state_is_terminal(
            existing_states.get(str(state["id"]))
        )
    ]
    print(
        f"Self-Refine supplement: examples={len(ordered_states)} "
        f"pending={len(pending)} model={model_config['model_name']} "
        f"workers={args.workers} request_interval_s="
        f"{args.request_interval_s}",
        flush=True,
    )

    writer = baseline.CheckpointWriter(supplement_checkpoint)
    completed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = {
                executor.submit(
                    run_one,
                    source_state,
                    existing_states.get(str(source_state["id"])),
                    source_checkpoint,
                    config,
                    writer,
                ): str(source_state["id"])
                for source_state in pending
            }
            for future in concurrent.futures.as_completed(futures):
                state = future.result()
                completed += 1
                print(
                    f"[PROGRESS] {completed}/{len(pending)} "
                    f"id={state['id']} "
                    f"cot_repair={state.get('cot_repair_status')} "
                    f"correction={state.get('self_correction_repair_status')} "
                    f"feedback={state.get('feedback_status')} "
                    f"refinement={state.get('refinement_status')}",
                    flush=True,
                )
    finally:
        final_supplement_states = baseline.load_checkpoint(
            supplement_checkpoint
        )
        summary = materialize_results(
            ordered_states,
            final_supplement_states,
            source_run_dir,
        )
        print(
            json.dumps(summary, ensure_ascii=False, indent=2),
            flush=True,
        )


if __name__ == "__main__":
    main()
