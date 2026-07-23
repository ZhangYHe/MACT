#!/usr/bin/env python3
"""Evaluate CRT-QA results by the operation and reasoning types in Table 3."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from evaluate_wtq_official import (
    check_denotation,
    load_results,
    prediction_to_items,
    to_value_list,
)


DEFAULT_CRT_DATASET_PATH = "/home/zhangyunhe/nas/dataset/CRT-QA/CRT-QA"

OPERATION_TYPES = ("Index", "Sort", "Group", "Filter")
REASONING_TYPES = ("GRO", "CAT", "TEM", "AGG", "ARI", "SPA", "QUA", "OTH")
DIRECTNESS_TYPES = ("Explicit", "Implicit")
COMPOSITION_TYPES = ("Bridging", "Intersection", "Comparison")

OPERATION_TYPE_MAP = {
    "indexing": "Index",
    "sorting": "Sort",
    "grouping": "Group",
    "filter": "Filter",
    "filtering": "Filter",
}

REASONING_TYPE_MAP = {
    "grounding": "GRO",
    "auto-categorization": "CAT",
    "temporal reasoning": "TEM",
    "aggregating": "AGG",
    "arithmetic": "ARI",
    "geographical/spatial reasoning": "SPA",
    "spatial/geographical reasoning": "SPA",
    "reasoning with quantifiers": "QUA",
    "other commonsense reasoning": "OTH",
}

@dataclass(frozen=True)
class Annotation:
    operations: frozenset[str]
    reasoning: frozenset[str]
    directness: str | None
    composition_type: str | None
    unknown_operations: tuple[str, ...]
    unknown_reasoning: tuple[str, ...]
    unknown_step_types: tuple[str, ...]
    missing_steps: tuple[str, ...]


def normalize_type_label(value: object) -> str:
    """Normalize taxonomy labels without changing their meaning."""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    return re.sub(r"\s+", " ", text)


def normalize_metadata_category(value: object, categories: tuple[str, ...]) -> str | None:
    """Canonicalize known metadata labels while preserving unsupported values."""
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    known_categories = {category.casefold(): category for category in categories}
    return known_categories.get(text.casefold(), text)


def normalize_strict_item(value: object) -> str:
    """Apply only Unicode, case, and whitespace normalization for strict EM."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def normalized_string_match(gold_items: Iterable[object], pred_items: Iterable[object]) -> bool:
    """Return ordered exact match after light string normalization."""
    normalized_gold = [normalize_strict_item(item) for item in gold_items]
    normalized_pred = [normalize_strict_item(item) for item in pred_items]
    return normalized_gold == normalized_pred


def resolve_dataset_json(crt_dataset_path: Path) -> Path:
    dataset_json = crt_dataset_path / "dataset.json" if crt_dataset_path.is_dir() else crt_dataset_path
    if not dataset_json.is_file():
        raise FileNotFoundError(f"CRT-QA dataset.json does not exist: {dataset_json}")
    return dataset_json.resolve()


def _classify_step(
    step_name: str,
    step: object,
    operations: set[str],
    reasoning: set[str],
    unknown_operations: list[str],
    unknown_reasoning: list[str],
    unknown_step_types: list[str],
    missing_steps: list[str],
) -> None:
    if not isinstance(step, dict):
        missing_steps.append(step_name)
        return

    raw_type = step.get("type")
    raw_name = step.get("name")
    if not str(raw_type or "").strip() or not str(raw_name or "").strip():
        missing_steps.append(step_name)
        return

    step_type = normalize_type_label(raw_type)
    type_name = normalize_type_label(raw_name)
    if step_type == "operation":
        category = OPERATION_TYPE_MAP.get(type_name)
        if category is None:
            unknown_operations.append(str(raw_name).strip())
        else:
            operations.add(category)
    elif step_type == "reasoning":
        category = REASONING_TYPE_MAP.get(type_name)
        if category is None:
            unknown_reasoning.append(str(raw_name).strip())
        else:
            reasoning.add(category)
    else:
        unknown_step_types.append(str(raw_type).strip())


def load_annotations(dataset_json: Path) -> dict[str, Annotation]:
    with dataset_json.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected CRT-QA dataset format in {dataset_json}: expected object")

    annotations: dict[str, Annotation] = {}
    for table_id, questions in data.items():
        if not isinstance(questions, list):
            raise ValueError(f"Unexpected question list for CRT-QA table {table_id!r}")
        for question_index, question in enumerate(questions):
            if not isinstance(question, dict):
                raise ValueError(
                    f"Unexpected CRT-QA question at {table_id}:{question_index}: expected object"
                )
            operations: set[str] = set()
            reasoning: set[str] = set()
            unknown_operations: list[str] = []
            unknown_reasoning: list[str] = []
            unknown_step_types: list[str] = []
            missing_steps: list[str] = []
            for step_index in range(1, 5):
                step_name = f"step{step_index}"
                _classify_step(
                    step_name,
                    question.get(step_name),
                    operations,
                    reasoning,
                    unknown_operations,
                    unknown_reasoning,
                    unknown_step_types,
                    missing_steps,
                )

            example_id = f"crt:answerable:{table_id}:{question_index}"
            if example_id in annotations:
                raise ValueError(f"Duplicate CRT-QA annotation ID: {example_id}")
            annotations[example_id] = Annotation(
                operations=frozenset(operations),
                reasoning=frozenset(reasoning),
                directness=normalize_metadata_category(
                    question.get("Directness"), DIRECTNESS_TYPES
                ),
                composition_type=normalize_metadata_category(
                    question.get("Composition Type"), COMPOSITION_TYPES
                ),
                unknown_operations=tuple(unknown_operations),
                unknown_reasoning=tuple(unknown_reasoning),
                unknown_step_types=tuple(unknown_step_types),
                missing_steps=tuple(missing_steps),
            )
    return annotations


def evaluate_result_rows(
    results: list[dict], annotations: dict[str, Annotation]
) -> tuple[dict, list[dict]]:
    seen_ids: dict[str, int] = {}
    details: list[dict] = []

    for line_number, item in enumerate(results, start=1):
        example_id = str(item.get("id", ""))
        if example_id in seen_ids:
            raise ValueError(
                f"Duplicate result ID {example_id!r} at rows "
                f"{seen_ids[example_id]} and {line_number}"
            )
        seen_ids[example_id] = line_number

        gold_items = prediction_to_items(item.get("answer"))
        pred_items = prediction_to_items(item.get("pred_answer"))
        errors: list[str] = []
        if "answer" not in item:
            errors.append("missing_answer")
        elif not gold_items:
            errors.append("empty_gold")

        valid_gold = not errors
        if valid_gold:
            denotation_em = check_denotation(
                to_value_list(gold_items),
                to_value_list(pred_items),
            )
            strict_em = normalized_string_match(gold_items, pred_items)
        else:
            denotation_em = None
            strict_em = None

        annotation = annotations.get(example_id)
        if annotation is None:
            errors.append("missing_annotation")
            operations: list[str] = []
            reasoning: list[str] = []
            directness = None
            composition_type = None
            unknown_operations: list[str] = []
            unknown_reasoning: list[str] = []
            unknown_step_types: list[str] = []
            missing_steps: list[str] = []
        else:
            operations = [name for name in OPERATION_TYPES if name in annotation.operations]
            reasoning = [name for name in REASONING_TYPES if name in annotation.reasoning]
            directness = annotation.directness
            composition_type = annotation.composition_type
            unknown_operations = list(annotation.unknown_operations)
            unknown_reasoning = list(annotation.unknown_reasoning)
            unknown_step_types = list(annotation.unknown_step_types)
            missing_steps = list(annotation.missing_steps)

        details.append(
            {
                "id": example_id,
                "valid_gold": valid_gold,
                "denotation_em": denotation_em,
                "normalized_string_em": strict_em,
                "gold_answer": item.get("answer"),
                "pred_answer": item.get("pred_answer"),
                "gold_items": [str(value) for value in gold_items],
                "pred_items": [str(value) for value in pred_items],
                "operation_types": operations,
                "reasoning_types": reasoning,
                "directness": directness,
                "composition_type": composition_type,
                "unknown_operation_types": unknown_operations,
                "unknown_reasoning_types": unknown_reasoning,
                "unknown_step_types": unknown_step_types,
                "missing_steps": missing_steps,
                "errors": errors,
            }
        )

    metrics = summarize(details, total_results=len(results))
    return metrics, details


def score_group(details: list[dict], metric_name: str) -> dict:
    valid_details = [detail for detail in details if detail["valid_gold"]]
    correct = sum(detail[metric_name] is True for detail in valid_details)
    evaluated = len(valid_details)
    return {
        "total_members": len(details),
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": round(correct / evaluated, 4) if evaluated else 0.0,
        "invalid_gold_count": len(details) - evaluated,
    }


def _summarize_metric(details: list[dict], metric_name: str) -> dict:
    by_operation = {
        category: score_group(
            [detail for detail in details if category in detail["operation_types"]],
            metric_name,
        )
        for category in OPERATION_TYPES
    }
    by_reasoning = {
        category: score_group(
            [detail for detail in details if category in detail["reasoning_types"]],
            metric_name,
        )
        for category in REASONING_TYPES
    }
    by_directness = {
        category: score_group(
            [detail for detail in details if detail["directness"] == category],
            metric_name,
        )
        for category in DIRECTNESS_TYPES
    }
    by_composition = {
        category: score_group(
            [detail for detail in details if detail["composition_type"] == category],
            metric_name,
        )
        for category in COMPOSITION_TYPES
    }
    return {
        "overall": score_group(details, metric_name),
        "by_operation": by_operation,
        "by_reasoning": by_reasoning,
        "by_directness": by_directness,
        "by_composition": by_composition,
    }


def _type_diagnostics(details: list[dict], field: str) -> dict[str, dict[str, int]]:
    step_counts: Counter[str] = Counter()
    question_counts: Counter[str] = Counter()
    for detail in details:
        values = detail[field]
        step_counts.update(values)
        question_counts.update(set(values))
    return {
        name: {
            "question_count": question_counts[name],
            "step_count": step_counts[name],
        }
        for name in sorted(step_counts)
    }


def _metadata_diagnostics(
    details: list[dict], field: str, categories: tuple[str, ...]
) -> tuple[list[str], dict[str, dict[str, object]]]:
    annotated_details = [
        detail for detail in details if "missing_annotation" not in detail["errors"]
    ]
    missing_ids = [
        detail["id"] for detail in annotated_details if detail[field] is None
    ]
    unsupported: dict[str, dict[str, object]] = {}
    unsupported_values = sorted(
        {
            detail[field]
            for detail in annotated_details
            if detail[field] is not None and detail[field] not in categories
        }
    )
    for value in unsupported_values:
        ids = [
            detail["id"] for detail in annotated_details if detail[field] == value
        ]
        unsupported[value] = {"question_count": len(ids), "ids": ids}
    return missing_ids, unsupported


def summarize(details: list[dict], total_results: int) -> dict:
    invalid_gold_ids = [detail["id"] for detail in details if not detail["valid_gold"]]
    empty_prediction_ids = [
        detail["id"] for detail in details if not detail["pred_items"]
    ]
    missing_annotation_ids = [
        detail["id"] for detail in details if "missing_annotation" in detail["errors"]
    ]
    missing_step_ids = [detail["id"] for detail in details if detail["missing_steps"]]
    unknown_step_type_counts = Counter(
        step_type
        for detail in details
        for step_type in detail["unknown_step_types"]
    )
    missing_directness_ids, unsupported_directness_types = _metadata_diagnostics(
        details, "directness", DIRECTNESS_TYPES
    )
    missing_composition_ids, unsupported_composition_types = _metadata_diagnostics(
        details, "composition_type", COMPOSITION_TYPES
    )

    return {
        "dataset": "crt",
        "total_results": total_results,
        "evaluated": total_results - len(invalid_gold_ids),
        "metrics": {
            "denotation_em": _summarize_metric(details, "denotation_em"),
            "normalized_string_em": _summarize_metric(
                details, "normalized_string_em"
            ),
        },
        "diagnostics": {
            "invalid_gold_count": len(invalid_gold_ids),
            "invalid_gold_ids": invalid_gold_ids,
            "empty_prediction_count": len(empty_prediction_ids),
            "empty_prediction_ids": empty_prediction_ids,
            "missing_annotation_count": len(missing_annotation_ids),
            "missing_annotation_ids": missing_annotation_ids,
            "missing_directness_count": len(missing_directness_ids),
            "missing_directness_ids": missing_directness_ids,
            "unsupported_directness_types": unsupported_directness_types,
            "missing_composition_count": len(missing_composition_ids),
            "missing_composition_ids": missing_composition_ids,
            "excluded_composition_types": unsupported_composition_types,
            "questions_with_missing_steps_count": len(missing_step_ids),
            "questions_with_missing_steps_ids": missing_step_ids,
            "missing_step_count": sum(len(detail["missing_steps"]) for detail in details),
            "unknown_operation_types": _type_diagnostics(
                details, "unknown_operation_types"
            ),
            "unknown_reasoning_types": _type_diagnostics(
                details, "unknown_reasoning_types"
            ),
            "unknown_step_types": dict(sorted(unknown_step_type_counts.items())),
        },
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _format_score(score: dict) -> str:
    return f"{score['accuracy']:.4f} ({score['correct']}/{score['evaluated']})"


def _format_percentage(score: dict) -> str:
    return f"{score['accuracy'] * 100:.2f}"


def write_markdown(path: Path, metrics: dict, result_jsonl: Path, dataset_json: Path) -> None:
    denotation = metrics["metrics"]["denotation_em"]
    strict = metrics["metrics"]["normalized_string_em"]
    metric_rows = (
        ("Denotation EM", denotation),
        ("Normalized string EM", strict),
    )

    diagnostics = metrics["diagnostics"]
    unsupported_directness_counts = {
        name: value["question_count"]
        for name, value in diagnostics["unsupported_directness_types"].items()
    }
    excluded_composition_counts = {
        name: value["question_count"]
        for name, value in diagnostics["excluded_composition_types"].items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        output_file.write("# CRT-QA Type EM Evaluation\n\n")
        output_file.write(f"- Results: `{result_jsonl}`\n")
        output_file.write(f"- Dataset: `{dataset_json}`\n")
        output_file.write(
            f"- Evaluated: {metrics['evaluated']}/{metrics['total_results']} "
            f"(invalid gold: {diagnostics['invalid_gold_count']})\n\n"
        )
        categories = (
            OPERATION_TYPES
            + REASONING_TYPES
            + DIRECTNESS_TYPES
            + COMPOSITION_TYPES
        )
        output_file.write(
            "| Metric | Overall | " + " | ".join(categories) + " |\n"
        )
        output_file.write(
            "| --- | ---: | " + " | ".join("---:" for _ in categories) + " |\n"
        )
        for metric_label, metric in metric_rows:
            operation_values = [
                _format_percentage(metric["by_operation"][category])
                for category in OPERATION_TYPES
            ]
            reasoning_values = [
                _format_percentage(metric["by_reasoning"][category])
                for category in REASONING_TYPES
            ]
            directness_values = [
                _format_percentage(metric["by_directness"][category])
                for category in DIRECTNESS_TYPES
            ]
            composition_values = [
                _format_percentage(metric["by_composition"][category])
                for category in COMPOSITION_TYPES
            ]
            values = (
                operation_values
                + reasoning_values
                + directness_values
                + composition_values
            )
            output_file.write(
                f"| {metric_label} | {_format_percentage(metric['overall'])} | "
                + " | ".join(values)
                + " |\n"
            )
        output_file.write("\n")
        output_file.write(
            "Values are EM percentages. Exact correct/evaluated counts are available "
            "in `crt_type_metrics.json`.\n"
        )
        count_metric = denotation
        count_groups = [
            ("Overall", count_metric["overall"]),
            *(
                (category, count_metric["by_operation"][category])
                for category in OPERATION_TYPES
            ),
            *(
                (category, count_metric["by_reasoning"][category])
                for category in REASONING_TYPES
            ),
            *(
                (category, count_metric["by_directness"][category])
                for category in DIRECTNESS_TYPES
            ),
            *(
                (category, count_metric["by_composition"][category])
                for category in COMPOSITION_TYPES
            ),
        ]
        output_file.write("\nQuestion counts (valid gold / total members):\n\n")
        for category, group in count_groups:
            output_file.write(
                f"- {category} {group['evaluated']}/{group['total_members']}\n"
            )

        output_file.write("\n## Diagnostics\n\n")
        output_file.write(
            f"- Empty predictions: {diagnostics['empty_prediction_count']}\n"
        )
        output_file.write(
            f"- Missing annotations: {diagnostics['missing_annotation_count']}\n"
        )
        output_file.write(
            f"- Missing Directness annotations: "
            f"{diagnostics['missing_directness_count']}\n"
        )
        output_file.write(
            "- Unsupported Directness types: "
            f"`{json.dumps(unsupported_directness_counts, ensure_ascii=False)}`\n"
        )
        output_file.write(
            f"- Missing Composition annotations: "
            f"{diagnostics['missing_composition_count']}\n"
        )
        output_file.write(
            "- Excluded Composition types: "
            f"`{json.dumps(excluded_composition_counts, ensure_ascii=False)}`\n"
        )
        output_file.write(
            f"- Missing step annotations: {diagnostics['missing_step_count']} steps "
            f"across {diagnostics['questions_with_missing_steps_count']} questions\n"
        )
        output_file.write(
            "- Unknown operation types: "
            f"`{json.dumps(diagnostics['unknown_operation_types'], ensure_ascii=False)}`\n"
        )
        output_file.write(
            "- Unknown reasoning types: "
            f"`{json.dumps(diagnostics['unknown_reasoning_types'], ensure_ascii=False)}`\n"
        )
        output_file.write(
            "- Unknown step types: "
            f"`{json.dumps(diagnostics['unknown_step_types'], ensure_ascii=False)}`\n"
        )


def _denotation_matches(answer: object, gold_answer: object) -> bool:
    gold_items = prediction_to_items(gold_answer)
    predicted_items = prediction_to_items(answer)
    if not gold_items:
        return False
    return check_denotation(
        to_value_list(gold_items),
        to_value_list(predicted_items),
    )


def _finish_candidates(history: object) -> list[str]:
    return re.findall(
        r"^Action\s+\d+:\s*Finish\[(.*)\]\s*$",
        str(history or ""),
        flags=re.MULTILINE,
    )


def build_replay_metrics(
    results: list[dict],
    annotations: dict[str, Annotation],
    baseline_results: list[dict] | None = None,
) -> dict:
    baseline_by_id = {
        str(item.get("id")): item for item in (baseline_results or [])
    }
    flips = Counter()
    status_counts = defaultdict(lambda: Counter(total=0, correct=0))
    tool_path_counts = defaultdict(lambda: Counter(total=0, correct=0))
    operation_counts = defaultdict(lambda: Counter(total=0, correct=0))
    accepted_wrong = 0
    rejected_correct = 0
    fallback_overrode_correct = 0
    direct_policy_correct = 0
    direct_policy_evaluated = 0
    raw_available = 0

    for item in results:
        if not prediction_to_items(item.get("answer")):
            continue
        current_correct = _denotation_matches(
            item.get("pred_answer"), item.get("answer"))
        candidates = _finish_candidates(item.get("history"))
        raw_answer = item.get("raw_pred_answer")
        if raw_answer in (None, "") and candidates:
            raw_answer = candidates[-1]
        if raw_answer not in (None, ""):
            raw_available += 1
            raw_correct = _denotation_matches(raw_answer, item.get("answer"))
            flips[(bool(raw_correct), bool(current_correct))] += 1

        status = str(item.get("run_status") or "unknown")
        status_counts[status]["total"] += 1
        status_counts[status]["correct"] += int(current_correct)
        tools = "+".join(dict.fromkeys(
            str(event.get("tool"))
            for event in (item.get("tool_events") or [])
            if event.get("tool")
        )) or "none"
        tool_path_counts[tools]["total"] += 1
        tool_path_counts[tools]["correct"] += int(current_correct)
        annotation = annotations.get(str(item.get("id")))
        for operation in (annotation.operations if annotation else []):
            operation_counts[operation]["total"] += 1
            operation_counts[operation]["correct"] += int(current_correct)

        if status == "finished" and not current_correct:
            accepted_wrong += 1
        any_correct_finish = any(
            _denotation_matches(candidate, item.get("answer"))
            for candidate in candidates
        )
        if not current_correct and any_correct_finish:
            rejected_correct += 1
        if (
            not current_correct
            and status.startswith("fallback")
            and candidates
            and _denotation_matches(candidates[-1], item.get("answer"))
        ):
            fallback_overrode_correct += 1

        verifier_failures = str(item.get("history") or "").count(
            "Final verification failed."
        )
        baseline_item = baseline_by_id.get(str(item.get("id")))
        selected_answer = item.get("pred_answer")
        if verifier_failures >= 2 and baseline_item is not None:
            selected_answer = baseline_item.get("pred_answer")
        if baseline_item is not None:
            direct_policy_evaluated += 1
            direct_policy_correct += int(
                _denotation_matches(selected_answer, item.get("answer"))
            )

    def summarize_groups(groups: dict[str, Counter]) -> dict:
        return {
            key: {
                "correct": counts["correct"],
                "evaluated": counts["total"],
                "accuracy": round(
                    counts["correct"] / counts["total"], 4
                ) if counts["total"] else 0.0,
            }
            for key, counts in sorted(groups.items())
        }

    replay = {
        "raw_available": raw_available,
        "raw_to_final_flips": {
            "wrong_to_right": flips[(False, True)],
            "right_to_wrong": flips[(True, False)],
            "right_to_right": flips[(True, True)],
            "wrong_to_wrong": flips[(False, False)],
        },
        "verifier": {
            "accepted_wrong": accepted_wrong,
            "wrong_final_with_any_correct_finish": rejected_correct,
            "fallback_overrode_last_correct_finish": fallback_overrode_correct,
        },
        "by_run_status": summarize_groups(status_counts),
        "by_tool_path": summarize_groups(tool_path_counts),
        "by_operation": summarize_groups(operation_counts),
    }
    if direct_policy_evaluated:
        replay["direct_after_two_verifier_failures"] = {
            "correct": direct_policy_correct,
            "evaluated": direct_policy_evaluated,
            "accuracy": round(
                direct_policy_correct / direct_policy_evaluated, 4),
        }
    return replay


def write_replay_markdown(path: Path, replay: dict) -> None:
    flips = replay["raw_to_final_flips"]
    verifier = replay["verifier"]
    with path.open("w", encoding="utf-8") as output_file:
        output_file.write("# CRT Candidate Replay Diagnostics\n\n")
        output_file.write(f"- Raw candidates available: {replay['raw_available']}\n")
        output_file.write(f"- Raw → final wrong→right: {flips['wrong_to_right']}\n")
        output_file.write(f"- Raw → final right→wrong: {flips['right_to_wrong']}\n")
        output_file.write(
            f"- Verifier accepted wrong finals: {verifier['accepted_wrong']}\n"
        )
        output_file.write(
            "- Wrong finals with a correct Finish candidate: "
            f"{verifier['wrong_final_with_any_correct_finish']}\n"
        )
        output_file.write(
            "- Fallbacks overriding the last correct Finish: "
            f"{verifier['fallback_overrode_last_correct_finish']}\n"
        )
        policy = replay.get("direct_after_two_verifier_failures")
        if policy:
            output_file.write(
                "- Direct after ≥2 verifier failures: "
                f"{policy['correct']}/{policy['evaluated']} "
                f"({100 * policy['accuracy']:.2f}%)\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CRT-QA results by the operation and reasoning types from "
            "Table 3 of the CRT-QA paper."
        )
    )
    parser.add_argument("--result_jsonl", required=True, help="Path to CRT result JSONL.")
    parser.add_argument(
        "--crt_dataset_path",
        default=DEFAULT_CRT_DATASET_PATH,
        help=(
            "CRT-QA directory containing dataset.json, or dataset.json itself. "
            f"Default: {DEFAULT_CRT_DATASET_PATH}"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Output directory. Defaults to the result JSONL parent directory.",
    )
    parser.add_argument(
        "--baseline_result_jsonl",
        default="",
        help=(
            "Optional independent CRT result JSONL used to replay selecting "
            "Direct after at least two verifier failures."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_jsonl = Path(args.result_jsonl).expanduser().resolve()
    dataset_json = resolve_dataset_json(Path(args.crt_dataset_path).expanduser())
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else result_jsonl.parent
    )

    results = load_results(result_jsonl)
    annotations = load_annotations(dataset_json)
    metrics, details = evaluate_result_rows(results, annotations)
    metrics["result_jsonl"] = str(result_jsonl)
    metrics["dataset_json"] = str(dataset_json)

    metrics_path = output_dir / "crt_type_metrics.json"
    markdown_path = output_dir / "crt_type_metrics.md"
    details_path = output_dir / "crt_type_details.jsonl"
    write_json(metrics_path, metrics)
    write_markdown(markdown_path, metrics, result_jsonl, dataset_json)
    write_jsonl(details_path, details)

    baseline_results = None
    if args.baseline_result_jsonl:
        baseline_results = load_results(
            Path(args.baseline_result_jsonl).expanduser().resolve())
    replay = build_replay_metrics(results, annotations, baseline_results)
    replay_path = output_dir / "crt_replay_metrics.json"
    replay_markdown_path = output_dir / "crt_replay_metrics.md"
    write_json(replay_path, replay)
    write_replay_markdown(replay_markdown_path, replay)

    denotation = metrics["metrics"]["denotation_em"]["overall"]
    strict = metrics["metrics"]["normalized_string_em"]["overall"]
    print(f"Evaluated {metrics['evaluated']}/{metrics['total_results']} CRT examples")
    print(f"Denotation EM: {_format_score(denotation)}")
    print(f"Normalized string EM: {_format_score(strict)}")
    print(f"Metrics: {metrics_path}")
    print(f"Markdown: {markdown_path}")
    print(f"Details: {details_path}")
    print(f"Replay metrics: {replay_path}")


if __name__ == "__main__":
    main()
