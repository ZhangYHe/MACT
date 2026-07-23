"""Targeted, structure-based patches for CRT-QA.

The registry deliberately matches question/profile structure only.  It must
never use example IDs, table IDs, or gold answers.
"""

from __future__ import annotations

import re
from copy import deepcopy


_AUXILIARY_YES_NO = re.compile(
    r"^\s*(?:are|is|did|does|do|was|were|has|have|can|could|would)\b",
    flags=re.IGNORECASE,
)


CRT_PATCH_REGISTRY = (
    {
        "rule_id": "count_metric_condition",
        "description": "How-many questions return a count even if their predicate contains a rate.",
        "question_pattern": r"^\s*how many\b.*\b(?:percentage|percent|average|mean|ratio|rate)\b",
        "contract_override": {
            "output_kind": "count",
            "representation": "number",
            "representation_candidates": ["number"],
            "precision": None,
            "precision_policy": "integer_count",
        },
        "reasoning_hint": (
            "The requested denotation is the number of qualifying rows/entities. "
            "Compute the percentage, average, ratio, or rate only as the membership "
            "predicate; Finish with the integer count and no percent sign or unit."
        ),
        "calculation_guard": "count_after_metric_predicate",
        "final_rendering_policy": "integer_count",
    },
    {
        "rule_id": "entity_ranked_by_metric",
        "description": "Which/who arg-extreme questions return the entity, not the metric.",
        "question_pattern": (
            r"^\s*(?:which|who)\b.*\b(?:highest|lowest|largest|smallest|most|least|"
            r"fastest|greatest|best|worst)\b.*\b(?:average|mean|percentage|percent|"
            r"ratio|rate|difference|total|number|score|attendance|crowd)\b"
        ),
        "contract_override": {
            "output_kind": "entity",
            "representation": "text",
            "representation_candidates": ["text"],
            "precision": None,
            "precision_policy": "none",
        },
        "reasoning_hint": (
            "Use the metric only to rank/group rows. Finish with the winning entity "
            "name from the requested target column, not the numeric metric."
        ),
        "calculation_guard": "arg_extreme_return_entity_and_metric",
        "final_rendering_policy": "entity_only",
    },
    {
        "rule_id": "direction_change_label",
        "description": "Increase/decrease questions return the requested direction.",
        "question_pattern": r"\b(?:increase|increased)\b\s+or\s+\b(?:decrease|decreased)\b|\b(?:decrease|decreased)\b\s+or\s+\b(?:increase|increased)\b",
        "contract_override": {
            "output_kind": "relation_label",
            "representation": "text",
            "representation_candidates": ["text"],
            "precision": None,
            "precision_policy": "none",
        },
        "reasoning_hint": (
            "Compute any supporting difference or trend, but Finish with only the "
            "direction word requested by the question, not the numeric difference."
        ),
        "calculation_guard": "trend_supports_direction",
        "final_rendering_policy": "direction_label",
    },
    {
        "rule_id": "explicit_from_to_range",
        "description": "Questions explicitly requesting From-to preserve that wrapper.",
        "question_pattern": r"\(\s*from(?:\s+min)?\s+to(?:\s+max)?\s*\)|\bfrom\s+min\s+to\s+max\b",
        "contract_override": {
            "output_kind": "range",
            "representation": "text",
            "representation_candidates": ["from_to_range"],
            "precision_policy": "preserve_values",
        },
        "reasoning_hint": (
            "After computing the minimum and maximum, Finish exactly in the requested "
            "'From <min> to <max>' form and preserve any requested units."
        ),
        "calculation_guard": "range_min_max",
        "final_rendering_policy": "from_to_range",
    },
    {
        "rule_id": "party_incumbent_proportion_fraction",
        "description": "CRT incumbent party proportions use numerator/denominator form.",
        "question_pattern": r"\bproportion of democratic to republican incumbents\b",
        "contract_override": {
            "output_kind": "ratio",
            "representation": "fraction",
            "representation_candidates": ["fraction"],
            "precision": None,
            "precision_policy": "preserve_pair",
        },
        "reasoning_hint": (
            "Count Democratic and Republican incumbents separately and Finish with the "
            "unsimplified Democratic/Republican fraction using '/'."
        ),
        "calculation_guard": "preserve_unsimplified_numerator_denominator",
        "final_rendering_policy": "fraction_pair",
    },
    {
        "rule_id": "aggregate_ratio_preserve_pair",
        "description": "Selected aggregate-ratio templates preserve the raw group totals.",
        "question_pattern": (
            r"\bratio of medals won between the top\b.*\bbottom\b|"
            r"\bratio of silver to gold medals among the nations that earned a gold medal\b"
        ),
        "contract_override": {
            "output_kind": "ratio",
            "representation": "colon_ratio",
            "representation_candidates": ["colon_ratio"],
            "precision": None,
            "precision_policy": "preserve_pair",
        },
        "reasoning_hint": (
            "Return the two aggregate totals as an unsimplified numerator:denominator "
            "pair. Do not reduce the ratio."
        ),
        "calculation_guard": "preserve_unsimplified_numerator_denominator",
        "final_rendering_policy": "colon_pair",
    },
    {
        "rule_id": "table_score_spacing",
        "description": "Score-like answers preserve the exact table cell spelling.",
        "question_pattern": r"\b(?:most common score|score for matches|aggregate score)\b",
        "contract_override": {
            "output_kind": "table_value",
            "precision": None,
            "precision_policy": "preserve_table_cell",
        },
        "reasoning_hint": (
            "If the answer is a score or record copied from the table, reproduce the "
            "winning table cell exactly, including spaces around its hyphen."
        ),
        "calculation_guard": "mode_preserve_source_value",
        "final_rendering_policy": "preserve_table_cell",
    },
    {
        "rule_id": "rowwise_release_intervals",
        "description": "Release-interval averages stay row-wise unless uniqueness is requested.",
        "question_pattern": r"\baverage number of years between\b.*\breleases?\b",
        "contract_override": {
            "output_kind": "number",
            "representation": "number",
            "representation_candidates": ["number"],
            "precision_policy": "question_template",
        },
        "reasoning_hint": (
            "Use the table rows in chronological order, including repeated rows, unless "
            "the question explicitly says unique or distinct. Compute consecutive gaps "
            "over that row sequence."
        ),
        "calculation_guard": "rowwise_no_implicit_dedup",
        "final_rendering_policy": "number",
    },
)


def _normalize_question(question: object) -> str:
    return re.sub(r"\s+", " ", str(question or "")).strip()


def match_crt_patches(question: object, profile: dict | None = None) -> list[dict]:
    """Return registry rules matching question/profile structure."""
    text = _normalize_question(question)
    matched = []
    for rule in CRT_PATCH_REGISTRY:
        if re.search(rule["question_pattern"], text, flags=re.IGNORECASE):
            matched.append(deepcopy(rule))

    # General binary-label inference is intentionally last so a more specific
    # direction/closed-set patch can override it.
    profile = profile or {}
    existing_contract = profile.get("answer_contract") or {}
    has_specific_relation = any(
        rule["rule_id"] == "direction_change_label" for rule in matched
    )
    if (
        _AUXILIARY_YES_NO.search(text)
        and not has_specific_relation
        and not existing_contract.get("allowed_labels")
        and not re.search(
            r"\bor\b|\bif so\b|\b(?:which|who|what|where|when|how many)\b",
            text,
            flags=re.IGNORECASE,
        )
        and not re.search(
            r"answer\s+with\s+only\s+['\"][^'\"]+['\"]\s+or\s+['\"]",
            text,
            flags=re.IGNORECASE,
        )
    ):
        matched.append({
            "rule_id": "implicit_binary_question",
            "description": "Auxiliary-led CRT questions return Yes or No.",
            "question_pattern": _AUXILIARY_YES_NO.pattern,
            "contract_override": {
                "output_kind": "yes_no",
                "label_type": "yes_no",
                "allowed_labels": ["Yes", "No"],
                "representation": "text",
                "representation_candidates": ["text"],
                "precision": None,
                "precision_policy": "none",
            },
            "reasoning_hint": (
                "This is a binary question. Use calculations only as evidence and "
                "Finish with exactly Yes or No."
            ),
            "calculation_guard": "binary_conclusion",
            "final_rendering_policy": "allowed_label",
        })
    return matched


def apply_contract_overrides(contract: dict, patches: list[dict]) -> dict:
    """Apply matched contract overrides in registry order."""
    updated = deepcopy(contract)
    for patch in patches:
        updated.update(deepcopy(patch.get("contract_override") or {}))
    return updated


def patch_ids(patches: list[dict]) -> list[str]:
    return [str(patch["rule_id"]) for patch in patches]


def patch_prompt_hints(patches: list[dict]) -> str:
    hints = [
        f"- [{patch['rule_id']}] {patch['reasoning_hint']}"
        for patch in patches
        if patch.get("reasoning_hint")
    ]
    if not hints:
        return ""
    return "CRT targeted patch hints (override conflicting generic hints):\n" + "\n".join(hints)
