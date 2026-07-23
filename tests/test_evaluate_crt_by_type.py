import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from evaluate_crt_by_type import (  # noqa: E402
    COMPOSITION_TYPES,
    DIRECTNESS_TYPES,
    OPERATION_TYPES,
    REASONING_TYPES,
    build_replay_metrics,
    evaluate_result_rows,
    load_annotations,
    normalized_string_match,
    write_json,
    write_jsonl,
    write_markdown,
)


def step(step_type, name):
    return {"type": step_type, "name": name, "detail": "test"}


def question(
    *steps,
    answer="answer",
    directness="Explicit",
    composition_type="Other",
):
    row = {
        "Question name": "test question",
        "Tittle": "test table",
        "Answer": answer,
        "Directness": directness,
        "Composition Type": composition_type,
    }
    for index in range(1, 5):
        row[f"step{index}"] = steps[index - 1] if index <= len(steps) else {}
    return row


class NormalizedStringEmTest(unittest.TestCase):
    def test_normalizes_only_unicode_case_and_whitespace(self):
        self.assertTrue(normalized_string_match([" Ｙes\nAnswer "], ["ｙES answer"]))
        self.assertFalse(normalized_string_match(["6"], ["6.0"]))
        self.assertFalse(normalized_string_match(["Netherlands"], ["Netherlands (NED)"]))
        self.assertFalse(normalized_string_match(["1-7"], ["1–7"]))
        self.assertFalse(normalized_string_match(["a", "b"], ["b", "a"]))


class ReplayDiagnosticsTest(unittest.TestCase):
    def test_reports_verifier_and_direct_fallback_counterfactuals(self):
        example_id = "crt:answerable:table.csv:0"
        annotations = {
            example_id: type("AnnotationStub", (), {
                "operations": frozenset({"Group"}),
            })()
        }
        results = [{
            "id": example_id,
            "answer": ["more"],
            "pred_answer": "larger",
            "run_status": "fallback_answered",
            "history": (
                "Action 1: Finish[more]\n"
                "Observation 1: Final verification failed.\n"
                "Action 2: Finish[more]\n"
                "Observation 2: Final verification failed.\n"
            ),
            "tool_events": [{"tool": "Calculate", "status": "success"}],
        }]
        baseline = [{
            "id": example_id,
            "answer": ["more"],
            "pred_answer": "more",
        }]

        replay = build_replay_metrics(results, annotations, baseline)

        self.assertEqual(replay["raw_to_final_flips"]["right_to_wrong"], 1)
        self.assertEqual(
            replay["verifier"]["fallback_overrode_last_correct_finish"], 1)
        self.assertEqual(
            replay["direct_after_two_verifier_failures"]["correct"], 1)


class AnnotationLoadingTest(unittest.TestCase):
    def write_dataset(self, root, data):
        dataset_json = Path(root) / "dataset.json"
        dataset_json.write_text(json.dumps(data), encoding="utf-8")
        return dataset_json

    def test_maps_all_paper_types_and_deduplicates_repeated_steps(self):
        operation_names = ["Indexing", "Sorting", "Grouping", "Filter"]
        reasoning_names = [
            "Grounding",
            "Auto-categorization",
            "Temporal Reasoning",
            "Aggregating",
            "Arithmetic",
            "Geographical/Spatial Reasoning",
            "Reasoning with Quantifiers",
            "Other Commonsense Reasoning",
        ]
        rows = [question(step("Operation", name)) for name in operation_names]
        rows += [question(step("Reasoning", name)) for name in reasoning_names]
        rows.append(
            question(
                step("Operation", "Indexing"),
                step("Operation", "Indexing"),
                step("Reasoning", "Arithmetic"),
            )
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            annotations = load_annotations(
                self.write_dataset(tmp_dir, {"table.csv": rows})
            )

        observed_operations = {
            next(iter(annotation.operations))
            for annotation in list(annotations.values())[:4]
        }
        observed_reasoning = {
            next(iter(annotation.reasoning))
            for annotation in list(annotations.values())[4:12]
        }
        repeated = annotations["crt:answerable:table.csv:12"]
        self.assertEqual(observed_operations, set(OPERATION_TYPES))
        self.assertEqual(observed_reasoning, set(REASONING_TYPES))
        self.assertEqual(repeated.operations, frozenset({"Index"}))
        self.assertEqual(repeated.reasoning, frozenset({"ARI"}))
        self.assertEqual(repeated.directness, "Explicit")
        self.assertEqual(repeated.composition_type, "Other")

    def test_keeps_unknown_and_missing_steps_outside_paper_categories(self):
        data = {
            "table.csv": [
                question(
                    step("Reasoning", "Analytic entailment"),
                    {"detail": "missing type and name"},
                    step("Operation", "Unlisted operation"),
                    step("Other", "Unlisted step type"),
                )
            ]
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            annotations = load_annotations(
                self.write_dataset(tmp_dir, data)
            )

        annotation = annotations["crt:answerable:table.csv:0"]
        self.assertFalse(annotation.operations)
        self.assertFalse(annotation.reasoning)
        self.assertEqual(annotation.unknown_reasoning, ("Analytic entailment",))
        self.assertEqual(annotation.unknown_operations, ("Unlisted operation",))
        self.assertEqual(annotation.unknown_step_types, ("Other",))
        self.assertEqual(annotation.missing_steps, ("step2",))


class EvaluationTest(unittest.TestCase):
    def make_annotations(self, root):
        data = {
            "table.csv": [
                question(
                    step("Operation", "Indexing"),
                    step("Operation", "Indexing"),
                    step("Reasoning", "Arithmetic"),
                    step("Reasoning", "Analytic entailment"),
                    composition_type="Bridging",
                    answer="6",
                ),
                question(
                    step("Operation", "Grouping"),
                    step("Operation", "Filter"),
                    step("Reasoning", "Reasoning with Quantifiers"),
                    directness="Implicit",
                    composition_type="Intersection",
                    answer="Yes",
                ),
                question(
                    step("Operation", "Sorting"),
                    step("Reasoning", "Grounding"),
                    directness=None,
                    composition_type=None,
                    answer="",
                ),
                question(
                    step("Reasoning", "Other Commonsense Reasoning"),
                    composition_type="Comparison",
                    answer="x",
                ),
                question(
                    step("Reasoning", "Grounding"),
                    composition_type="Other",
                    answer="other",
                ),
            ]
        }
        dataset_json = Path(root) / "dataset.json"
        dataset_json.write_text(json.dumps(data), encoding="utf-8")
        return dataset_json, load_annotations(dataset_json)

    def test_scores_overlapping_groups_and_reports_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_json, annotations = self.make_annotations(tmp_dir)
            results = [
                {
                    "id": "crt:answerable:table.csv:0",
                    "answer": ["6"],
                    "pred_answer": "6.0",
                },
                {
                    "id": "crt:answerable:table.csv:1",
                    "answer": ["Yes"],
                    "pred_answer": " yes ",
                },
                {
                    "id": "crt:answerable:table.csv:2",
                    "answer": [],
                    "pred_answer": "No",
                },
                {
                    "id": "crt:answerable:table.csv:3",
                    "answer": ["x"],
                    "pred_answer": "",
                },
                {
                    "id": "crt:answerable:table.csv:4",
                    "answer": ["other"],
                    "pred_answer": "other",
                },
                {
                    "id": "crt:answerable:missing.csv:0",
                    "answer": ["known"],
                    "pred_answer": "known",
                },
            ]
            metrics, details = evaluate_result_rows(results, annotations)

            denotation = metrics["metrics"]["denotation_em"]
            strict = metrics["metrics"]["normalized_string_em"]
            self.assertEqual(denotation["overall"]["correct"], 4)
            self.assertEqual(denotation["overall"]["evaluated"], 5)
            self.assertEqual(strict["overall"]["correct"], 3)
            self.assertEqual(strict["overall"]["evaluated"], 5)
            self.assertEqual(denotation["by_operation"]["Index"]["total_members"], 1)
            self.assertEqual(denotation["by_operation"]["Index"]["correct"], 1)
            self.assertEqual(strict["by_operation"]["Index"]["correct"], 0)
            self.assertEqual(denotation["by_operation"]["Group"]["correct"], 1)
            self.assertEqual(denotation["by_operation"]["Filter"]["correct"], 1)
            self.assertEqual(denotation["by_operation"]["Sort"]["evaluated"], 0)
            self.assertEqual(
                denotation["by_operation"]["Sort"]["invalid_gold_count"], 1
            )
            self.assertEqual(set(denotation["by_directness"]), set(DIRECTNESS_TYPES))
            self.assertEqual(set(denotation["by_composition"]), set(COMPOSITION_TYPES))
            self.assertEqual(denotation["by_directness"]["Explicit"]["correct"], 2)
            self.assertEqual(denotation["by_directness"]["Implicit"]["correct"], 1)
            self.assertEqual(denotation["by_composition"]["Bridging"]["correct"], 1)
            self.assertEqual(
                denotation["by_composition"]["Intersection"]["correct"], 1
            )
            self.assertEqual(denotation["by_composition"]["Comparison"]["correct"], 0)

            diagnostics = metrics["diagnostics"]
            self.assertEqual(diagnostics["invalid_gold_count"], 1)
            self.assertEqual(diagnostics["empty_prediction_count"], 1)
            self.assertEqual(diagnostics["missing_annotation_count"], 1)
            self.assertEqual(diagnostics["missing_directness_count"], 1)
            self.assertEqual(diagnostics["missing_composition_count"], 1)
            self.assertEqual(
                diagnostics["excluded_composition_types"]["Other"]["question_count"],
                1,
            )
            self.assertEqual(
                diagnostics["unknown_reasoning_types"]["Analytic entailment"],
                {"question_count": 1, "step_count": 1},
            )
            self.assertEqual(details[0]["operation_types"], ["Index"])

            output_dir = Path(tmp_dir) / "output"
            metrics_path = output_dir / "crt_type_metrics.json"
            details_path = output_dir / "crt_type_details.jsonl"
            markdown_path = output_dir / "crt_type_metrics.md"
            write_json(metrics_path, metrics)
            write_jsonl(details_path, details)
            write_markdown(
                markdown_path,
                metrics,
                Path(tmp_dir) / "results.jsonl",
                dataset_json,
            )
            self.assertTrue(metrics_path.is_file())
            self.assertEqual(len(details_path.read_text().splitlines()), 6)
            markdown = markdown_path.read_text()
            self.assertIn(
                "| Metric | Overall | Index | Sort | Group | Filter | GRO | CAT | "
                "TEM | AGG | ARI | SPA | QUA | OTH | Explicit | Implicit | "
                "Bridging | Intersection | Comparison |",
                markdown,
            )
            self.assertIn("| Denotation EM |", markdown)
            self.assertIn("| 80.00 |", markdown)
            self.assertIn("- Overall 5/6", markdown)
            self.assertIn("- Explicit 3/3", markdown)
            self.assertIn("- Implicit 1/1", markdown)
            self.assertIn("- Bridging 1/1", markdown)
            self.assertIn("- Intersection 1/1", markdown)
            self.assertIn("- Comparison 1/1", markdown)
            self.assertIn("- Excluded Composition types:", markdown)
            self.assertNotIn("<table>", markdown)
            self.assertNotIn("Operation Types", markdown)
            self.assertIn("Analytic entailment", markdown)

    def test_rejects_duplicate_result_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, annotations = self.make_annotations(tmp_dir)
            row = {
                "id": "crt:answerable:table.csv:0",
                "answer": ["6"],
                "pred_answer": "6",
            }
            with self.assertRaisesRegex(ValueError, "Duplicate result ID"):
                evaluate_result_rows([row, dict(row)], annotations)


if __name__ == "__main__":
    unittest.main()
