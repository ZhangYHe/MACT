#!/usr/bin/env python3
"""No-weight tests for the CRT-QA TAPAS baseline."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer


MODULE_PATH = Path(__file__).with_name("run_tapas_crt.py")
SPEC = importlib.util.spec_from_file_location("run_tapas_crt", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import {MODULE_PATH}")
TAPAS_RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TAPAS_RUN)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"transformers\.models\.tapas\.tokenization_tapas",
)


class FakeTapasModel:
    """Select row 0, column 0 with NONE aggregation for every table."""

    config = SimpleNamespace(aggregation_labels=TAPAS_RUN.EXPECTED_AGGREGATIONS)

    def __call__(self, **model_inputs: torch.Tensor) -> SimpleNamespace:
        token_types = model_inputs["token_type_ids"]
        row_ids = token_types[:, :, 2]
        column_ids = token_types[:, :, 1]
        cell_mask = (row_ids == 1) & (column_ids == 1)
        logits = torch.full(row_ids.shape, -100.0, device=row_ids.device)
        logits[cell_mask] = 100.0
        aggregation_logits = torch.full(
            (row_ids.shape[0], 4), -100.0, device=row_ids.device
        )
        aggregation_logits[:, 0] = 100.0
        return SimpleNamespace(
            logits=logits,
            logits_aggregation=aggregation_logits,
        )


class TapasPostprocessTests(unittest.TestCase):
    def test_none_single_and_multiple(self) -> None:
        self.assertEqual(
            TAPAS_RUN.postprocess_prediction("NONE", ["alpha"]),
            ("alpha", "ok", []),
        )
        self.assertEqual(
            TAPAS_RUN.postprocess_prediction("NONE", ["alpha", "beta"]),
            (["alpha", "beta"], "ok", []),
        )

    def test_count_sum_and_average(self) -> None:
        self.assertEqual(
            TAPAS_RUN.postprocess_prediction("COUNT", ["a", "b"]),
            ("2", "ok", []),
        )
        self.assertEqual(
            TAPAS_RUN.postprocess_prediction("SUM", ["1,000", "$2"]),
            ("1002", "ok", [1000.0, 2.0]),
        )
        self.assertEqual(
            TAPAS_RUN.postprocess_prediction("AVERAGE", ["50%", "100%"]),
            ("75", "ok", [50.0, 100.0]),
        )

    def test_tapas_numeric_parsing_and_stable_formatting(self) -> None:
        self.assertEqual(TAPAS_RUN.extract_numeric_value("1,234.5"), 1234.5)
        self.assertEqual(TAPAS_RUN.extract_numeric_value("$1,250"), 1250.0)
        self.assertEqual(TAPAS_RUN.extract_numeric_value("50%"), 50.0)
        self.assertEqual(TAPAS_RUN.format_number(4.0), "4")
        self.assertEqual(TAPAS_RUN.format_number(1.0 / 3.0), "0.333333333333333")

    def test_empty_and_unparseable_aggregation(self) -> None:
        self.assertEqual(
            TAPAS_RUN.postprocess_prediction("SUM", []),
            ("", "empty_selection", []),
        )
        self.assertEqual(
            TAPAS_RUN.postprocess_prediction("SUM", ["n/a"]),
            ("", "non_numeric_aggregation", [None]),
        )


class TapasBatchAndResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"transformers\.models\.tapas\.tokenization_tapas",
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(
            TAPAS_RUN.DEFAULT_MODEL_PATH,
            local_files_only=True,
            clean_up_tokenization_spaces=True,
        )
        cls.examples = [
            {
                "id": "first",
                "statement": "What is the first value?",
                "table_text": [["value", "other"], ["10", "x"]],
                "answer": ["10"],
            },
            {
                "id": "second",
                "statement": "What is the first city?",
                "table_text": [["city"], ["Paris"], ["Rome"]],
                "answer": ["Paris"],
            },
        ]

    def test_different_tables_are_padded_and_coordinates_are_converted(self) -> None:
        results = TAPAS_RUN.run_batch(
            batch=self.examples,
            tokenizer=self.tokenizer,
            model=FakeTapasModel(),
            device=torch.device("cpu"),
            model_path=TAPAS_RUN.DEFAULT_MODEL_PATH,
            max_source_length=1024,
            cell_classification_threshold=0.5,
        )
        self.assertEqual([item["pred_answer"] for item in results], ["10", "Paris"])
        self.assertEqual(
            [item["tapas_metadata"]["selected_coordinates"] for item in results],
            [[[0, 0]], [[0, 0]]],
        )
        self.assertTrue(all(item["run_status"] == "completed" for item in results))
        self.assertTrue(all("answer" in item for item in results))

    def test_resume_requires_an_exact_completed_prefix(self) -> None:
        completed = [
            {
                **self.examples[0],
                "pred_answer": "10",
                "run_status": "completed",
            }
        ]
        TAPAS_RUN.validate_resume_prefix(self.examples, completed)
        with self.assertRaisesRegex(ValueError, "exact prefix"):
            TAPAS_RUN.validate_resume_prefix(
                self.examples,
                [{**completed[0], "id": "second"}],
            )
        with self.assertRaisesRegex(ValueError, "non-completed"):
            TAPAS_RUN.validate_resume_prefix(
                self.examples,
                [{**completed[0], "run_status": "failed"}],
            )

    def test_jsonl_schema_order_and_final_validation(self) -> None:
        rows = [
            {
                **example,
                "pred_answer": example["answer"][0],
                "pred_answer_all": [example["answer"][0]],
                "run_status": "completed",
                "tapas_metadata": {},
            }
            for example in self.examples
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "results.jsonl"
            with output_path.open("w", encoding="utf-8") as output_file:
                for row in rows:
                    output_file.write(json.dumps(row) + "\n")
            TAPAS_RUN.validate_final_output(output_path, self.examples)


if __name__ == "__main__":
    unittest.main()
