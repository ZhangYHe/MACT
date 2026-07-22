#!/usr/bin/env python3
"""No-weight tests for the OmniTab CRT baseline."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoTokenizer, logging


logging.set_verbosity_error()


MODULE_PATH = Path(__file__).with_name("run_omnitab_crt.py")
SPEC = importlib.util.spec_from_file_location("run_omnitab_crt", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class OmniTabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(
            MODULE.DEFAULT_MODEL_PATH,
            local_files_only=True,
            clean_up_tokenization_spaces=True,
        )
        cls.examples = [
            {
                "id": "a",
                "statement": "What is the value?",
                "table_text": [["name", "value"], ["x", "10"]],
                "answer": ["10"],
            },
            {
                "id": "b",
                "statement": "Which city is listed?",
                "table_text": [["city"], ["Paris"], ["Rome"]],
                "answer": ["Paris"],
            },
        ]

    def test_different_tables_pad_without_gold(self) -> None:
        class SpyTokenizer:
            def __init__(self, inner):
                self.inner = inner
                self.answers = []

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def prepare_table_query(self, *args, **kwargs):
                self.answers.append(kwargs.get("answer"))
                return self.inner.prepare_table_query(*args, **kwargs)

        spy = SpyTokenizer(self.tokenizer)
        encoded, lengths = MODULE.encode_batch(self.examples, spy, 1024, 42)
        self.assertEqual(encoded["input_ids"].shape[0], 2)
        self.assertEqual(len(lengths), 2)
        self.assertEqual(spy.answers, [None, None])

    def test_resume_prefix(self) -> None:
        MODULE.validate_resume_prefix(
            self.examples,
            [{"id": "a", "run_status": "completed", "pred_answer": "10"}],
        )
        with self.assertRaisesRegex(ValueError, "exact input prefix"):
            MODULE.validate_resume_prefix(
                self.examples,
                [{"id": "b", "run_status": "completed", "pred_answer": "Paris"}],
            )

    def test_mock_generation_writes_mact_fields(self) -> None:
        class DecodeTokenizer:
            def __init__(self, inner):
                self.inner = inner

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def batch_decode(self, _generated, skip_special_tokens=True):
                return ["10", "Paris"]

        class FakeModel:
            def generate(self, **_kwargs):
                return torch.zeros((2, 2), dtype=torch.long)

        results = MODULE.run_batch(
            self.examples,
            DecodeTokenizer(self.tokenizer),
            FakeModel(),
            torch.device("cpu"),
            MODULE.DEFAULT_MODEL_PATH,
            1024,
            64,
            42,
        )
        self.assertEqual([item["pred_answer"] for item in results], ["10", "Paris"])
        self.assertTrue(all(item["run_status"] == "completed" for item in results))
        self.assertTrue(all("answer" in item for item in results))

    def test_schema_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"id":"x","statement":"q","table_text":[["a"],["1","2"]]}\n')
            with self.assertRaisesRegex(ValueError, "does not match"):
                MODULE.load_examples(path, None)


if __name__ == "__main__":
    unittest.main()
