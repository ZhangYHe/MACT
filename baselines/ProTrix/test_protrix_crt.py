#!/usr/bin/env python3
"""No-weight tests for ProTrix prompt, SQL, extraction, and two-stage flow."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


SQL = load_module("protrix_sql", "protrix_sql.py")
RUN = load_module("run_protrix_crt", "run_protrix_crt.py")


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


class FakeEngine:
    def __init__(self):
        self.calls = 0

    def generate(self, prompts, _params, use_tqdm=False):
        self.calls += 1
        if self.calls == 1:
            texts = [
                "Plan.\n```sql\nSELECT value FROM table WHERE name = 'x'\n```",
                "The answer is Yes",
            ]
        else:
            texts = ["Reasoning from the execution.\nThe answer is 10"]
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text)]) for text in texts]


class ProTrixTests(unittest.TestCase):
    def setUp(self):
        self.examples = [
            {
                "id": "a",
                "statement": "What is x's value?",
                "table_text": [["name", "value"], ["x", "10"]],
                "answer": ["10"],
            },
            {
                "id": "b",
                "statement": "Is x listed?",
                "table_text": [["name"], ["x"]],
                "answer": ["Yes"],
            },
        ]

    def test_prompt_has_no_gold_answer(self):
        prompt = RUN.build_base_prompt(self.examples[0], self.examples[0]["table_text"])
        self.assertIn("What is x's value?", prompt)
        self.assertNotIn("['10']", prompt)
        self.assertTrue(prompt.endswith("## Answer:\n"))

    def test_answer_extraction(self):
        self.assertEqual(RUN.extract_short_answer("The answer is Yes"), ("Yes", "answer_marker"))
        self.assertEqual(RUN.extract_short_answer("work\nFinal Answer: 1-7."), ("1-7", "answer_marker"))
        self.assertEqual(RUN.extract_short_answer("work\nParis"), ("Paris", "last_nonempty_line"))

    def test_safe_sql_and_blocking(self):
        queries = SQL.extract_sql_queries("```sql\nSELECT value FROM table\n```")
        execution = SQL.execute_queries(self.examples[0]["table_text"], queries)[0]
        self.assertEqual(execution["status"], "ok")
        self.assertEqual(str(execution["rows"][0]["value"]), "10")
        blocked = SQL.execute_queries(
            self.examples[0]["table_text"], ["DROP TABLE w"]
        )[0]
        self.assertEqual(blocked["status"], "error")
        literal_table = [["x", "value"], ["x", "7"]]
        literal = SQL.execute_queries(
            literal_table, ["SELECT value FROM table WHERE x = 'x'"]
        )[0]
        self.assertEqual(literal["status"], "ok")
        self.assertEqual(str(literal["rows"][0]["value"]), "7")
        cte = SQL.execute_queries(
            self.examples[0]["table_text"],
            ["WITH t AS (SELECT value FROM table) SELECT value FROM t"],
        )[0]
        self.assertEqual(cte["status"], "ok")

    def test_two_stage_mock_flow(self):
        results = RUN.run_batch(
            self.examples,
            FakeTokenizer(),
            FakeEngine(),
            object(),
            RUN.DEFAULT_MODEL_PATH,
            3072,
            1024,
        )
        self.assertEqual([item["pred_answer"] for item in results], ["10", "Yes"])
        self.assertTrue(results[0]["protrix_metadata"]["used_second_pass"])
        self.assertFalse(results[1]["protrix_metadata"]["used_second_pass"])

    def test_resume_prefix(self):
        RUN.validate_resume_prefix(
            self.examples, [{"id": "a", "run_status": "completed"}]
        )
        with self.assertRaisesRegex(ValueError, "exact input prefix"):
            RUN.validate_resume_prefix(
                self.examples, [{"id": "b", "run_status": "completed"}]
            )


if __name__ == "__main__":
    unittest.main()
