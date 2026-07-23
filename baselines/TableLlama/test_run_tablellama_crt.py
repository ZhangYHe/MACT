from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import run_tablellama_crt as baseline


def example(example_id: str = "crt:answerable:table:0") -> dict:
    return {
        "id": example_id,
        "statement": "Which value is listed?",
        "table_text": [["name", "value"], ["alpha", "one"], ["beta", "two"]],
        "answer": ["SECRET_GOLD"],
        "dataset": "crt",
    }


class WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        extra = 1 if add_special_tokens else 0
        return list(range(len(text.split()) + extra))


class FakeEngine:
    def __init__(self, responses: list[str] | None = None, error: Exception | None = None):
        self.responses = responses or []
        self.error = error

    def generate(self, prompts, sampling_params, use_tqdm=False):
        if self.error:
            raise self.error
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text)]) for text in self.responses]


class TableLlamaBaselineTests(unittest.TestCase):
    def test_official_prompt_does_not_include_gold(self) -> None:
        prompt = baseline.build_prompt(example())
        self.assertIn("### Instruction:\nThis is a table QA task.", prompt)
        self.assertIn("[TAB] col: | name | value | [SEP]", prompt)
        self.assertIn("| alpha | one | [SEP]", prompt)
        self.assertIn("### Question:\nWhich value is listed?", prompt)
        self.assertTrue(prompt.endswith("### Response:\n"))
        self.assertNotIn("SECRET_GOLD", prompt)

    def test_serialization_normalizes_newlines_and_preserves_unicode(self) -> None:
        table = [["a|b", "城市"], ["line1\nline2", "München"]]
        serialized = baseline.serialize_table(table)
        self.assertIn("a|b", serialized)
        self.assertIn("城市", serialized)
        self.assertIn("line1 line2", serialized)
        self.assertIn("München", serialized)

    def test_row_truncation_keeps_largest_fitting_prefix(self) -> None:
        item = example()
        tokenizer = WordTokenizer()
        zero_row_length = baseline.token_count(tokenizer, baseline.build_prompt(item, 0))
        one_row_length = baseline.token_count(tokenizer, baseline.build_prompt(item, 1))
        prompt, length, kept, removed = baseline.truncate_prompt_to_fit(
            item, tokenizer, one_row_length
        )
        self.assertEqual(length, one_row_length)
        self.assertEqual((kept, removed), (1, 1))
        self.assertIn("alpha", prompt)
        self.assertNotIn("beta", prompt)
        self.assertLess(zero_row_length, one_row_length)

    def test_run_batch_produces_mact_schema(self) -> None:
        item = example()
        output = baseline.run_batch(
            [item],
            WordTokenizer(),
            FakeEngine(["  one  "]),
            object(),
            Path("/models/TableLlama"),
            4096,
            3968,
            128,
            42,
            "torch.bfloat16",
        )[0]
        self.assertEqual(output["pred_answer"], "one")
        self.assertEqual(output["pred_answer_all"], ["one"])
        self.assertEqual(output["run_status"], "completed")
        self.assertEqual(output["answer"], ["SECRET_GOLD"])
        self.assertEqual(output["tablellama_metadata"]["raw_response"], "  one  ")
        self.assertEqual(output["tablellama_metadata"]["truncated_data_rows"], 0)

    def test_run_batch_reports_actionable_oom(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Lower BATCH_SIZE"):
            baseline.run_batch(
                [example()],
                WordTokenizer(),
                FakeEngine(error=RuntimeError("CUDA out of memory")),
                object(),
                Path("/models/TableLlama"),
                4096,
                3968,
                128,
                42,
                "torch.bfloat16",
            )

    def test_resume_requires_exact_completed_prefix(self) -> None:
        examples = [example("a"), example("b")]
        baseline.validate_resume_prefix(examples, [{"id": "a", "run_status": "completed"}])
        with self.assertRaisesRegex(ValueError, "exact input prefix"):
            baseline.validate_resume_prefix(
                examples, [{"id": "b", "run_status": "completed"}]
            )
        with self.assertRaisesRegex(ValueError, "non-completed"):
            baseline.validate_resume_prefix(examples, [{"id": "a", "run_status": "failed"}])

    def test_existing_results_reject_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "results.jsonl"
            row = {"id": "duplicate", "run_status": "completed"}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate existing output id"):
                baseline.load_existing_results(path)

    def test_model_validation_reads_indexed_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory)
            (model_path / "config.json").write_text(
                json.dumps({"model_type": "llama", "torch_dtype": "bfloat16"}),
                encoding="utf-8",
            )
            for name in (
                "special_tokens_map.json",
                "tokenizer.model",
                "tokenizer_config.json",
            ):
                (model_path / name).write_text("x", encoding="utf-8")
            (model_path / "shard-1.bin").write_bytes(b"weights")
            (model_path / "pytorch_model.bin.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 7},
                        "weight_map": {"layer.weight": "shard-1.bin"},
                    }
                ),
                encoding="utf-8",
            )
            metadata = baseline.validate_model_directory(model_path)
            self.assertEqual(metadata["weights_size_bytes"], 7)
            self.assertEqual(metadata["config_torch_dtype"], "bfloat16")


if __name__ == "__main__":
    unittest.main()
