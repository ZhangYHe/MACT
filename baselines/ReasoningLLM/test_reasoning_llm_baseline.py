#!/usr/bin/env python3
"""Mock-only tests for the low-call GPT reasoning baselines."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import reasoning_llm_baseline as baseline  # noqa: E402


def chat_result(contents: list[str]) -> dict:
    return {
        "http_status": 200,
        "model": "gpt-5-test",
        "choices": [
            {
                "index": index,
                "content": content,
                "finish_reason": "stop",
            }
            for index, content in enumerate(contents)
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": len(contents) * 4,
            "total_tokens": 10 + len(contents) * 4,
        },
    }


def runtime_config(mode: str = "n") -> dict:
    return {
        "api_key": "not-a-real-key",
        "base_url": "https://example.invalid/v1",
        "model_name": "gpt-5-test",
        "multi_choice_mode": mode,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 2000,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "request_timeout": 1.0,
        "max_retries": 0,
        "request_limiter": None,
    }


def dataset_item() -> dict:
    return {
        "id": "crt:test:1",
        "statement": "How many rows are present?",
        "table_text": [["name"], ["a"], ["b"]],
        "answer": ["2"],
        "task": "crt",
    }


class FakeHTTPResponse:
    def __init__(self, payload: dict, status: int = 200):
        self.payload = json.dumps(payload).encode("utf-8")
        self.status = status

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self.payload


class PromptAndParsingTest(unittest.TestCase):
    def test_prompt_is_short_and_contains_no_gold(self) -> None:
        prompt = baseline.build_cot_prompt("Count the rows.", "name\na\nb")
        self.assertEqual(
            prompt,
            "TABLE\nname\na\nb\n\nQ: Count the rows.\n"
            "Reason in ≤3 short steps. End with `FINAL_ANSWER: <answer>`.",
        )
        self.assertNotIn("gold", prompt.casefold())
        self.assertNotIn("2", prompt)

    def test_extracts_last_explicit_marker(self) -> None:
        answer, status = baseline.extract_final_answer(
            "1. Count.\nFINAL_ANSWER: 2\nFINAL_ANSWER: `two`"
        )
        self.assertEqual(answer, "two")
        self.assertEqual(status, "marker")

    def test_falls_back_to_last_nonempty_line_without_an_api_call(self) -> None:
        answer, status = baseline.extract_final_answer("short check\n42\n")
        self.assertEqual(answer, "42")
        self.assertEqual(status, "last_nonempty_line")

    def test_vote_keys_are_gold_independent_and_normalize_safe_equivalents(self) -> None:
        self.assertEqual(
            baseline.canonical_vote_key("3"),
            baseline.canonical_vote_key("3.0"),
        )
        self.assertEqual(
            baseline.canonical_vote_key("Yes."),
            baseline.canonical_vote_key("yes"),
        )
        self.assertEqual(
            baseline.canonical_vote_key("['A', 'B']"),
            baseline.canonical_vote_key("['b', 'a']"),
        )
        self.assertNotEqual(
            baseline.canonical_vote_key("1:2"),
            baseline.canonical_vote_key("0.5"),
        )

    def test_self_consistency_majority_and_all_different_tie(self) -> None:
        majority = baseline.select_self_consistent_answer(
            [
                {"pred_answer": "3"},
                {"pred_answer": "3.0"},
                {"pred_answer": "4"},
            ]
        )
        self.assertEqual(majority["pred_answer"], "3")
        self.assertEqual(majority["selected_choice_index"], 0)
        self.assertFalse(majority["all_different_tie"])

        tie = baseline.select_self_consistent_answer(
            [
                {"pred_answer": "A"},
                {"pred_answer": "B"},
                {"pred_answer": "C"},
            ]
        )
        self.assertEqual(tie["pred_answer"], "A")
        self.assertEqual(tie["selected_choice_index"], 0)
        self.assertTrue(tie["all_different_tie"])


class TransportTest(unittest.TestCase):
    @mock.patch.object(baseline.urllib.request, "urlopen")
    def test_n_three_is_sent_once_and_all_choices_are_parsed(
        self, mocked_urlopen: mock.Mock
    ) -> None:
        mocked_urlopen.return_value = FakeHTTPResponse(
            {
                "model": "gpt-5-test",
                "choices": [
                    {
                        "index": index,
                        "message": {"content": f"answer-{index}"},
                        "finish_reason": "stop",
                    }
                    for index in range(3)
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 6,
                    "total_tokens": 10,
                },
            }
        )

        result = baseline.send_chat_completion(
            api_key="secret",
            base_url="https://example.invalid/v1",
            model_name="gpt-5-test",
            messages=[{"role": "user", "content": "Reply A."}],
            n=3,
            temperature=0.6,
            top_p=0.95,
            max_tokens=32,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            request_timeout=1.0,
            max_retries=0,
            request_limiter=None,
        )

        self.assertEqual(mocked_urlopen.call_count, 1)
        request = mocked_urlopen.call_args.args[0]
        request_payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request_payload["n"], 3)
        self.assertEqual(len(result["choices"]), 3)
        self.assertEqual(result["usage"]["total_tokens"], 10)


class CheckpointAndMaterializationTest(unittest.TestCase):
    def test_resume_reuses_cot_and_only_retries_correction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.jsonl"
            writer = baseline.CheckpointWriter(checkpoint_path)
            item = dataset_item()

            first_calls = [
                chat_result(
                    [
                        "1. Count.\nFINAL_ANSWER: 2",
                        "Count two.\nFINAL_ANSWER: 2",
                        "Rows are two.\nFINAL_ANSWER: 2",
                    ]
                ),
                RuntimeError("correction unavailable"),
            ]
            with mock.patch.object(
                baseline,
                "send_chat_completion",
                side_effect=first_calls,
            ):
                first_state = baseline.run_one(
                    item, None, runtime_config("n"), writer
                )

            self.assertEqual(first_state["cot_status"], "success")
            self.assertEqual(first_state["self_correction_status"], "fail")
            saved_state = baseline.load_checkpoint(checkpoint_path)[item["id"]]

            with mock.patch.object(
                baseline,
                "send_chat_completion",
                return_value=chat_result(
                    ["The count is correct.\nFINAL_ANSWER: 2"]
                ),
            ) as retry_call:
                resumed_state = baseline.run_one(
                    item,
                    saved_state,
                    runtime_config("n"),
                    writer,
                )

            self.assertEqual(retry_call.call_count, 1)
            self.assertEqual(resumed_state["cot_status"], "success")
            self.assertEqual(
                resumed_state["self_correction_status"], "success"
            )
            self.assertEqual(
                resumed_state["self_correction_pred_answer"], "2"
            )

    def test_materialized_files_keep_evaluator_fields(self) -> None:
        item = dataset_item()
        state = {
            "id": item["id"],
            "model_name": "gpt-5-test",
            "prompt_version": baseline.PROMPT_VERSION,
            "multi_choice_mode": "n",
            "generation_config": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 2000,
            },
            "cot_status": "success",
            "cot_choices": [
                {
                    "index": 0,
                    "response": "FINAL_ANSWER: 2",
                    "pred_answer": "2",
                    "parse_status": "marker",
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "response": "FINAL_ANSWER: 2",
                    "pred_answer": "2",
                    "parse_status": "marker",
                    "finish_reason": "stop",
                },
                {
                    "index": 2,
                    "response": "FINAL_ANSWER: 3",
                    "pred_answer": "3",
                    "parse_status": "marker",
                    "finish_reason": "stop",
                },
            ],
            "self_consistency": {
                "pred_answer": "2",
                "selected_choice_index": 0,
                "all_different_tie": False,
                "vote_counts": [],
            },
            "self_correction_status": "success",
            "self_correction_response": "Check.\nFINAL_ANSWER: 2",
            "self_correction_pred_answer": "2",
            "self_correction_parse_status": "marker",
            "self_correction_finish_reason": "stop",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = baseline.materialize_results(
                [item], {item["id"]: state}, Path(temp_dir)
            )
            for path in paths.values():
                with path.open("r", encoding="utf-8") as input_file:
                    result = json.loads(input_file.readline())
                self.assertEqual(result["id"], item["id"])
                self.assertEqual(result["answer"], ["2"])
                self.assertEqual(result["pred_answer"], "2")
                self.assertEqual(result["execute_status"], "success")


if __name__ == "__main__":
    unittest.main()
