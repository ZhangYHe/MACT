#!/usr/bin/env python3
"""Mock-only tests for technical repair and Self-Refine-1 supplementation."""

from __future__ import annotations

import hashlib
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
import self_refine_supplement as supplement  # noqa: E402


def item() -> dict:
    return {
        "id": "crt:test:1",
        "statement": "How many rows are present?",
        "table_text": [["name"], ["a"], ["b"]],
        "answer": ["2"],
        "task": "crt",
    }


def answer_choice(
    response: str,
    *,
    index: int,
    finish_reason: str = "stop",
) -> dict:
    pred_answer, parse_status = baseline.extract_final_answer(response)
    return {
        "index": index,
        "response": response,
        "pred_answer": pred_answer,
        "parse_status": parse_status,
        "finish_reason": finish_reason,
    }


def source_state(*, cot_status: str = "success") -> dict:
    dataset_item = item()
    cot_prompt = baseline.build_cot_prompt(
        dataset_item["statement"], "name\na\nb"
    )
    state = {
        "schema_version": 1,
        "prompt_version": baseline.PROMPT_VERSION,
        "id": dataset_item["id"],
        "item": dataset_item,
        "model_name": "gpt-5-test",
        "generation_config": {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 2000,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "sc_samples": 3,
        },
        "cot_prompt_sha256": hashlib.sha256(
            cot_prompt.encode("utf-8")
        ).hexdigest(),
        "cot_status": cot_status,
        "cot_error": "",
        "cot_choices": [
            answer_choice("FINAL_ANSWER: 2", index=0),
            answer_choice("FINAL_ANSWER: 2", index=1),
            answer_choice("FINAL_ANSWER: 3", index=2),
        ],
        "cot_usage": {},
        "self_correction_status": "success",
        "self_correction_response": "Check.\nFINAL_ANSWER: 2",
        "self_correction_pred_answer": "2",
        "self_correction_parse_status": "marker",
        "self_correction_finish_reason": "stop",
        "self_correction_usage": {},
    }
    if cot_status != "success":
        state.update(
            {
                "cot_error": "HTTP 400: content_filter",
                "cot_choices": [],
                "self_correction_status": None,
            }
        )
    return state


def runtime_config() -> dict:
    return {
        "api_key": "not-a-real-key",
        "base_url": "https://example.invalid/v1",
        "model_name": "gpt-5-test",
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 2000,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "request_timeout": 1.0,
        "max_retries": 0,
        "request_limiter": None,
    }


def chat_result(
    contents: list[str],
    *,
    finish_reasons: list[str] | None = None,
) -> dict:
    reasons = finish_reasons or ["stop"] * len(contents)
    return {
        "http_status": 200,
        "model": "gpt-5-test",
        "choices": [
            {
                "index": index,
                "content": content,
                "finish_reason": reasons[index],
            }
            for index, content in enumerate(contents)
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": len(contents) * 4,
            "total_tokens": 10 + len(contents) * 4,
        },
    }


class SupplementPromptAndRetryTest(unittest.TestCase):
    def test_prompts_separate_feedback_from_refinement(self) -> None:
        feedback = supplement.build_feedback_messages("prompt", "initial")
        refinement = supplement.build_refinement_messages(
            "prompt", "initial", "feedback"
        )
        self.assertEqual(
            feedback[-1]["content"], supplement.FEEDBACK_INSTRUCTION
        )
        self.assertNotIn("FINAL_ANSWER", feedback[-1]["content"])
        self.assertEqual(
            refinement[-1]["content"], supplement.REFINEMENT_INSTRUCTION
        )
        self.assertIn("FINAL_ANSWER", refinement[-1]["content"])

    def test_empty_length_response_retries_once_at_4000(self) -> None:
        calls = [
            chat_result([""], finish_reasons=["length"]),
            chat_result(["Feedback is brief."]),
        ]
        with mock.patch.object(
            baseline,
            "send_chat_completion",
            side_effect=calls,
        ) as mocked_send:
            choice, attempts = supplement._send_single_with_length_retry(
                messages=[{"role": "user", "content": "test"}],
                config=runtime_config(),
                validator=supplement.feedback_choice_is_valid,
                initial_max_tokens=2000,
            )
        self.assertEqual(choice["response"], "Feedback is brief.")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(
            mocked_send.call_args_list[0].kwargs["max_tokens"], 2000
        )
        self.assertEqual(
            mocked_send.call_args_list[1].kwargs["max_tokens"], 4000
        )


class RepairAndResumeTest(unittest.TestCase):
    def test_only_invalid_cot_slots_are_replaced_and_correction_is_redone(
        self,
    ) -> None:
        source = source_state()
        source["cot_choices"][0] = answer_choice(
            "", index=0, finish_reason="length"
        )
        source["cot_choices"][2] = answer_choice(
            "FINAL_ANSWER: 3",
            index=2,
            finish_reason="content_filter",
        )
        source_path = Path("/tmp/source-checkpoint.jsonl")

        calls = [
            chat_result(["FINAL_ANSWER: 2", "FINAL_ANSWER: 2"]),
            chat_result(["Check.\nFINAL_ANSWER: 2"]),
            chat_result(["The initial answer is correct."]),
            chat_result(["FINAL_ANSWER: 2"]),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = baseline.CheckpointWriter(
                Path(temp_dir) / "supplement.jsonl"
            )
            with mock.patch.object(
                baseline,
                "send_chat_completion",
                side_effect=calls,
            ) as mocked_send:
                result = supplement.run_one(
                    source,
                    None,
                    source_path,
                    runtime_config(),
                    writer,
                )

        self.assertEqual(mocked_send.call_count, 4)
        self.assertEqual(mocked_send.call_args_list[0].kwargs["n"], 2)
        self.assertEqual(
            mocked_send.call_args_list[0].kwargs["max_tokens"], 4000
        )
        self.assertEqual(result["cot_repaired_indices"], [0, 2])
        self.assertEqual(
            result["self_correction_repair_mode"], "regenerated"
        )
        self.assertEqual(result["refinement_pred_answer"], "2")
        repaired = supplement.merged_cot_choices(source, result)
        self.assertEqual(repaired[1]["response"], "FINAL_ANSWER: 2")

    def test_source_prompt_filter_is_terminal_without_an_api_call(self) -> None:
        source = source_state(cot_status="fail")
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = baseline.CheckpointWriter(
                Path(temp_dir) / "supplement.jsonl"
            )
            with mock.patch.object(
                baseline, "send_chat_completion"
            ) as mocked_send:
                result = supplement.run_one(
                    source,
                    None,
                    Path("/tmp/source-checkpoint.jsonl"),
                    runtime_config(),
                    writer,
                )
        mocked_send.assert_not_called()
        self.assertEqual(result["source_status"], "unavailable")
        self.assertEqual(result["refinement_status"], "blocked")
        self.assertTrue(supplement.state_is_terminal(result))

    def test_resume_after_refinement_failure_calls_only_refinement(self) -> None:
        source = source_state()
        source_path = Path("/tmp/source-checkpoint.jsonl")
        existing = supplement._new_state(
            source, source_path, runtime_config()
        )
        existing.update(
            {
                "source_status": "available",
                "cot_repair_status": "success",
                "cot_repaired_indices": [],
                "self_correction_repair_status": "success",
                "self_correction_repair_mode": "reused",
                "feedback_status": "success",
                "feedback_response": "The answer is correct.",
                "feedback_finish_reason": "stop",
                "refinement_status": "fail",
                "refinement_error": "temporary failure",
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = baseline.CheckpointWriter(
                Path(temp_dir) / "supplement.jsonl"
            )
            with mock.patch.object(
                baseline,
                "send_chat_completion",
                return_value=chat_result(["FINAL_ANSWER: 2"]),
            ) as mocked_send:
                result = supplement.run_one(
                    source,
                    existing,
                    source_path,
                    runtime_config(),
                    writer,
                )
        self.assertEqual(mocked_send.call_count, 1)
        self.assertEqual(
            mocked_send.call_args.kwargs["messages"][-1]["content"],
            supplement.REFINEMENT_INSTRUCTION,
        )
        self.assertEqual(result["refinement_status"], "success")


class MaterializationTest(unittest.TestCase):
    def test_outputs_keep_evaluator_fields_and_original_results_separate(
        self,
    ) -> None:
        source = source_state()
        source_path = Path("/tmp/source-checkpoint.jsonl")
        state = supplement._new_state(
            source, source_path, runtime_config()
        )
        state.update(
            {
                "source_status": "available",
                "cot_repair_status": "success",
                "cot_repaired_indices": [],
                "self_correction_repair_status": "success",
                "self_correction_repair_mode": "reused",
                "feedback_status": "success",
                "feedback_response": "The answer is correct.",
                "feedback_finish_reason": "stop",
                "refinement_status": "success",
                "refinement_response": "FINAL_ANSWER: 2",
                "refinement_pred_answer": "2",
                "refinement_parse_status": "marker",
                "refinement_finish_reason": "stop",
            }
        )
        supplement._update_usage(state)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            original = run_dir / "zero_shot_cot" / "results.jsonl"
            original.parent.mkdir(parents=True)
            original.write_text("original remains untouched\n")
            summary = supplement.materialize_results(
                [source], {source["id"]: state}, run_dir
            )
            result_path = (
                run_dir / "self_refine_1" / "results.jsonl"
            )
            result = json.loads(result_path.read_text().splitlines()[0])

            self.assertEqual(
                original.read_text(), "original remains untouched\n"
            )
            self.assertEqual(result["id"], source["id"])
            self.assertEqual(result["answer"], ["2"])
            self.assertEqual(result["pred_answer"], "2")
            self.assertEqual(result["execute_status"], "success")
            self.assertEqual(
                summary["coverage"]["self_refine_1"]["success"], 1
            )


if __name__ == "__main__":
    unittest.main()
