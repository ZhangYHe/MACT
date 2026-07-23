import json
import os
import sys
import unittest
from unittest import mock


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

import agents  # noqa: E402
from utils import normalize_table_rows, table2df, table_linear  # noqa: E402


def make_crt_agent(question, table=None):
    table = table or [["name", "value"], ["a", "1"]]
    normalized = normalize_table_rows(table)
    agent = agents.ReactAgent.__new__(agents.ReactAgent)
    agent.question = question
    agent.context = ""
    agent.task = "crt"
    agent.table_string = table_linear(normalized, num_row=None)
    agent.table_df = table2df(normalized)
    agent.table_dfs = [agent.table_df]
    agent.df_path = None
    agent.question_profile = None
    agent.tool_events = []
    agent.scratchpad = ""
    agent.verification_cache = {}
    agent.use_verifier = True
    agent.disable_search = False
    agent.disable_calculate = False
    agent.disable_coding_agent = False
    agent.debug_llm_io = False
    agent.debug_log_path = ""
    agent.example_id = "crt-test-example"
    agent.step_n = 1
    agent.last_tool_error = ""
    agent.last_verifier_feedback = ""
    agent.postprocess_pred_answer = True
    agent.raw_pred_answer = ""
    agent.postprocess_trace = []
    agent.candidate_source = ""
    agent.verifier_attempts = 0
    agent.verifier_rejections = 0
    agent.finish_candidates = []
    agent.finish_candidate_records = []
    agent.applied_crt_patches = []
    agent.applied_patch_ids = []
    agent.without_tool = False
    agent.question_profile = agent._apply_router_guards(
        agent._default_question_profile(), agent.get_literal_header_matches())
    return agent


class NoApiTestCase(unittest.TestCase):
    def setUp(self):
        self.api_guard = mock.patch.object(
            agents,
            "get_completion",
            side_effect=AssertionError("CRT unit tests must not call an external API"),
        )
        self.api_guard.start()

    def tearDown(self):
        self.api_guard.stop()


class CrtAnswerContractTest(NoApiTestCase):
    def test_preserves_explicit_comparison_vocabulary(self):
        agent = make_crt_agent(
            "How do wins compare? Answer with only 'more', 'less' or 'equal' that is most accurate."
        )

        self.assertEqual(
            agent.question_profile["answer_contract"]["allowed_labels"],
            ["more", "less", "equal"],
        )
        self.assertEqual(agent._postprocess_final_answer("more"), "more")
        self.assertEqual(agent._postprocess_final_answer("less"), "less")
        self.assertEqual(agent._postprocess_final_answer("equal"), "equal")

    def test_non_explicit_comparison_is_not_rewritten_to_a_synonym(self):
        agent = make_crt_agent(
            "How does the total gross of studio A compare to studio B?"
        )

        self.assertEqual(agent._postprocess_final_answer("higher"), "higher")

    def test_postprocess_does_not_round_or_change_score_spacing(self):
        average = make_crt_agent("What is the average value?")
        correlation = make_crt_agent("What is the correlation coefficient?")
        ratio = make_crt_agent("What is the ratio of wins to losses?")
        score = make_crt_agent("What is the most common score?")

        self.assertEqual(average._postprocess_final_answer("1.373626"), "1.373626")
        self.assertEqual(correlation._postprocess_final_answer("0.9775108"), "0.9775108")
        self.assertEqual(ratio._postprocess_final_answer("3:2"), "3:2")
        self.assertEqual(score._postprocess_final_answer("1 - 0"), "1 - 0")

    def test_target_aware_contract_patches(self):
        count = make_crt_agent(
            "How many teams had a winning percentage greater than 70%?"
        )
        entity = make_crt_agent(
            "Which ground had the highest average crowd attendance?"
        )
        direction = make_crt_agent(
            "Did the average duration decrease or increase over time?"
        )
        implicit_binary = make_crt_agent(
            "Are there any teams with the same score?"
        )
        range_agent = make_crt_agent(
            "What was the range of scores? (From min to max)"
        )

        self.assertEqual(
            count.question_profile["answer_contract"]["output_kind"], "count")
        self.assertIn("count_metric_condition", count.applied_patch_ids)
        self.assertEqual(
            entity.question_profile["answer_contract"]["output_kind"], "entity")
        self.assertIn("entity_ranked_by_metric", entity.applied_patch_ids)
        self.assertEqual(
            direction.question_profile["answer_contract"]["allowed_labels"],
            ["increase", "decrease"],
        )
        self.assertEqual(
            implicit_binary.question_profile["answer_contract"]["allowed_labels"],
            ["Yes", "No"],
        )
        self.assertEqual(
            range_agent.question_profile["answer_contract"]["output_kind"], "range")

    def test_contract_rejects_metric_when_question_asks_for_entity(self):
        agent = make_crt_agent(
            "Which ground had the highest average crowd attendance?"
        )
        result = agent._rule_verify_answer("20069.000")

        self.assertFalse(result["valid"])
        self.assertEqual(result["error_type"], "answer_shape_error")

    def test_crt_verifier_prompt_has_no_scitab_policy(self):
        agent = make_crt_agent(
            "Answer with only 'Yes' or 'No' that is most accurate."
        )
        captured = []

        def verify_call(prompt, phase):
            captured.append((prompt, phase))
            return json.dumps({
                "valid": True,
                "error_type": "none",
                "reason": "supported",
                "suggested_next_action": "",
            })

        agent._call_plan_llm_once = verify_call
        result = agent.verify_finish_answer("Yes")

        self.assertTrue(result["valid"])
        self.assertEqual(captured[0][1], "verifier")
        self.assertIn("answerable CRT-QA", captured[0][0])
        self.assertNotIn("SciTab label policy", captured[0][0])

    def test_rejects_abstention_and_preserves_finish_candidate(self):
        agent = make_crt_agent(
            "Answer with only 'Yes' or 'No' that is most accurate."
        )
        agent.finish_candidates = ["No."]
        agent.get_crt_direct_fallback_answer = lambda: "N/A"

        result = agent._get_safe_quick_answer()

        self.assertEqual(result, "No")
        self.assertEqual(agent.candidate_source, "preserved_finish_candidate")

    def test_two_rejections_preserve_finish_without_direct_fallback(self):
        agent = make_crt_agent(
            "How do wins compare? Answer with only 'more', 'less' or 'equal' that is most accurate."
        )
        agent.plan_backend = "openai"
        agent.direct_reasoning = False
        agent.use_pre_answer = False
        agent.use_router = False
        agent.max_steps = 10
        agent.max_actual_steps = 10
        agent.actual_step_n = 1
        agent.finished = False
        agent.answer = ""
        agent.run_status = "running"
        agent.pre_ans = None
        agent.empty_parse_streak = 0
        agent.parse_failures = []
        agent.prompt_agent_gpt = lambda **_kwargs: ["mocked"]
        agent.as_reward_fn = lambda _sampled: (
            f"Thought {agent.step_n}: supported",
            f"Action {agent.step_n}: Finish[more]",
            "",
            [],
        )
        agent.verify_finish_answer = lambda _answer: {
            "valid": False,
            "error_type": "unsupported_answer",
            "reason": "mock rejection",
            "suggested_next_action": "retry",
        }
        calls = []
        agent.get_crt_direct_fallback_answer = lambda: calls.append("direct") or "more"

        agent.step()
        agent.step()
        agent.run(reset=False)

        self.assertEqual(agent.verifier_rejections, 2)
        self.assertEqual(agent.verifier_attempts, 2)
        self.assertEqual(calls, [])
        self.assertEqual(agent.answer, "more")
        self.assertEqual(agent.candidate_source, "preserved_finish_candidate")

    def test_no_finish_candidate_uses_one_mocked_evidence_fallback(self):
        agent = make_crt_agent("What is the total value?")
        calls = []
        agent.get_crt_direct_fallback_answer = (
            lambda: calls.append("direct") or "1"
        )

        result = agent._get_safe_quick_answer()

        self.assertEqual(result, "1")
        self.assertEqual(calls, ["direct"])
        self.assertEqual(agent.candidate_source, "crt_evidence_fallback")


class CrtCalculationGuardTest(NoApiTestCase):
    def test_row_count_shortcut_only_accepts_unconditional_retrieved_rows(self):
        agent = make_crt_agent(
            "How many rows match?",
            [["score"], ["2-3"], ["3-1"]],
        )

        self.assertTrue(agent._instruction_counts_recent_rows(
            "count the matching rows in the retrieved table"
        ))
        self.assertFalse(agent._instruction_counts_recent_rows(
            "compute each score difference and count rows where difference equals 1"
        ))
        self.assertFalse(agent._instruction_counts_recent_rows(
            "group by team and count each team"
        ))

    def test_calculation_invariants_reject_invalid_scalars(self):
        agent = make_crt_agent(
            "What is the correlation coefficient?",
            [["x", "y"], ["1", "2"], ["2", "3"]],
        )

        self.assertIn("[-1, 1]", agent._validate_crt_calculation_result(
            "calculate the correlation coefficient", "1.5", agent.table_df, 2
        ))
        agent.question_profile["aggregation_operator"] = "count"
        self.assertIn("complete input row count", agent._validate_crt_calculation_result(
            "count rows where the values are equal", "2", agent.table_df, 2
        ))
        self.assertEqual(agent._validate_crt_calculation_result(
            "calculate the correlation coefficient", "0.75", agent.table_df, 2
        ), "")

    def test_assignment_statement_does_not_use_direct_eval_path(self):
        agent = make_crt_agent("What percentage matches?")
        calls = []
        agent.numerical_tool = (
            lambda instruction, recent, *_args, **_kwargs:
            calls.append(instruction) or ["3.85%"]
        )

        result = agent.calculator_tool(
            "count_total = 26; count_within = 1; "
            "percentage = (count_within / count_total) * 100",
            agent.table_df,
        )

        self.assertEqual(result, ["3.85%"])
        self.assertEqual(len(calls), 1)

    def test_missing_python_block_retries_once_with_mock(self):
        agent = make_crt_agent("What is the total?")
        agent.code_backend = "openai"
        calls = []
        agent.prompt_agent_gpt_coder = (
            lambda prompt, phase:
            calls.append((prompt, phase)) or ["```python\nfinal_result = '1'\n```"]
        )

        output, retried = agent._retry_crt_missing_python_block(
            "The answer is one.", "mock task", "calculate_code_retry")

        self.assertTrue(retried)
        self.assertIn("final_result", output)
        self.assertEqual(len(calls), 1)

    def test_crt_calculation_scope_exposes_full_and_retrieved_tables(self):
        agent = make_crt_agent(
            "What percentage matches?",
            [["name", "value"], ["a", "1"], ["b", "2"]],
        )
        recent = table2df([["name", "value"], ["a", "1"]])

        scope_code = agent._crt_dataframe_scope_code(recent)

        self.assertIn("retrieved_df = df.copy()", scope_code)
        self.assertIn("full_table_df = df.copy()", scope_code)
        self.assertEqual(
            agent._crt_validation_row_count(1, "result = len(full_table_df)"),
            2,
        )

    def test_crt_filters_search_from_allowed_tools(self):
        agent = make_crt_agent("What is the value?")

        self.assertNotIn("Search", agent.question_profile["allowed_tools"])


if __name__ == "__main__":
    unittest.main()
