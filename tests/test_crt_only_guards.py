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

    def test_non_explicit_comparison_can_use_crt_canonical_relation(self):
        agent = make_crt_agent(
            "How does the total gross of studio A compare to studio B?"
        )

        self.assertEqual(agent._postprocess_final_answer("higher"), "larger")

    def test_applies_numeric_precision_without_changing_ratio_shape(self):
        average = make_crt_agent("What is the average value?")
        correlation = make_crt_agent("What is the correlation coefficient?")
        ratio = make_crt_agent("What is the ratio of wins to losses?")

        self.assertEqual(average._postprocess_final_answer("1.373626"), "1.374")
        self.assertEqual(correlation._postprocess_final_answer("0.9775108"), "0.9775")
        self.assertEqual(ratio._postprocess_final_answer("3:2"), "3:2")

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

    def test_two_rejections_trigger_one_mocked_direct_fallback(self):
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
        self.assertEqual(calls, ["direct"])
        self.assertEqual(agent.answer, "more")


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


if __name__ == "__main__":
    unittest.main()
