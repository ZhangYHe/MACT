""" Utility classes and functions related to MACT (NAACL 2025). 

Copyright (c) 2025 Robert Bosch GmbH 


This program is free software: you can redistribute it and/or modify 

it under the terms of the GNU Affero General Public License as published 

by the Free Software Foundation, either version 3 of the License, or 

(at your option) any later version. 

This program is distributed in the hope that it will be useful, 

but WITHOUT ANY WARRANTY; without even the implied warranty of 

MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 

GNU Affero General Public License for more details. 

You should have received a copy of the GNU Affero General Public License 

along with this program.  If not, see <https://www.gnu.org/licenses/>. 

""" 

try:
    from langchain.prompts import PromptTemplate
except Exception:
    class PromptTemplate:
        def __init__(self, input_variables, template):
            self.input_variables = input_variables
            self.template = template

        def format(self, **kwargs):
            return self.template.format(**kwargs)

DIRECT_AGENT = """You are given a question, some optinal context, and a table in string format. Solve the question with reasoning and output the final answer following: Therefore, the answer is: [answer].
Here are some examples:
{examples}
(END OF EXAMPLES)
Now answer the following question
Table: 
{table}
Context: {context}
Question: {question}
Answer: 

[BREAK]
You are given a question, some optinal context, and a table in python dataframe format. Continue writing the code to solve the question and store the final answer in a variable "result". You may also directly answer the question by storing the answer in the variable "result" if no code is needed.
Here are some examples:
{examples}
(END OF EXAMPLES)
Now answer the following question:
Dataframe code: {table}
Context: {context}
Question: {question}
Code: 

"""


QUESTION_ROUTER_PROMPT = """You are an expert in table question answering. Classify the question using the question, context, table headers, and representative column values.

Choose exactly one question_type from:
lookup, filter, comparison, aggregation, arithmetic, multi_hop.

Return only a JSON object with these keys:
{{
  "question_type": "...",
  "target_columns": ["..."],
  "candidate_columns": ["..."],
  "constraints": ["..."],
  "required_operations": ["..."],
  "aggregation_operator": "none",
  "membership_predicate": "",
  "answer_shape": "scalar",
  "composite_columns": [],
  "allowed_tools": ["Retrieve", "Calculate", "Search", "Finish"],
  "ambiguous": false,
  "ambiguity_reason": "",
  "requires_evidence": false,
  "reasoning_pattern": "..."
}}

Use "Search" only when external entity knowledge may be needed. Include "Verify" in allowed_tools only when verification may be useful for ambiguous, multi-step, or calculation-heavy questions.
Treat literal header matches as primary evidence: include them in target_columns or candidate_columns, and do not replace a matched header with a semantically related column without first retrieving both. Use representative values to distinguish column-level patterns, row lookups, and outside-world interpretations. Set ambiguous and requires_evidence to true when multiple columns are plausible, and require retrieval of the competing columns before calculation.
Choose aggregation_operator from none, count, sum, average, min, max, ratio, common_prefix, compare. State the exact membership predicate for filtered sets; do not treat every value other than "No" as affirmative. Choose answer_shape from scalar, multi_value, composite. A composite is one answer made from multiple columns; list those columns in original table-header order.

Question: {question}
Context: {context}
Table headers: {headers}
Literal header matches in the question: {literal_header_matches}
Representative column values: {column_samples}
"""


VERIFY_PROMPT = """You are verifying a table question answering reasoning step or final answer.

Judge whether the claim directly answers the question and is supported by the table, context, and successful tool evidence. Check whether the claim is non-empty, uses the intended target column, satisfies every constraint, performs the required operation, and has the expected final-answer shape. A claim merely appearing somewhere in the table or reasoning history is not sufficient. Reject mathematically correct calculations that use the wrong column.

Final-answer shape checks:
- If the question asks for a compact comparison, the final claim should be the relationship or conclusion, not a long evidence list.
- If the answer is one combined item made from multiple fields, it should remain one answer item; do not split it as multiple answers.
- If the question asks for average, ratio, or percentage and does not specify precision, prefer a concise rounded answer, usually two decimal places, over a long floating-point artifact.
- Preserve compact table-style numeric records such as 1-7 rather than adding spaces around the hyphen.

SciTab label policy:
- The final answer must be exactly one of supports, refutes, or not enough info.
- Use refutes only when the table or caption directly contradicts a required value, trend, entity, condition, or comparison.
- Missing exact metrics, missing precision/recall when only F-score is shown, missing statistical significance, missing causal mechanism, unclear experiment-setting mapping, or insufficient table-number/context evidence should usually be not enough info rather than refutes.
- If successful retrieved evidence clearly identifies a relevant table section or column that differs from the router profile, evaluate the claim against the retrieved evidence instead of rejecting solely because of the initial router profile.

Return only a JSON object with these keys:
{{
  "valid": true,
  "error_type": "none",
  "reason": "...",
  "suggested_next_action": "..."
}}

If the claim is not supported, set valid to false and choose one error_type from:
empty_result, missing_constraint, wrong_operation, unsupported_answer, unclear.

Question: {question}
Context: {context}
Table:
{table}
Question routing profile:
{question_profile}
Tool execution records:
{tool_events}
Reasoning history:
{scratchpad}
Claim to verify: {claim}
"""


CRT_VERIFY_PROMPT = """You are verifying a candidate answer for an answerable CRT-QA table question.

Judge whether the candidate directly answers the question and is supported by the table, title, and successful tool evidence. Check the intended columns, every filter and grouping constraint, the required calculation, and the supplied answer contract. A candidate merely appearing in the table or reasoning history is not sufficient.

CRT-specific policy:
- Every question is answerable from the supplied table and title. Never request or return supports, refutes, not enough info, or N/A.
- If answer_contract.allowed_labels is non-empty, the candidate must use exactly one of those labels. Do not replace an explicitly requested label with a synonym.
- A numeric answer must preserve the representation requested by the contract: number, percentage, fraction, or colon ratio.
- A compact final answer must contain only the denotation, without an explanation or evidence list.
- Failed older tool attempts do not invalidate a later successful calculation. Judge the latest relevant successful evidence.

Return only a JSON object with these keys:
{{
  "valid": true,
  "error_type": "none",
  "reason": "...",
  "suggested_next_action": "..."
}}

If the candidate is not supported, set valid to false and choose one error_type from:
empty_result, missing_constraint, wrong_operation, unsupported_answer, answer_shape_error, unclear.

Question: {question}
Table title: {context}
Table:
{table}
Question routing profile:
{question_profile}
Answer contract:
{answer_contract}
Tool execution records:
{tool_events}
Reasoning history:
{scratchpad}
Candidate answer: {claim}
"""


CRT_DIRECT_FALLBACK_PROMPT = """Answer this answerable CRT-QA table question independently.

Use only the supplied table and table title. Return only the final denotation, with no reasoning, prefix, explanation, or markdown. Follow the answer contract exactly. Never return supports, refutes, not enough info, or N/A.

Table:
{table}
Table title: {context}
Question: {question}
Answer contract: {answer_contract}
Final answer:
"""


SCITAB_VERIFY_PROMPT = """You are independently classifying the relationship between a scientific claim and one table plus its caption.

Do not evaluate or defend a previously proposed label. Determine the evidence relation from scratch.

Return only this JSON object:
{{
  "evidence_relation": "supports",
  "alignment": {{
    "entity": true,
    "metric": true,
    "condition": true,
    "unit": true,
    "semantic_role": true
  }},
  "reason": "...",
  "suggested_next_action": "..."
}}

evidence_relation must be exactly supports, refutes, or not enough info.

Decision policy:
- supports: the aligned table/caption evidence entails the claim.
- refutes: entity, metric, condition, unit, and semantic role all align, and the evidence directly shows an opposite value, direction, entity, or comparison.
- not enough info: any required field is absent or misaligned, or the evidence neither entails nor directly contradicts the claim.
- A missing metric is not a contradiction. Precision cannot be inferred from F-score; recall cannot be inferred from F-score; statistical significance cannot be inferred from unmarked numeric differences.
- A different table number means the supplied evidence may not be the table referenced by the claim, so use not enough info rather than refutes.
- Do not use an output representation column to contradict a claim about model inputs, or vice versa. Treat different semantic roles as not enough info.
- A claim about low-degree nodes is aligned with a table section that explicitly buckets Max Node Out-degree or node degree. Compare the low-degree bucket before deciding; do not mark it not enough info merely because the wording is not identical.
- For a multi-metric "A outperforms B" claim, A supports the claim when it is no worse on every corresponding metric and strictly better on at least one; ties do not require not enough info.

Original scientific claim:
{original_claim}
Table caption/context:
{context}
Table:
{table}
Question routing profile:
{question_profile}
Successful and failed tool execution records:
{tool_events}
Reasoning history:
{scratchpad}
"""


ROUTED_CONTEXT_TEMPLATE = """Question routing profile:
- question_type: {question_type}
- target_columns: {target_columns}
- candidate_columns: {candidate_columns}
- literal_header_matches: {literal_header_matches}
- constraints: {constraints}
- required_operations: {required_operations}
- aggregation_operator: {aggregation_operator}
- membership_predicate: {membership_predicate}
- answer_shape: {answer_shape}
- answer_contract: {answer_contract}
- composite_columns: {composite_columns}
- allowed_tools: {allowed_tools}
- ambiguous: {ambiguous}
- ambiguity_reason: {ambiguity_reason}
- requires_evidence: {requires_evidence}
- reasoning_pattern: {reasoning_pattern}
Use this profile to choose actions, but keep following the original task format. When requires_evidence is true, retrieve the literal header matches and competing candidate columns before calculating or finishing.
"""


VERIFY_ACTION_INSTRUCTION = """Additional optional action:
(Verify) Verify[claim], which checks whether a retrieved result, calculated result, or candidate final answer is supported. Use Verify only when you are uncertain about evidence, constraints, or a multi-step/calculation-heavy result. Do not verify every step.
"""


REACT_INSTRUCTION_TAT = """Solve a table question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be four types: 
(1) Retrieve[cells], which retrieves certain cell(s) from the table and returns the retrieved cells in string format.
(2) Look up[information], which looks up the information in the context (if any) and returns the information in string format.
(3) Calculate[formular/instruction], which carries out calculations based on the formular, or the instruction and returns the calculated results.
(4) Finish[answer], which returns only the final answer and finishes the task. Do not include explanations in Finish. If the answer contains multiple items, return them as structured separate items instead of a prose sentence.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Now generating the Thought, Action, Observation for the following instance:
Table: 
{table}
Context: {context}
Question: {question}
{scratchpad}"""

REACT_INSTRUCTION_WTQ = """Solve a table question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be four types: 
(1) Retrieve[cells], which retrieves certain cell(s) from the table and returns the retrieved cells in string format.
(2) Calculate[formular/instruction], which carries out calculations based on the formular, or the instruction and returns the calculated results.
(3) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists.
(4) Finish[answer], which returns only the final answer and finishes the task. Do not include explanations in Finish. If the answer contains multiple items, return them as structured separate items instead of a prose sentence.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Now generating the Thought, Action, Observation for the following instance:
Table: 
{table}
Context: {context}
Question: {question}
{scratchpad}"""

REACT_INSTRUCTION_CRT = """Solve a table question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be four types: 
(1) Retrieve[cells], which retrieves certain cell(s) from the table and returns the retrieved cells in string format.
(2) Calculate[formular/instruction], which carries out calculations based on the formular, or the instruction and returns the calculated results.
(3) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists.
(4) Finish[answer], which returns only the final answer and finishes the task. Do not include explanations in Finish. If the answer contains multiple items, return them as structured separate items instead of a prose sentence.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Now generating the Thought, Action, Observation for the following instance:
Table: 
{table}
Table title: {context}
Question: {question}
{scratchpad}"""


REACT_INSTRUCTION_SCITAB = """Solve a table question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be four types: 
(1) Retrieve[cells], which retrieves certain cell(s) from the table and returns the retrieved cells in string format.
(2) Calculate[formular/instruction], which carries out calculations based on the formular, or the instruction and returns the calculated results.
(3) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists.
(4) Finish[answer], which returns only the final answer and finishes the task. Do not include explanations in Finish. If the answer contains multiple items, return them as structured separate items instead of a prose sentence.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Now generating the Thought, Action, Observation for the following instance:
Table: 
{table}
Context: {context}
{question}
{scratchpad}"""


REACT_INSTRUCTION_DATABENCH = """Solve a table question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be two types: 
(1) Operate[instruction], which carries out operations such as information retrieval or calculations based on the instruction and returns the retrieved or calculated results.
(2) Finish[answer], which returns only the final answer and finishes the task. Do not include explanations in Finish. If the answer contains multiple items, return them as structured separate items instead of a prose sentence.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Now generating the Thought, Action, Observation for the following instance:
Table: 
{table}
Context: {context}
{question}
{scratchpad}"""

TABLE_OPERATION_PROMPT = """
You are given an instruction and a table in pandas dataframe format. Write python code in one code block to retrieve the most relevant rows or/and columns according to the instruction. Return the result in pandas dataframe format and rename it after 'new_table'. Do not use print in the code.
Below are two examples:
{examples}
Now please write code for the following instruction.
Instruction:{instruction}
Table dataframe code:{table_df}
Code:
"""

NUMERICAL_OPERATION_PROMPT = """
According to the instruction, write python code in one code block to perform calculations based on the given pandas dataframe. Return the final result after the variable name final_result. The final result can be of either pandas dataframe or string type. Do not use other data type. Do not use print statement in the code block.
Below are two examples:
{examples}
Now generate python code according to the following instruction.
Instruction: {instruction}
Dataframe code: {table_df}
Code: 
"""

NUMERICAL_OPERATION_PROMPT_LONG_TABLE = """
According to the instruction, write a function named after 'target_function' in one python code block to perform calculations on a dataframe object. The given dataframe shows only two records of the original data due to its large size. However, you should be able to infer the data type based on the given dataframe. Return only the python function without any execution and do not use print statement in the code block.
Below are two examples:
{examples}
Now generate python code according to the following instruction.
Instruction: {instruction}
Dataframe code for the first two records: {table_df}
Code: 
"""

NUMERICAL_OPERATION_PROMPT_LONG_TABLE_GLOBAL = """
You are an expert in python code generation. \
Write a python function named 'target_function' according to the given plan using pandas dataframe. \
The given dataframe shows only two records of the original data due to its large size. The main goal of showing the dataframe is to show the data type associated to each column. \
However, you should not operate any code based on the given dataframe, since it does not contain all information about the table.  \
Below are two examples
{examples}
Now generate the python function according to the given plan.
Plan: {instruction}
Dataframe code for the first two records: {table_df}
Code: 
"""


GLOBAL_PLAN_DATABENCH = """
You are an expert in analyzing table data and generate step-by-step plans to solve any questions related to long tables.
The following table only shows the first three rows of the table due to its large size.
Please generate a stey by step plan to address the question, following the below requirement:
1. A plan should contains no more than 4 steps.
2. Each step should be in one line.
3. Return only the step-wise plan and nothing else.
4. No repetition of the plan.
Please return only a four-step plan and nothing else.
Following are two examples
{examples}
Now generate a plan for to address the following question and table. The plan should contain maximum 4 steps, with each step one line.
Table: {table}
Context: {context}
Question: {question} 
"""


react_agent_prompt_wtq = PromptTemplate(
    input_variables=["examples", "table", "context", "question", "scratchpad"],
    template=REACT_INSTRUCTION_WTQ,
)

react_agent_prompt_crt = PromptTemplate(
    input_variables=["examples", "table", "context", "question", "scratchpad"],
    template=REACT_INSTRUCTION_CRT,
)

react_agent_prompt_scitab = PromptTemplate(
    input_variables=["examples", "table", "context", "question", "scratchpad"],
    template=REACT_INSTRUCTION_SCITAB,
)

react_agent_prompt_tat = PromptTemplate(
    input_variables=["examples", "table", "context", "question", "scratchpad"],
    template=REACT_INSTRUCTION_TAT,
)

react_agent_prompt_databench = PromptTemplate(
    input_variables=["examples", "table", "context", "question", "scratchpad"],
    template=REACT_INSTRUCTION_DATABENCH,
)


global_plan_prompt = PromptTemplate(
    input_variables=["examples", "table", "context", "question"],
    template=GLOBAL_PLAN_DATABENCH,
)
