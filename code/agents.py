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

import json
import os
import re
import string
import sys
import hashlib
from collections import Counter, OrderedDict, defaultdict

import pandas as pd
import wikipedia
from fewshots_table import (DEMO_CRT, DEMO_CRT_DIRECT, DEMO_SCITAB,
                            DEMO_SCITAB_DIRECT, DEMO_TAT, DEMO_TAT_DIRECT,
                            DEMO_WTQ, DEMO_WTQ_DIRECT,
                            NUMERICAL_OPERATION_EXAMPLE,
                            TABLE_OPERATION_EXAMPLE, DEMO_DATABENCH,
                            NUMERICAL_OPERATION_EXAMPLE_LONG_TABLE, GLOBAL_PLAN_EXAMPLES,
                            NUMERICAL_OPERATION_EXAMPLE_LONG_TABLE_GLOBAL)
from llm import OpenSourceLLM
from prompts_table import (DIRECT_AGENT, NUMERICAL_OPERATION_PROMPT,
                           TABLE_OPERATION_PROMPT, react_agent_prompt_crt,
                           react_agent_prompt_scitab, react_agent_prompt_tat,
                           react_agent_prompt_wtq, NUMERICAL_OPERATION_PROMPT_LONG_TABLE,
                           NUMERICAL_OPERATION_PROMPT_LONG_TABLE_GLOBAL,
                           react_agent_prompt_databench, global_plan_prompt,
                           QUESTION_ROUTER_PROMPT, ROUTED_CONTEXT_TEMPLATE,
                           VERIFY_ACTION_INSTRUCTION, VERIFY_PROMPT)
from sglang import assistant, function, gen, user
from tot import llm_reward, vote_prompt_as
from utils import (extract_from_outputs, parse_action, table2df,
                   table_linear)

all_input_token, all_output_token = 0, 0
_WARNED_MISSING_USAGE = False
_LOGGED_DATASET_HINTS = set()
DEFAULT_OPENAI_MAX_TOKENS = 2000
OPENAI_EMPTY_LENGTH_RETRY_MAX_TOKENS = 4000


class SimpleWikipediaSearch:
    def search(self, entity):
        try:
            return wikipedia.summary(entity, sentences=3, auto_suggest=False)
        except wikipedia.DisambiguationError as e:
            if e.options:
                return wikipedia.summary(
                    e.options[0], sentences=3, auto_suggest=False)
            raise
        except wikipedia.PageError:
            results = wikipedia.search(entity, results=1)
            if results:
                return wikipedia.summary(
                    results[0], sentences=3, auto_suggest=False)
            raise


def _safe_model_dump(obj):
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return {
            key: value
            for key, value in obj.__dict__.items()
            if not key.startswith("_")
        }
    return {}


def get_completion(
    prompt,
    client,
    n,
    model,
    phase="",
    step_n=None,
    debug_logger=None,
    debug_full_prompt=False,
    max_tokens=DEFAULT_OPENAI_MAX_TOKENS,
    temperature=0.6,
    retry_empty_length=True,
):
    global all_input_token, all_output_token, _WARNED_MISSING_USAGE
    messages = [{"role": "user", "content": prompt}]

    def create_response(token_budget):
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=token_budget,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            n=n,
            stop=None
        )

    response = create_response(max_tokens)
    retry_reason = ""
    if retry_empty_length and _is_empty_length_response(response):
        retry_reason = (
            f"empty content with finish_reason=length; retrying with "
            f"max_tokens={OPENAI_EMPTY_LENGTH_RETRY_MAX_TOKENS}"
        )
        if debug_logger is not None:
            _log_completion_debug(
                response=response,
                prompt=prompt,
                model=model,
                n=n,
                phase=phase,
                step_n=step_n,
                debug_logger=debug_logger,
                debug_full_prompt=debug_full_prompt,
                warning=retry_reason,
                max_tokens=max_tokens,
                temperature=temperature,
                retry_attempt=0,
            )
        response = create_response(max(max_tokens * 2, OPENAI_EMPTY_LENGTH_RETRY_MAX_TOKENS))
    usage = getattr(response, "usage", None)
    input_token_num = getattr(usage, "prompt_tokens", None) if usage else None
    output_token_num = getattr(usage, "completion_tokens", None) if usage else None
    missing_usage = input_token_num is None or output_token_num is None
    warning = (
        "LLM response usage is missing prompt_tokens or completion_tokens; "
        "token counters will use 0 for missing values."
    ) if missing_usage else ""
    if missing_usage and not _WARNED_MISSING_USAGE:
        print(f"Warning: {warning}", file=sys.stderr)
        _WARNED_MISSING_USAGE = True
    input_token_num = input_token_num or 0
    output_token_num = output_token_num or 0
    all_input_token += input_token_num
    all_output_token += output_token_num
    # print(all_input_token, all_output_token)
    if debug_logger is not None:
        _log_completion_debug(
            response=response,
            prompt=prompt,
            model=model,
            n=n,
            phase=phase,
            step_n=step_n,
            debug_logger=debug_logger,
            debug_full_prompt=debug_full_prompt,
            warning="; ".join(item for item in [warning, retry_reason] if item),
            max_tokens=max(max_tokens * 2, OPENAI_EMPTY_LENGTH_RETRY_MAX_TOKENS)
            if retry_reason else max_tokens,
            temperature=temperature,
            retry_attempt=1 if retry_reason else 0,
        )
    return [item.message.content or "" for item in response.choices]


def _is_empty_length_response(response):
    choices = getattr(response, "choices", []) or []
    if not choices:
        return False
    for item in choices:
        content = getattr(getattr(item, "message", None), "content", None) or ""
        if content.strip() or getattr(item, "finish_reason", None) != "length":
            return False
    return True


def _log_completion_debug(
    response,
    prompt,
    model,
    n,
    phase,
    step_n,
    debug_logger,
    debug_full_prompt,
    warning,
    max_tokens,
    temperature,
    retry_attempt,
):
    choices = []
    for index, item in enumerate(getattr(response, "choices", []) or []):
        content = getattr(getattr(item, "message", None), "content", None) or ""
        choices.append({
            "index": index,
            "finish_reason": getattr(item, "finish_reason", None),
            "content_len": len(content),
            "content_preview": content[:1000],
        })
    debug_logger({
        "phase": phase,
        "step_n": step_n,
        "model": model,
        "n": n,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "retry_attempt": retry_attempt,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_preview": prompt[:1000],
        "prompt": prompt if debug_full_prompt else None,
        "raw_choices": choices,
        "usage": _safe_model_dump(getattr(response, "usage", None)),
        "warning": warning,
    })


@function
def table_operation(s, instruction, table_df):
    prompt = TABLE_OPERATION_PROMPT.format(
        instruction=instruction, table_df=table_df, examples=TABLE_OPERATION_EXAMPLE)
    s += user(prompt)
    s += assistant(gen("result", max_tokens=2000, temperature=0.6))


@function
def code_revise(s, current_error, extracted_code, table_df):
    prompt = f"You are an expert in revising code. The following code results in an error when executing on the table dataframe (the dataframe only shows the first two records of original data due to its large size). Please revise the code to address the error and only return the revised code in one python code block. \n Table dataframe: {table_df}\n Erroneous code: {extracted_code}\n Error message: {current_error}\n Revised code:"
    s += user(prompt)
    s += assistant(gen("result", max_tokens=2000, temperature=0.6))


@function
def numerical_operation(s, instruction, table_df):
    prompt = NUMERICAL_OPERATION_PROMPT.format(
        instruction=instruction, table_df=table_df, examples=NUMERICAL_OPERATION_EXAMPLE)
    s += user(prompt)
    s += assistant(gen("result", max_tokens=4000, temperature=0.6))


@function
def numerical_operation_long_table(s, instruction, table_df, global_planning=False):
    if global_planning:
        prompt = NUMERICAL_OPERATION_PROMPT_LONG_TABLE_GLOBAL.format(
            instruction=instruction, table_df=table_df, examples=NUMERICAL_OPERATION_EXAMPLE_LONG_TABLE_GLOBAL)
    else:
        prompt = NUMERICAL_OPERATION_PROMPT_LONG_TABLE.format(
            instruction=instruction, table_df=table_df, examples=NUMERICAL_OPERATION_EXAMPLE_LONG_TABLE)
    s += user(prompt)
    s += assistant(gen("result", max_tokens=4000, temperature=0.6))


@function
def direct_code(s, prompt):
    s += user(prompt)
    s += assistant(gen("result", max_tokens=4000, temperature=0.6))


def validate_gloabl_result(executed_results, threshold=3):
    answer = Counter(executed_results).most_common(1)[0][0]
    frequency = Counter(executed_results).most_common(1)[0][1]
    if frequency >= threshold and answer != "":
        return True, answer
    else:
        return False, answer


class ReactAgent:
    def __init__(self,
                 question: str,
                 table: str,
                 table_df: str,
                 df_path: str,
                 context: str,
                 key: str,
                 answer: str = '',
                 plan_model_name: str = '',
                 code_model_name: str = '',
                 model=None,
                 tokenizer=None,
                 max_steps: int = 5,
                 task: str = '',
                 codeagent_endpoint=None,
                 plan_sample: int = 5,
                 code_sample: int = 5,
                 max_actual_steps: int = 5,
                 as_reward='consistency',
                 use_pre_answer=False,
                 answer_aggrement=1.0,
                 direct_reasoning=False,
                 without_tool=False,
                 long_table_op='ignore',
                 code_as_observation=False,
                 debugging=False,
                 client=None,
                 plan_backend='auto',
                 code_backend='auto',
                 use_router=False,
                 use_verifier=False,
                 use_repair=False,
                 use_code_repair=False,
                 disable_search=False,
                 disable_calculate=False,
                 disable_coding_agent=False,
                 log_router=True,
                 example_id="",
                 debug_llm_io=False,
                 debug_full_prompt=False,
                 debug_log_path="",
                 openai_max_tokens=DEFAULT_OPENAI_MAX_TOKENS,
                 openai_temperature=0.6,
                 ) -> None:

        vllm = model
        self.plan_backend = plan_backend
        self.code_backend = code_backend
        self.use_router = use_router
        self.use_verifier = use_verifier
        self.use_code_repair = use_code_repair or use_repair
        self.use_repair = self.use_code_repair
        self.disable_search = disable_search
        self.disable_calculate = disable_calculate
        self.disable_coding_agent = disable_coding_agent
        self.log_router = log_router
        self.example_id = example_id
        self.debug_llm_io = debug_llm_io
        self.debug_full_prompt = debug_full_prompt
        self.debug_log_path = debug_log_path
        self.openai_max_tokens = openai_max_tokens
        self.openai_temperature = openai_temperature
        self.question_profile = None
        self.verification_cache = {}
        if self.plan_backend == "local":
            self.llm = OpenSourceLLM(
                model_name=plan_model_name,
                model=model,
                vllm=vllm,
                tokenizer=tokenizer
            )
        self.client = client
        self.tokenizer = tokenizer
        self.question = question
        self.table_string = table_linear(
            table, num_row=None) if isinstance(table, list) else table
        self.long_table = False
        self.debugging = debugging
        # if len(table) * len(table[0]) > 50:  # 10*5    300
        #     self.long_table = True
        #     if long_table_op == 'short-table':
        #         self.table_string = table_linear(table, num_row=5)   ##num_row=20
        #         remain = len(table) - 5
        #         self.table_string += f"\n[...Remaining {remain} rows not shown due to large table size...]"
        self.table_df = table_df
        self.table_dfs = [table_df]
        self.df_path = df_path
        self.context = context
        self.answer = answer
        self.plan_model_name = plan_model_name
        self.code_model_name = code_model_name
        self.key = " ".join(key) if isinstance(key, list) else key
        self.max_steps = max_steps
        self.codeagent_endpoint = codeagent_endpoint
        self.plan_sample = plan_sample
        self.code_sample = code_sample
        self.max_actual_steps = max_actual_steps
        self.as_reward = as_reward
        self.task = task
        self.evaluator_output = []
        self.use_pre_answer = use_pre_answer
        self.pre_ans_all = []
        self.docstore = SimpleWikipediaSearch()  # Search
        self.answer_aggrement = answer_aggrement
        self.direct_reasoning = direct_reasoning
        self.without_tool = without_tool
        self.long_table_op = long_table_op
        self.code_as_observation = code_as_observation
        self.llm_sampled = []
        self.code_sampled = []
        self.direct_sampled = []
        self.dataset_hint = self._load_dataset_hint(task)

        if not self.direct_reasoning:
            if task == "tat":
                self.react_examples = DEMO_TAT
                self.agent_prompt = react_agent_prompt_tat
            elif task == "scitab":
                self.react_examples = DEMO_SCITAB
                self.agent_prompt = react_agent_prompt_scitab
            elif task == "crt":
                self.react_examples = DEMO_CRT
                self.agent_prompt = react_agent_prompt_crt
            elif task == "wtq":
                self.react_examples = DEMO_WTQ
                self.agent_prompt = react_agent_prompt_wtq
            elif task == "databench":
                self.react_examples = DEMO_DATABENCH
                self.agent_prompt = react_agent_prompt_databench
                self.global_plan_prompt = global_plan_prompt
                self.global_plan_examples = GLOBAL_PLAN_EXAMPLES

        else:
            self.agent_prompt = DIRECT_AGENT
            if task == "tat":
                self.react_examples = DEMO_TAT_DIRECT
            elif task == "scitab":
                self.react_examples = DEMO_SCITAB_DIRECT
            elif task == "crt":
                self.react_examples = DEMO_CRT_DIRECT
            elif task == "wtq":
                self.react_examples = DEMO_WTQ_DIRECT
            self.code_prompt = self.agent_prompt.split("[BREAK]")[-1].strip()
            self.code_examples = self.react_examples.split(
                "[BREAK]")[-1].strip()
            self.text_prompt = self.agent_prompt.split("[BREAK]")[0].strip()
            self.text_examples = self.react_examples.split("[BREAK]")[
                0].strip()

        self.__reset_agent()

    def code_extract_retrieve(self, code_strings):
        rows = []
        new_table = ""
        p = re.compile(r"```[Python|python].*```", re.DOTALL)
        try:
            executable_code = re.findall(p, code_strings)[0]
            executable_code = "\n".join(executable_code.split("\n")[1:-1])
            df_string = self.table_df
            executable_code = "\n".join([df_string, executable_code])
            loc = {}
            exec(executable_code, globals(), loc)
            new_table = loc['new_table']
        except:
            pass
        if isinstance(new_table, pd.Series):
            new_table = new_table.to_frame()
        if isinstance(new_table, pd.DataFrame):
            if not new_table.empty:
                # to string format
                header = new_table.columns.tolist()
                rows = new_table.values.tolist()
                rows.insert(0, header)
        return rows

    def retriever_tool(self, instruction):
        if self.disable_coding_agent:
            return []
        max_attempt = self.code_sample
        results = []
        results2dfs = defaultdict(list)
        if self.code_model_name == self.plan_model_name:
            # use one base model
            prompt = TABLE_OPERATION_PROMPT.format(
                instruction=instruction, table_df=self.table_df, examples=TABLE_OPERATION_EXAMPLE)
            messages = [{"role": "user", "content": prompt}]
            if self.code_backend == "local":
                codes = self.llm(
                    messages, num_return_sequences=max_attempt, return_prob=False)
            else:
                codes = self.prompt_agent_gpt_coder(prompt)

            for code_strings in codes:
                rows = self.code_extract_retrieve(code_strings)
                if rows != []:
                    result = table_linear(rows, num_row=None).strip()
                    results2dfs[result].append(table2df(rows))
                else:
                    result = ""
                results.append(result)

        else:
            # code generation batching
            if self.code_backend == "local":
                batch_data = [{"instruction": instruction,
                               "table_df": self.table_df} for i in range(max_attempt)]
                states = table_operation.run_batch(
                    batch_data, progress_bar=True, backend=self.codeagent_endpoint)
                code_strings = [s["result"] for s in states]
            else:
                prompt = TABLE_OPERATION_PROMPT.format(
                    instruction=instruction, table_df=self.table_df, examples=TABLE_OPERATION_EXAMPLE)
                code_strings = self.prompt_agent_gpt_coder(prompt)

            for code_string in code_strings:
                rows = self.code_extract_retrieve(code_string)
                if isinstance(rows, list) and rows != []:
                    # if len(rows) > 7:  # not showing the rest
                    #     remain = len(rows) - 7
                    #     result = table_linear(rows, num_row=7).strip(
                    #     ) + f"\n[...Remaining {remain} rows not shown due to large table size...]"
                    # else:
                    result = table_linear(rows, num_row=None)
                    results2dfs[result.strip()].append(table2df(rows))
                else:
                    result = ""
                results.append(result)

        results = [res for res in results if not res == ""]
        try:
            sorted_df = sorted(results2dfs, key=lambda key: len(
                results2dfs[key]), reverse=True)
            target_df = list(sorted_df.values())[0][0]
            self.table_dfs.append(target_df)
        except:
            pass
        return results

    def calculator_tool(self, eqution, recent_table_df):
        def clean_eqution(eqution):
            eqution = eqution.replace(",", "")
            eqution = eqution.replace("$", "")
            return eqution

        def recent_table_row_count(table_df):
            try:
                if isinstance(table_df, pd.DataFrame):
                    return len(table_df)
                if isinstance(table_df, str):
                    loc = {}
                    exec(table_df, globals(), loc)
                    df = loc.get("df")
                    if isinstance(df, pd.DataFrame):
                        return len(df)
            except Exception:
                pass
            return None

        def should_count_recent_rows(instruction):
            text = str(instruction).lower()
            if "count" not in text:
                return False
            grouped_count_terms = [
                "count each",
                "count by",
                "frequency",
                "frequencies",
                "unique",
                "distinct",
            ]
            if any(term in text for term in grouped_count_terms):
                return False
            scoped_to_recent = any(
                term in text
                for term in [
                    "observation",
                    "recent table",
                    "retrieved",
                    "listed",
                    "matching rows",
                ]
            )
            row_like = any(
                term in text
                for term in [
                    "row",
                    "rows",
                    "entry",
                    "entries",
                    "item",
                    "items",
                    "number of",
                ]
            )
            return scoped_to_recent and row_like

        if should_count_recent_rows(eqution):
            row_count = recent_table_row_count(recent_table_df)
            if row_count is not None:
                return row_count

        try:
            eqution = clean_eqution(eqution)
            loc = {}
            eqution_ = "result = "+eqution
            exec(eqution_, globals(), loc)
            if self.without_tool:
                return [], ""
            else:
                return loc['result'], ""
        except:
            result = ""
            # try with the coder
            if not self.disable_coding_agent:
                try:
                    result = self.numerical_tool(
                        eqution, recent_table_df, self.df_path, global_planning=False)
                except:
                    pass
            return result

    def code_extract_calculator(self, code_strings, table_df, original_df):
        result = ""
        rows = []
        current_error = None
        executable_code = None
        p = re.compile(r"```[Python|python].*```", re.DOTALL)
        if not self.task == "databench":
            try:
                executable_code = re.findall(p, code_strings)[0]
                executable_code = "\n".join(executable_code.split("\n")[1:-1])
                df_string = table_df
                executable_code = "\n".join([df_string, executable_code])
                loc = {}
                exec(executable_code, globals(), loc)
                result = loc['final_result']
            except Exception as e:
                # print(e)
                current_error = e
            if isinstance(result, pd.Series):
                result = result.to_frame()

            if isinstance(result, pd.DataFrame) and not result.empty:
                # to string format
                header = result.columns.tolist()
                rows = result.values.tolist()
                rows.insert(0, header)
                result = table_linear(rows, num_row=None)

            if not isinstance(result, str):
                try:
                    # if it is numpy array
                    rows = result.tolist()
                    result = table_linear(rows, num_row=None)
                except:
                    result = str(result)
            return result, rows, current_error, executable_code
        else:
            try:
                executable_code = re.findall(p, code_strings)[0]
                executable_code = "\n".join(executable_code.split("\n")[1:-1])
                # make sure only function is returned
                return_ids = [i for i, line in enumerate(executable_code.split(
                    "\n")) if "return" in line and "#" not in line.split("return")[0]]
                if return_ids:
                    return_ids = return_ids[-1]
                    executable_code = "\n".join(
                        executable_code.split("\n")[:return_ids+1])
                executable_code = "\n".join(
                    ["import pandas as pd\nimport numpy as np\nimport pandas\nimport numpy\n", executable_code, f"final_result=target_function(original_df)"])
                loc = {"original_df": original_df}
                exec(executable_code, globals(), loc)
                result = loc['final_result']
            except Exception as e:
                # print(e)
                current_error = e
            if isinstance(result, pd.Series):
                result = result.to_frame()
            if isinstance(result, pd.DataFrame) and not result.empty:
                # to string format
                self.original_df = result
                header = result.columns.tolist()
                rows = result.values.tolist()
                rows.insert(0, header)
                if len(result) > 10:
                    # too long
                    remain_line = len(result) - 4
                    result = table_linear(
                        rows, num_row=3) + f"\n ...[remaining {remain_line} rows not shown due to large table size]..."
                    rows = rows[:3]
                else:
                    result = table_linear(rows, num_row=None)

            if not isinstance(result, str):
                # result is a variable
                with open("temp.txt", "w") as f:
                    print(result, file=f)
                with open("temp.txt", "r") as f:
                    result = f.readlines()
                result = "\n".join(result)
            return result, rows, current_error, executable_code

    def numerical_tool(self, instruction, table_df, df_path=None, global_planning=False):
        if self.disable_coding_agent:
            return []
        max_attempt = self.code_sample
        results, generated_code = [], []
        results2df = defaultdict(list)
        original_df = None
        if df_path:
            original_df = pd.read_parquet(df_path, engine='pyarrow')

        if self.code_model_name == self.plan_model_name:
            prompt = NUMERICAL_OPERATION_PROMPT.format(
                instruction=instruction, table_df=table_df, examples=NUMERICAL_OPERATION_EXAMPLE)
            messages = [{"role": "user", "content": prompt}]
            if self.code_backend == "local":
                codes = self.llm(
                    messages, num_return_sequences=max_attempt, return_prob=False)
            else:
                codes = self.prompt_agent_gpt_coder(prompt)
            for code_strings in codes:
                result, rows, error, extracted_code = self.code_extract_calculator(
                    code_strings, table_df, original_df)
                if self.use_code_repair and error is not None and extracted_code:
                    revised_code = self.revise_code(
                        error, extracted_code, table_df)
                    if revised_code:
                        result, rows, _, _ = self.code_extract_calculator(
                            revised_code, table_df, original_df)
                if result != "" and rows != []:
                    try:
                        result = result.strip()
                        results2df[result].append(table2df(rows))
                    except:
                        pass
                results.append(result)
                generated_code.append(extracted_code)

        else:
            if self.code_backend == "local":
                # code generation batching
                batch_data = [{"instruction": instruction, "table_df": table_df}
                              for i in range(max_attempt)]
                if self.task != "databench":
                    states = numerical_operation.run_batch(
                        batch_data, progress_bar=True, backend=self.codeagent_endpoint)
                else:
                    if not global_planning:
                        states = numerical_operation_long_table.run_batch(
                            batch_data, progress_bar=True, backend=self.codeagent_endpoint)
                    else:
                        batch_data = [{"instruction": instruction, "table_df": self.table_df, "global_planning": True}
                                      for i in range(max_attempt)]
                        states = numerical_operation_long_table.run_batch(
                            batch_data, progress_bar=True, backend=self.codeagent_endpoint)
                code_strings = [s["result"] for s in states]

            else:
                prompt = NUMERICAL_OPERATION_PROMPT.format(
                    instruction=instruction, table_df=table_df, examples=NUMERICAL_OPERATION_EXAMPLE)
                code_strings = self.prompt_agent_gpt_coder(prompt)

            for code_string in code_strings:
                result, rows, error, extracted_code = self.code_extract_calculator(
                    code_string, table_df, original_df)
                if self.use_code_repair and error is not None and extracted_code:
                    revised_code = self.revise_code(
                        error, extracted_code, table_df)
                    if revised_code:
                        result, rows, _, _ = self.code_extract_calculator(
                            revised_code, table_df, original_df)
                if result != "" and rows != []:
                    try:
                        result = result.strip()
                        results2df[result].append(table2df(rows))
                    except:
                        pass
                results.append(result)
                generated_code.append(extracted_code)
        if not global_planning:
            results = [res for res in results if not res == ""]
            try:
                sorted_df = sorted(results2df, key=lambda key: len(
                    results2df[key]), reverse=True)
                target_df = list(sorted_df.values())[0][0]
                self.table_dfs.append(target_df)
            except:
                pass
            if self.code_as_observation:
                if len(results) > 0:
                    results = Counter(results).most_common(1)[0][0]
            return results
        else:
            self.generated_code = generated_code
            return results

    def as_llm(self, thoughts, actions, observations):
        all_paths = ""
        assert len(thoughts) == len(actions)
        if len(thoughts) > 0:
            all_paths = f"Question: {self.question}\nTable:{self.table_string}Past reasonings:{self.scratchpad}\n"
            current_paths = ""
            for i, (t, a, o) in enumerate(zip(thoughts, actions, observations)):
                sc = "\n".join([t, a, o])
                all_paths += f'current reasoning path {i+1}: {sc}\n'
                current_paths += f'current reasoning path {i+1}: {sc}\n'
            outputs, _, _ = llm_reward(reasoning_paths=all_paths, vote_prompt=vote_prompt_as, model_type="open",
                                       model_name=self.plan_model_name, tokenizer=self.tokenizer, model=self.llm)
            self.evaluator_output.append([current_paths, outputs])
            target_choice = extract_from_outputs(outputs, len(thoughts))
            target_thought = thoughts[target_choice]
            target_action = actions[target_choice]
            try:
                target_observation = observations[target_choice]
            except:
                target_observation = ""
        else:
            target_thought, target_action, target_observation = "", "", ""
        return target_thought, target_action, target_observation

    def as_reward_fn(self, sampled):
        # a reward function to select the most promising steps among sampled
        global all_input_token, all_output_token

        def get_current_step(instance):
            return self._extract_current_step(instance)

        def get_preliminary_ans(sampled):
            mapping = []
            threshold = len(sampled)*self.answer_aggrement
            pre_ans = None
            pre_answers = []
            for i, instance in enumerate(sampled):
                try:
                    instance_ = [line for line in instance.split(
                        "\n") if line.strip() != ""]
                    answer_line = [
                        line for line in instance_ if "Finish" in line]
                    if len(answer_line) > 0:
                        _, pre_answer = parse_action(answer_line[0])
                        pre_answers.append(pre_answer.lower())
                        mapping.append(i)
                except:
                    pass
            try:
                most_common, num_most_common = Counter(
                    pre_answers).most_common(1)[0]  # val, times
            except:
                most_common = ""
                num_most_common = 0
            if num_most_common >= threshold:
                pre_ans = most_common
            assert len(pre_answers) == len(mapping)
            return pre_ans, pre_answers, mapping

        def as_rollout(sampled, actions):
            _, pre_ans_all, mapping = get_preliminary_ans(sampled)
            try:
                common = Counter(pre_ans_all).most_common(1)[0][0]
                sampled_id = [i for i, item in enumerate(
                    pre_ans_all) if item == common]
                sampled_id = [mapping[item] for item in sampled_id]
            except:
                pass
            try:
                target_action = actions[sampled_id[0]]
            except:
                target_action = ""
            return target_action

        def as_consistency(action_thought, observations):
            target_thought, target_action, target_observation = "", "", ""
            if target_thought == "" and target_action == "":
                action_thought = OrderedDict(
                    sorted(action_thought.items(), key=lambda x: len(x[1]), reverse=True))
                # majority action
                try:
                    target_action = list(action_thought.keys())[0]
                    target_thought = [
                        item for item in action_thought[target_action] if item != ""][0]
                    try:
                        target_observation = Counter(
                            observations).most_common(1)[0][0]
                    except:
                        pass
                except:
                    pass
            return target_thought, target_action, target_observation

        thoughts, actions, observations = [], [], []
        pre_ans = None
        action_thought = defaultdict(list)
        action_observation = defaultdict(list)
        # get perliminary answer

        if self.as_reward == "logp" or self.as_reward == "combined":
            log_probs = sampled.pop(-1)

        if self.step_n == 1:
            pre_ans, pre_ans_all, _ = get_preliminary_ans(sampled)
            self.pre_ans = pre_ans
            self.pre_ans_all = pre_ans_all

        target_sample = []
        for i, item in enumerate(sampled):
            t, a, o = get_current_step(item)
            if not t == "" and not a == "":
                thoughts.append(t)
                actions.append(a)
                observations.append(o)
                action_thought[a].append(t)
                action_observation[a].append(o)
                target_sample.append(i)

        if self.as_reward == "consistency":
            target_thought, target_action, target_observation = as_consistency(
                action_thought, observations)

        elif self.as_reward == "llm":
            target_thought, target_action, target_observation = self.as_llm(
                thoughts, actions, observations)

        elif self.as_reward == "logp":
            target_thought, target_action, target_observation = "", "", ""
            log_probs = [log_probs[item] for item in target_sample]
            assert len(log_probs) == len(actions)
            try:
                target_action = actions[log_probs.index(max(log_probs))]
                target_thought = [
                    item for item in action_thought[target_action] if item != ""][0]
                try:
                    target_observation = [
                        item for item in action_observation[target_action] if item != ""][0]
                except:
                    pass
            except:
                pass

        elif self.as_reward == "rollout":
            target_thought, target_action, target_observation = "", "", ""
            target_action = as_rollout(sampled, actions)
            try:
                target_thought = [
                    item for item in action_thought[target_action] if item != ""][0]
                try:
                    target_observation = [
                        item for item in action_observation[target_action] if item != ""][0]
                except:
                    pass
            except:
                pass

        elif self.as_reward == "combined":
            target_thought, target_action, target_observation = "", "", ""
            ac_lst = []
            _, ac, _, = as_consistency(action_thought, observations)
            ac_lst.append(ac)
            _, ac, _ = self.as_llm(thoughts, actions, observations)
            ac_lst.append(ac)
            log_probs = [log_probs[item] for item in target_sample]
            try:
                target_action = actions[log_probs.index(max(log_probs))]
                ac_lst.append(ac)
            except:
                pass
            ac = as_rollout(sampled, actions)
            ac_lst.append(ac)
            target_action = Counter(ac_lst).most_common(1)[0][0]
            try:
                target_thought = [
                    item for item in action_thought[target_action] if item != ""][0]
                try:
                    target_observation = [
                        item for item in action_observation[target_action] if item != ""][0]
                except:
                    pass
            except:
                pass

        return target_thought, target_action, target_observation, observations

    def get_answer_from_llm(self, instance) -> str:
        return instance.split(":")[-1].strip()

    def get_answer_from_code(self, instance) -> str:
        # exec
        p = re.compile(r"```[Python|python].*```", re.DOTALL)
        try:
            executable_code = re.findall(p, instance)[0]
            executable_code = "\n".join(executable_code.split("\n")[1:-1])
            df_string = self.table_df
            executable_code = "\n".join([df_string, executable_code])
            loc = {}
            exec(executable_code, globals(), loc)
            result = loc['result']
        except:
            result = ""
        if not isinstance(result, str):
            result = str(result)
        return result

    def _call_plan_llm_once(self, prompt, phase=""):
        if self.plan_backend == "openai":
            return get_completion(
                prompt,
                client=self.client,
                n=1,
                model=self.plan_model_name,
                phase=phase,
                step_n=self.step_n,
                debug_logger=self._log_llm_io,
                debug_full_prompt=self.debug_full_prompt,
                max_tokens=self.openai_max_tokens,
                temperature=self.openai_temperature,
            )[0]
        return self.llm(prompt, num_return_sequences=1, return_prob=False)[0]

    def _extract_json_object(self, output):
        try:
            output = output.strip()
            if output.startswith("```"):
                output = re.sub(r"^```(?:json)?", "", output)
                output = re.sub(r"```$", "", output).strip()
            matched = re.search(r"\{.*\}", output, re.DOTALL)
            if matched:
                output = matched.group(0)
            return json.loads(output)
        except Exception:
            return None

    def _default_question_profile(self):
        allowed_tools = ["Retrieve", "Calculate", "Search", "Finish"]
        if self.task == "tat":
            allowed_tools = ["Retrieve", "Calculate", "Finish"]
        elif self.task == "databench":
            allowed_tools = ["Operate", "Finish"]
        allowed_tools = self._filter_disabled_tools(allowed_tools)
        if self.use_verifier:
            allowed_tools.append("Verify")
        return {
            "question_type": "multi_hop",
            "target_columns": [],
            "constraints": [],
            "required_operations": [],
            "allowed_tools": allowed_tools,
            "reasoning_pattern": "Use the original MACT ReAct process and choose tools according to the question."
        }

    def _normalize_question_profile(self, profile):
        default_profile = self._default_question_profile()
        if not isinstance(profile, dict):
            return default_profile
        for key, value in default_profile.items():
            profile.setdefault(key, value)
        if not isinstance(profile["allowed_tools"], list):
            profile["allowed_tools"] = default_profile["allowed_tools"]
        if self.use_verifier and "Verify" not in profile["allowed_tools"]:
            profile["allowed_tools"].append("Verify")
        if not self.use_verifier:
            profile["allowed_tools"] = [
                tool for tool in profile["allowed_tools"] if tool != "Verify"]
        profile["allowed_tools"] = self._filter_disabled_tools(
            profile["allowed_tools"])
        return profile

    def _filter_disabled_tools(self, tools):
        filtered = []
        for tool in tools:
            if self.disable_search and tool == "Search":
                continue
            if self.disable_calculate and tool in ["Calculate", "Operate"]:
                continue
            if self.disable_coding_agent and tool == "Retrieve":
                continue
            filtered.append(tool)
        return filtered

    def get_table_headers(self):
        try:
            loc = {}
            exec(self.table_df, globals(), loc)
            return list(loc["df"].columns)
        except Exception:
            try:
                header_line = self.table_string.split("\n")[0]
                return [item.strip() for item in header_line.split("|") if item.strip()]
            except Exception:
                return []

    def _log_llm_io(self, record):
        if not self.debug_llm_io or not self.debug_log_path:
            return
        record = {
            "example_id": self.example_id,
            "task": self.task,
            **record,
        }
        if record.get("prompt") is None:
            record.pop("prompt", None)
        log_dir = os.path.dirname(self.debug_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self.debug_log_path, "a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _strip_markdown_fence(self, text):
        text = str(text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()
        return text

    def _normalize_direct_answer_candidate(self, text):
        text = self._strip_markdown_fence(text)
        if not text:
            return ""
        if "\n" in text:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if len(lines) == 1:
                text = lines[0]
            else:
                answer_lines = [
                    line for line in lines
                    if re.match(r"^(?:final\s+)?answer\s*[:：]", line, flags=re.I)
                ]
                if len(answer_lines) == 1:
                    text = re.split(r"[:：]", answer_lines[0], maxsplit=1)[-1].strip()
                else:
                    return ""
        if re.match(r"^(?:final\s+)?answer\s*[:：]", text, flags=re.I):
            text = re.split(r"[:：]", text, maxsplit=1)[-1].strip()
        if re.search(r"\b(Thought|Action|Observation)\b", text, flags=re.I):
            return ""
        if re.search(r"\b(Retrieve|Operate|Finish|Search|Calculate|Verify)\s*\[", text):
            return ""
        if re.match(r"^(Retrieve|Operate|Search|Calculate|Verify)\b", text, flags=re.I):
            return ""
        if len(text) > 200:
            return ""
        return text.strip()

    def _extract_current_step(self, instance):
        current_thought, current_action, current_observation = "", "", ""
        text = self._strip_markdown_fence(instance)
        if not text:
            return current_thought, current_action, current_observation

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        thought_pattern = re.compile(
            rf"^(?:[-*]\s*)?Thought(?:\s+{self.step_n})?\s*[:\-]\s*(.*)$",
            flags=re.I,
        )
        action_pattern = re.compile(
            rf"^(?:[-*]\s*)?Action(?:\s+{self.step_n})?\s*[:\-]\s*(.*)$",
            flags=re.I,
        )
        observation_pattern = re.compile(
            rf"^(?:[-*]\s*)?Observation(?:\s+{self.step_n})?\s*[:\-]\s*(.*)$",
            flags=re.I,
        )
        next_thought_pattern = re.compile(
            rf"^(?:[-*]\s*)?Thought(?:\s+{self.step_n + 1})?\s*[:\-]",
            flags=re.I,
        )

        observation_start = None
        observation_end = None
        for index, line in enumerate(lines):
            thought_match = thought_pattern.match(line)
            if thought_match and not current_thought:
                current_thought = f"Thought {self.step_n}: {thought_match.group(1).strip()}"
                continue
            action_match = action_pattern.match(line)
            if action_match and not current_action:
                action_body = action_match.group(1).strip()
                if parse_action(action_body)[0] is None:
                    parsed_action = parse_action(line)
                    if parsed_action[0] is not None:
                        action_body = line[line.find(parsed_action[0]):].strip()
                current_action = f"Action {self.step_n}: {action_body}"
                continue
            if observation_pattern.match(line) and observation_start is None:
                observation_start = index
            elif observation_start is not None and next_thought_pattern.match(line):
                observation_end = index
                break

        if not current_action:
            action_match = re.search(
                r"\b(Retrieve|Operate|Finish|Search|Calculate|Verify)\s*\[",
                text,
            )
            if action_match:
                action_type, argument = parse_action(text[action_match.start():])
                if action_type is not None:
                    current_action = f"Action {self.step_n}: {action_type}[{argument}]"
                    if not current_thought:
                        current_thought = f"Thought {self.step_n}: I can proceed with the parsed action."

        if observation_start is not None:
            observation_end = observation_end if observation_end is not None else len(lines)
            current_observation = "\n".join(lines[observation_start:observation_end])

        if current_action and parse_action(current_action)[0] is None:
            current_action = ""
        if current_action and not current_thought:
            current_thought = f"Thought {self.step_n}: I can proceed with the parsed action."
        return current_thought, current_action, current_observation

    def _record_parse_failure(self, sampled):
        sampled = sampled or []
        previews = [str(item or "")[:1000] for item in sampled[:3]]
        direct_candidates = [
            candidate for candidate in
            [self._normalize_direct_answer_candidate(item) for item in sampled]
            if candidate
        ]
        if direct_candidates:
            self.direct_answer_candidate = Counter(direct_candidates).most_common(1)[0][0]
        failure = {
            "step_n": self.step_n,
            "actual_step_n": self.actual_step_n,
            "sampled_count": len(sampled),
            "empty_content_count": sum(1 for item in sampled if not str(item or "").strip()),
            "missing_thought_action_count": len(sampled),
            "first_raw_output_preview": previews[0] if previews else "",
            "raw_output_previews": previews,
            "direct_answer_candidate": self.direct_answer_candidate,
        }
        self.parse_failures.append(failure)
        return failure

    def route_question(self):
        prompt = QUESTION_ROUTER_PROMPT.format(
            question=self.question,
            context=self.context,
            headers=self.get_table_headers()
        )
        try:
            output = self._call_plan_llm_once(prompt, phase="router")
            profile = self._extract_json_object(output)
            self.question_profile = self._normalize_question_profile(profile)
        except Exception:
            self.question_profile = self._default_question_profile()
        if self.log_router:
            print("==============question router===========")
            print(json.dumps(self.question_profile, ensure_ascii=False))
        return self.question_profile

    def _build_control_context(self):
        control_context = ""
        if self.dataset_hint:
            control_context += self.dataset_hint + "\n"
        if self.use_router:
            if self.question_profile is None:
                self.question_profile = self._default_question_profile()
            control_context += ROUTED_CONTEXT_TEMPLATE.format(
                question_type=self.question_profile["question_type"],
                target_columns=self.question_profile["target_columns"],
                constraints=self.question_profile["constraints"],
                required_operations=self.question_profile["required_operations"],
                allowed_tools=self.question_profile["allowed_tools"],
                reasoning_pattern=self.question_profile["reasoning_pattern"]
            )
        if self.use_verifier:
            control_context += VERIFY_ACTION_INSTRUCTION
        if control_context:
            control_context += "\n"
        return control_context

    def _verification_cache_key(self, claim):
        return normalize_answer(str(claim))

    def _format_verification_observation(self, parsed, prefix="Verification"):
        valid = self._is_verification_valid(parsed)
        status = "passed" if valid else "failed"
        reason = parsed.get("reason", "")
        suggested_next_action = parsed.get("suggested_next_action", "")
        error_type = parsed.get("error_type", "none")
        observation = f"{prefix} {status}. error_type: {error_type}. reason: {reason}"
        if suggested_next_action:
            observation += f" suggested_next_action: {suggested_next_action}"
        return observation

    def _is_verification_valid(self, parsed):
        valid = parsed.get("valid", False)
        if isinstance(valid, str):
            return valid.strip().lower() in ["true", "yes", "valid", "pass", "passed"]
        return bool(valid)

    def _rule_verify_answer(self, answer):
        normalized_answer = normalize_answer(str(answer))
        if not normalized_answer:
            return {
                "valid": False,
                "error_type": "empty_result",
                "reason": "The answer is empty.",
                "suggested_next_action": "Continue reasoning and produce a non-empty answer."
            }
        evidence = "\n".join([
            self.scratchpad,
            self.context,
            self.table_string
        ])
        normalized_evidence = normalize_answer(evidence)
        if normalized_answer and normalized_answer in normalized_evidence:
            return {
                "valid": True,
                "error_type": "none",
                "reason": "The answer appears in the available table, context, or reasoning history.",
                "suggested_next_action": ""
            }
        return None

    def _llm_verify_claim(self, claim):
        cache_key = self._verification_cache_key(claim)
        if cache_key in self.verification_cache:
            return self.verification_cache[cache_key]
        prompt = VERIFY_PROMPT.format(
            question=self.question,
            context=self.context,
            table=self.table_string,
            scratchpad=self.scratchpad,
            claim=claim
        )
        try:
            output = self._call_plan_llm_once(prompt, phase="verifier")
            parsed = self._extract_json_object(output)
            if isinstance(parsed, dict):
                parsed.setdefault("valid", False)
                parsed.setdefault("error_type", "unclear")
                parsed.setdefault("reason", "")
                parsed.setdefault("suggested_next_action", "")
                parsed["valid"] = self._is_verification_valid(parsed)
            else:
                parsed = {
                    "valid": False,
                    "error_type": "unclear",
                    "reason": output.strip(),
                    "suggested_next_action": "Continue reasoning with stronger evidence."
                }
        except Exception as e:
            parsed = {
                "valid": False,
                "error_type": "verifier_error",
                "reason": str(e),
                "suggested_next_action": "Continue reasoning without relying on the failed verifier call."
            }
        self.verification_cache[cache_key] = parsed
        return parsed

    def verify_finish_answer(self, answer):
        cache_key = self._verification_cache_key(answer)
        if cache_key in self.verification_cache:
            return self.verification_cache[cache_key]
        rule_result = self._rule_verify_answer(answer)
        if rule_result is not None:
            self.verification_cache[cache_key] = rule_result
            return rule_result
        return self._llm_verify_claim(answer)

    def verifier_tool(self, claim):
        rule_result = self._rule_verify_answer(claim)
        if rule_result is not None:
            self.verification_cache[self._verification_cache_key(claim)] = rule_result
            parsed = rule_result
        else:
            parsed = self._llm_verify_claim(claim)
        return self._format_verification_observation(parsed)

    def repair_feedback(self, action_type, argument):
        return (
            f"Repair feedback: the {action_type} action returned an empty result. "
            f"Revise the next action or code instruction. Previous argument: {argument}"
        )

    def revise_code(self, current_error, extracted_code, table_df):
        try:
            if self.code_backend == "openai":
                prompt = (
                    "You are an expert in revising code. The following code results in an error when executing on "
                    "the table dataframe. Please revise the code to address the error and only return the revised "
                    f"code in one python code block.\nTable dataframe: {table_df}\nErroneous code: {extracted_code}\n"
                    f"Error message: {current_error}\nRevised code:"
                )
                return self.prompt_agent_gpt_coder(prompt, phase="code_repair")[0]
            return code_revise.run(
                current_error=str(current_error),
                extracted_code=extracted_code,
                table_df=table_df,
                backend=self.codeagent_endpoint
            )["result"]
        except Exception:
            return ""

    def run(self, reset=True, given_plan=None) -> None:
        if reset:
            self.__reset_agent()
        if self.use_router and not self.direct_reasoning:
            self.route_question()
        if self.task == "databench":
            if not self.is_finished():
                self.global_planning(given_plan)
        while not self.is_halted() and not self.is_finished():
            # if global planning fail, try step-wise planning
            self.step()

        if not self.answer:
            if self.use_pre_answer:
                valid_pre_answers = self._valid_pre_answers()
                if valid_pre_answers:
                    self.answer = Counter(valid_pre_answers).most_common(1)[0][0]
                    self.finished = True
                    self.run_status = "fallback_answered"
                elif self.direct_answer_candidate and self._can_use_direct_answer_candidate():
                    self.answer = self.direct_answer_candidate
                    self.fallback_answer = self.direct_answer_candidate
                    self.finished = True
                    self.run_status = "fallback_answered"
                else:
                    self.answer = self._get_safe_quick_answer()
                    self.finished = self.answer != "N/A"
                    self.run_status = "fallback_answered" if self.finished else "fallback_na"
            else:
                # direct prompting
                self.answer = self._get_safe_quick_answer()
                self.finished = self.answer != "N/A"
                self.run_status = "fallback_answered" if self.finished else "fallback_na"
        elif self.finished:
            self.run_status = "finished"
        elif self.is_halted() and self.run_status == "running":
            self.run_status = "halted"

    def step(self) -> None:
        if self.direct_reasoning:
            if self.plan_backend == "openai":
                llm_sampled = self.prompt_agent_gpt(mode="text", phase="direct_text")
            else:
                llm_sampled = self.prompt_agent(mode="text")
            llm_sampled_ = [self.get_answer_from_llm(
                item) for item in llm_sampled]
            prompt = self.code_prompt.format(
                examples=self.code_examples, table=self.table_df, question=self.question, context=self.context)
            if self.code_backend == "openai":
                code_sampled = self.prompt_agent_gpt_coder(prompt, phase="direct_code")
            else:
                code_sampled = [direct_code.run(prompt, backend=self.codeagent_endpoint)[
                    "result"] for i in range(self.code_sample)]
            code_sampled_ = [self.get_answer_from_code(
                item) for item in code_sampled]
            self.llm_sampled = [item for item in llm_sampled_ if item != ""]
            self.code_sampled = [item for item in code_sampled_ if item != ""]
            self.direct_sampled = self.llm_sampled + self.code_sampled
            self.history = [llm_sampled, code_sampled]
            self.answer = Counter(self.direct_sampled).most_common(1)[0][0]
            self.finished = True

        else:
            if self.plan_backend == "openai":
                sampled = self.prompt_agent_gpt(phase="react_step")
            else:
                sampled = self.prompt_agent(mode="both")
            self.actual_step_n += 1
            thought, action, observation, all_observations = self.as_reward_fn(
                sampled)
            if thought == "" or action == "":
                self.empty_parse_streak += 1
                self._record_parse_failure(sampled)
                if self.empty_parse_streak >= 3:
                    self.run_status = "halted"
                    self.actual_step_n = self.max_actual_steps + 1
            else:
                self.empty_parse_streak = 0
            if (
                self.use_pre_answer
                and self.pre_ans
                and not self._is_action_like_answer(self.pre_ans)
                and not self.use_verifier
            ):
                self.finished = True
                self.answer = self.pre_ans
            else:
                if thought != "" and action != "":
                    action_type, argument = parse_action(action)
                    if action_type != "Finish":
                        if action_type == "Calculate":
                            if self.disable_calculate:
                                observation = f"Observation {self.step_n}: Calculate tool disabled by ablation."
                            else:
                                recent_table_df = self.table_dfs[-1]
                                new_ob = self.calculator_tool(
                                    argument, recent_table_df=recent_table_df)
                                if not isinstance(new_ob, list):
                                    if new_ob != "":
                                        observation = f"Observation {self.step_n}: {new_ob}"
                                    elif self.disable_coding_agent:
                                        observation = f"Observation {self.step_n}: Coding agent disabled by ablation."
                                else:
                                    # majority voting among tool results and llm results
                                    if new_ob != []:
                                        new_ob = [
                                            f'Observation {self.step_n}: {item}' for item in new_ob]
                                        new_ob += all_observations
                                        observation = Counter(
                                            new_ob).most_common(1)[0][0]
                                    elif self.disable_coding_agent:
                                        observation = f"Observation {self.step_n}: Coding agent disabled by ablation."

                        elif action_type == "Retrieve":
                            if self.disable_coding_agent:
                                observation = f"Observation {self.step_n}: Coding agent disabled by ablation."
                            else:
                                new_ob = self.retriever_tool(
                                    instruction=argument)
                                if new_ob != []:
                                    new_ob = [
                                        f'Observation {self.step_n}: {item}' for item in new_ob]
                                    if not self.long_table and not self.code_as_observation:
                                        new_ob += all_observations
                                    observation = Counter(
                                        new_ob).most_common(1)[0][0]

                        elif action_type == "Search":
                            if self.disable_search:
                                observation = f"Observation {self.step_n}: Search tool disabled by ablation."
                            elif self.without_tool:
                                pass
                            else:
                                try:
                                    observation_wiki = self.docstore.search(
                                        argument)
                                    observation = f"Observation {self.step_n}: {observation_wiki}"
                                except Exception as e:
                                    # cannot find on wikipedia, use llm search results
                                    pass
                        elif action_type == "Operate":
                            if self.disable_calculate:
                                observation = f"Observation {self.step_n}: Calculate tool disabled by ablation."
                            else:
                                recent_table_df = self.table_dfs[-1]
                                new_ob = self.calculator_tool(
                                    argument, recent_table_df=recent_table_df)
                                if new_ob != "":
                                    observation = f"Observation {self.step_n}: {new_ob}"
                                elif self.disable_coding_agent:
                                    observation = f"Observation {self.step_n}: Coding agent disabled by ablation."

                        elif action_type == "Verify" and self.use_verifier:
                            verification = self.verifier_tool(argument)
                            observation = f"Observation {self.step_n}: {verification}"

                        if (
                            observation == ""
                            and action_type in ["Retrieve", "Calculate", "Operate"]
                        ):
                            observation = (
                                f"Observation {self.step_n}: No valid observation was "
                                "produced by the tool. Retry with a more specific action, "
                                "or finish only if the answer is directly supported by the "
                                "table/context already shown."
                            )

                        if observation != "":
                            self.scratchpad += thought + "\n"
                            self.scratchpad += action + "\n"
                            self.scratchpad += observation + "\n"
                            self.step_n += 1

                    else:
                        # finish in the action
                        self.scratchpad += thought + "\n"
                        self.scratchpad += action + "\n"
                        if self.use_verifier:
                            verification = self.verify_finish_answer(argument)
                            observation = self._format_verification_observation(
                                verification, prefix="Final verification")
                            self.scratchpad += f"Observation {self.step_n}: {observation}\n"
                            self.step_n += 1
                            if self._is_verification_valid(verification):
                                self.answer = argument
                                self.finished = True
                            else:
                                pass
                        else:
                            self.answer = argument
                            self.finished = True

                else:
                    # resample
                    pass
                print("==============current step===========")
                print(self.scratchpad)

    def prompt_agent_gpt(self, mode="both", phase="react_step") -> str:
        prompt = self._build_agent_prompt(mode=mode)
        preds = get_completion(
            prompt,
            client=self.client,
            n=self.plan_sample,
            model=self.plan_model_name,
            phase=phase,
            step_n=self.step_n,
            debug_logger=self._log_llm_io,
            debug_full_prompt=self.debug_full_prompt,
            max_tokens=self.openai_max_tokens,
            temperature=self.openai_temperature,
        )
        return preds

    def prompt_agent_gpt_coder(self, prompt, phase="code_generation") -> str:
        preds = get_completion(
            prompt,
            client=self.client,
            n=self.code_sample,
            model=self.code_model_name,
            phase=phase,
            step_n=self.step_n,
            debug_logger=self._log_llm_io,
            debug_full_prompt=self.debug_full_prompt,
            max_tokens=self.openai_max_tokens,
            temperature=self.openai_temperature,
        )
        return preds

    def global_planning(self, given_plan) -> None:
        if not given_plan:
            plan = self.get_global_plan()[0]
            plan = plan.split("Plan:")[-1].strip()
            self.generated_plan = plan
        else:
            self.generated_plan = given_plan
            plan = given_plan
        executed_results = self.numerical_tool(
            plan, self.table_df[0], self.df_path, global_planning=True)
        valid, result = validate_gloabl_result(executed_results)
        if valid:
            self.answer = result
            self.finished = True

    def get_quick_answer(self):
        if self.task == "tat":
            examples = DEMO_TAT_DIRECT
        elif self.task == "scitab":
            examples = DEMO_SCITAB_DIRECT
        elif self.task == "crt":
            examples = DEMO_CRT_DIRECT
        elif self.task == "wtq":
            examples = DEMO_WTQ_DIRECT
        text_prompt = DIRECT_AGENT.split("[BREAK]")[0].strip()
        text_examples = examples.split("[BREAK]")[
            0].strip()
        prompt = text_prompt.format(
            examples=text_examples,
            table=self.table_string,
            context=self.context,
            question=self.question)
        if self.plan_backend == "local":
            answer = self.llm(
                prompt, num_return_sequences=self.plan_sample, return_prob=False)
        else:
            answer = get_completion(
                prompt,
                client=self.client,
                n=self.plan_sample,
                model=self.plan_model_name,
                phase="quick_answer",
                step_n=self.step_n,
                debug_logger=self._log_llm_io,
                debug_full_prompt=self.debug_full_prompt,
                max_tokens=self.openai_max_tokens,
                temperature=self.openai_temperature,
            )
        answers = [ans.split(":")[-1].strip() for ans in answer]
        answer = Counter(answers).most_common(1)[0][0]
        return answer

    def prompt_agent(self, mode="both") -> str:
        prompt = self._build_agent_prompt(mode=mode)
        if self.as_reward == "logp" or self.as_reward == "combined":
            return_prob = True
        else:
            return_prob = False
        return self.llm(prompt, num_return_sequences=self.plan_sample, return_prob=return_prob)

    def get_global_plan(self):
        prompt = self.global_plan_prompt.format(
            examples=self.global_plan_examples,
            table=self.table_string,
            context=self.context,
            question=self.question)
        if self.plan_backend == "openai":
            return get_completion(
                prompt,
                client=self.client,
                n=1,
                model=self.plan_model_name,
                phase="global_plan",
                step_n=self.step_n,
                debug_logger=self._log_llm_io,
                debug_full_prompt=self.debug_full_prompt,
                max_tokens=self.openai_max_tokens,
                temperature=self.openai_temperature,
            )
        return self.llm(prompt, num_return_sequences=1, return_prob=False)

    def _build_agent_prompt(self, mode="both") -> str:
        if mode == "text":
            return self.text_prompt.format(
                examples=self.text_examples,
                table=self.table_string,
                context=self.context,
                question=self.question)
        elif mode == "both":
            format_instruction = (
                f"Format requirement: your next response must contain only the current step, "
                f"with exactly one line starting with Thought {self.step_n}: and exactly one "
                f"line starting with Action {self.step_n}:. Do not include explanations, "
                "markdown, or extra sections outside those two lines.\n"
            )
            scratchpad = self._build_control_context() + self.scratchpad + format_instruction
            return self.agent_prompt.format(
                examples=self.react_examples,
                table=self.table_string,
                context=self.context,
                question=self.question,
                scratchpad=scratchpad)

    def _load_dataset_hint(self, task):
        task = str(task).strip().lower()
        if not task:
            return ""
        hint_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "prompts",
            "dataset_hints",
            f"{task}_reasoning_hints.txt",
        )
        try:
            with open(hint_path, "r", encoding="utf-8") as input_file:
                hint = input_file.read().strip()
        except FileNotFoundError:
            return ""
        except OSError as exc:
            print(f"==============dataset hint warning===========")
            print(f"Failed to load dataset hint from {hint_path}: {exc}")
            return ""

        if hint and hint_path not in _LOGGED_DATASET_HINTS:
            print("==============dataset hint loaded===========")
            print(f"Task: {task}")
            print(f"Path: {hint_path}")
            print(hint)
            _LOGGED_DATASET_HINTS.add(hint_path)
        return hint

    def _valid_pre_answers(self):
        return [
            answer
            for answer in self.pre_ans_all
            if not self._is_action_like_answer(answer)
        ]

    def _can_use_direct_answer_candidate(self):
        if not self.direct_answer_candidate:
            return False
        if not self.use_verifier:
            return True
        verification = self.verify_finish_answer(self.direct_answer_candidate)
        self.direct_answer_candidate_verification = verification
        return self._is_verification_valid(verification)

    def _get_safe_quick_answer(self):
        answer = self.get_quick_answer()
        self.fallback_answer = answer
        if self._is_action_like_answer(answer):
            self.fallback_rejected_reason = "empty_or_action_like"
            print("==============answer fallback warning===========")
            print(f"Rejected action-like quick answer: {answer}")
            return "N/A"
        self.fallback_rejected_reason = ""
        return answer

    def _is_action_like_answer(self, answer):
        if answer is None:
            return True
        text = str(answer).strip().lower()
        if not text:
            return True
        action_prefixes = (
            "retrieve ",
            "retrieve[",
            "calculate ",
            "calculate[",
            "search ",
            "search[",
            "operate ",
            "operate[",
            "verify ",
            "verify[",
        )
        return text.startswith(action_prefixes)

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        if not isinstance(self.answer, str):
            self.answer = str(self.answer)
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (self.actual_step_n > self.max_actual_steps)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.actual_step_n = 1
        self.finished = False
        self.scratchpad: str = ''
        self.pre_ans = None
        self.pre_ans_all = []
        self.parse_failures = []
        self.empty_parse_streak = 0
        self.direct_answer_candidate = ""
        self.direct_answer_candidate_verification = None
        self.fallback_answer = ""
        self.fallback_rejected_reason = ""
        self.run_status = "running"
        self.verification_cache = {}

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)
