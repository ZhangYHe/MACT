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

import re
import string
from collections import Counter, OrderedDict, defaultdict

import pandas as pd
from fewshots_table import (DEMO_CRT, DEMO_CRT_DIRECT, DEMO_SCITAB,
                            DEMO_SCITAB_DIRECT, DEMO_TAT, DEMO_TAT_DIRECT,
                            DEMO_WTQ, DEMO_WTQ_DIRECT,
                            NUMERICAL_OPERATION_EXAMPLE,
                            TABLE_OPERATION_EXAMPLE, DEMO_DATABENCH,
                            NUMERICAL_OPERATION_EXAMPLE_LONG_TABLE, GLOBAL_PLAN_EXAMPLES,
                            NUMERICAL_OPERATION_EXAMPLE_LONG_TABLE_GLOBAL)
from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer
from llm import OpenSourceLLM
from prompts_table import (DIRECT_AGENT, NUMERICAL_OPERATION_PROMPT,
                           TABLE_OPERATION_PROMPT, react_agent_prompt_crt,
                           react_agent_prompt_scitab, react_agent_prompt_tat,
                           react_agent_prompt_wtq, NUMERICAL_OPERATION_PROMPT_LONG_TABLE,
                           NUMERICAL_OPERATION_PROMPT_LONG_TABLE_GLOBAL,
                           react_agent_prompt_databench, global_plan_prompt)
from sglang import assistant, function, gen, user
from tot import llm_reward, vote_prompt_as
from utils import (extract_from_outputs, parse_action, table2df,
                   table_linear)

all_input_token, all_output_token = 0, 0


def get_completion(prompt, client, n, model):
    global all_input_token, all_output_token
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.6,
        max_tokens=400,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        n=n,
        stop=None
    )
    input_token_num = response.usage.prompt_tokens
    output_token_num = response.usage.completion_tokens
    all_input_token += input_token_num
    all_output_token += output_token_num
    # print(all_input_token, all_output_token)
    return [item.message.content for item in response.choices]


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
                 code_backend='auto'
                 ) -> None:

        vllm = model
        self.plan_backend = plan_backend
        self.code_backend = code_backend
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
        self.docstore = DocstoreExplorer(Wikipedia())  # Search
        self.answer_aggrement = answer_aggrement
        self.direct_reasoning = direct_reasoning
        self.without_tool = without_tool
        self.long_table_op = long_table_op
        self.code_as_observation = code_as_observation
        self.llm_sampled = []
        self.code_sampled = []
        self.direct_sampled = []

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
            try:
                result = self.numerical_tool(
                    eqution, recent_table_df, self.df_path, global_planning=False)
            except:
                pass
            return result

    def code_extract_calculator(self, code_strings, table_df, original_df):
        result = ""
        rows = []
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
            except:
                # print(e)
                pass
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
            return result, rows, None, None
        else:
            current_error = None
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
                executable_code = None
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
        max_attempt = self.code_sample
        results, generated_code = [], []
        results2df = defaultdict(list)
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
                result, rows = self.code_extract_calculator(
                    code_strings, table_df, original_df)
                if result != "" and rows != []:
                    try:
                        result = result.strip()
                        results2df[result].append(table2df(rows))
                    except:
                        pass
                results.append(result)

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
            current_thought, current_action, current_observation = "", "", ""
            if instance:
                instance_ = [line for line in instance.split(
                    "\n") if line.strip() != ""]
            try:
                current_thought = [
                    line for line in instance_ if f"Thought {self.step_n}:" in line][0]
                current_action = [
                    line for line in instance_ if f"Action {self.step_n}:" in line][0]
                current_observation_start_id = [i for i, line in enumerate(
                    instance_) if f"Observation {self.step_n}:" in line]
                current_observation_end_id = [i for i, line in enumerate(
                    instance_) if f"Thought {self.step_n+1}:" in line]
                current_observation = "\n".join(
                    instance_[current_observation_start_id[0]:current_observation_end_id[0]])
            except:
                pass
            return current_thought, current_action, current_observation

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
            if num_most_common > threshold:
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

    def run(self, reset=True, given_plan=None) -> None:
        if reset:
            self.__reset_agent()
        if self.task == "databench":
            if not self.is_finished():
                self.global_planning(given_plan)
        while not self.is_halted() and not self.is_finished():
            # if global planning fail, try step-wise planning
            self.step()

        if not self.answer:
            if self.use_pre_answer:
                try:
                    self.answer = Counter(
                        self.pre_ans_all).most_common(1)[0][0]
                except:
                    # direct prompting
                    self.answer = self.get_quick_answer()
            else:
                # direct prompting
                self.answer = self.get_quick_answer()

    def step(self) -> None:
        if self.direct_reasoning:
            if self.plan_backend == "openai":
                llm_sampled = self.prompt_agent_gpt(mode="text")
            else:
                llm_sampled = self.prompt_agent(mode="text")
            llm_sampled_ = [self.get_answer_from_llm(
                item) for item in llm_sampled]
            prompt = self.code_prompt.format(
                examples=self.code_examples, table=self.table_df, question=self.question, context=self.context)
            if self.code_backend == "openai":
                code_sampled = self.prompt_agent_gpt_coder(prompt)
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
                sampled = self.prompt_agent_gpt()
            else:
                sampled = self.prompt_agent(mode="both")
            self.actual_step_n += 1
            thought, action, observation, all_observations = self.as_reward_fn(
                sampled)
            if self.use_pre_answer and self.pre_ans:
                self.finished = True
                self.answer = self.pre_ans
            else:
                if thought != "" and action != "":
                    if "Finish" not in action:
                        action_type, argument = parse_action(action)
                        if action_type == "Calculate":
                            recent_table_df = self.table_dfs[-1]
                            new_ob = self.calculator_tool(
                                argument, recent_table_df=recent_table_df)
                            if not isinstance(new_ob, list):
                                if new_ob != "":
                                    observation = f"Observation {self.step_n}: {new_ob}"
                            else:
                                # majority voting among tool results and llm results
                                if new_ob != []:
                                    new_ob = [
                                        f'Observation {self.step_n}: {item}' for item in new_ob]
                                new_ob += all_observations
                                observation = Counter(
                                    new_ob).most_common(1)[0][0]

                        elif action_type == "Retrieve":
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
                            if self.without_tool:
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
                            recent_table_df = self.table_dfs[-1]
                            new_ob = self.calculator_tool(
                                argument, recent_table_df=recent_table_df)
                            if new_ob != "":
                                observation = f"Observation {self.step_n}: {new_ob}"

                        if observation != "":
                            self.scratchpad += thought + "\n"
                            self.scratchpad += action + "\n"
                            self.scratchpad += observation + "\n"
                            self.step_n += 1

                    else:
                        # finish in the action
                        action_type, argument = parse_action(action)
                        self.answer = argument
                        self.scratchpad += thought + "\n"
                        self.scratchpad += action + "\n"
                        self.finished = True

                else:
                    # resample
                    pass
                print("==============current step===========")
                print(self.scratchpad)

    def prompt_agent_gpt(self, mode="both") -> str:
        prompt = self._build_agent_prompt(mode=mode)
        preds = get_completion(
            prompt, client=self.client, n=self.plan_sample, model=self.plan_model_name)
        return preds

    def prompt_agent_gpt_coder(self, prompt) -> str:
        preds = get_completion(
            prompt, client=self.client, n=self.code_sample, model=self.code_model_name)
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
                prompt, client=self.client, n=self.plan_sample, model=self.plan_model_name)
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
                prompt, client=self.client, n=1, model=self.plan_model_name)
        return self.llm(prompt, num_return_sequences=1, return_prob=False)

    def _build_agent_prompt(self, mode="both") -> str:
        if mode == "text":
            return self.text_prompt.format(
                examples=self.text_examples,
                table=self.table_string,
                context=self.context,
                question=self.question)
        elif mode == "both":
            return self.agent_prompt.format(
                examples=self.react_examples,
                table=self.table_string,
                context=self.context,
                question=self.question,
                scratchpad=self.scratchpad)

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
