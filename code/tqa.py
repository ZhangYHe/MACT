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


import traceback
import json
import argparse
import os
import httpx
import sglang as sgl
from agents import ReactAgent
from dotenv import load_dotenv
from openai import OpenAI
from sglang.lang.chat_template import (ChatTemplate, get_chat_template,
                                       register_chat_template)
from transformers import AutoTokenizer
from utils import normalize_table_rows, summarize_react_trial, table2df
from utils import get_databench_table
from vllm import LLM


def write_to_file(path, agent, idx, new_table_dataset, given_plan):
    with open(path, "a") as f:
        agent.run(given_plan)
        pred_answer = agent.answer
        item = new_table_dataset[idx]
        item["pred_answer"] = pred_answer
        item["history"] = agent.scratchpad
        item["pred_answer_all"] = agent.pre_ans_all
        item["run_status"] = agent.run_status
        item["parse_failures"] = agent.parse_failures
        item["fallback_answer"] = agent.fallback_answer
        item["fallback_rejected_reason"] = agent.fallback_rejected_reason
        item["raw_pred_answer"] = agent.raw_pred_answer
        item["answer_contract"] = (
            agent._get_crt_answer_contract() if agent.task == "crt" else {}
        )
        item["postprocess_trace"] = agent.postprocess_trace
        item["candidate_source"] = agent.candidate_source
        item["verifier_attempts"] = agent.verifier_attempts
        item["verifier_rejections"] = agent.verifier_rejections
        item["question_profile"] = agent.question_profile
        item["direct_answer_candidate"] = agent.direct_answer_candidate
        item["direct_answer_candidate_verification"] = agent.direct_answer_candidate_verification
        item["table_diagnostics"] = agent.table_diagnostics
        item["tool_events"] = agent.tool_events
        # item["code_log"] = agent.generated_code
        # item["plan_log"] = agent.generated_plan
        f.write(json.dumps(item)+"\n")
    return agent

# ===================================================


def load_codellama_template(endpoint2):
    codellama_template = ChatTemplate(
        name="codellama",
        default_system_prompt=(
            "You are an intelligent programming assistant."
        ),
        role_prefix_and_suffix={
            "system": ("### System Promopt\n", "\n"),
            "user": ("### User Message\n", "\n"),
            "assistant": ("### Assistant", ""),
        }
    )
    register_chat_template(codellama_template)
    endpoint2.chat_template = get_chat_template("codellama")


def resolve_backend(backend, model_name):
    if backend != "auto":
        return backend
    return "openai" if "gpt" in model_name else "local"


def default_env_file():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, ".env")


def resolve_row_task(row, cli_task):
    if cli_task not in {"auto", "mixed"}:
        return cli_task
    row_task = str(row.get("task", "")).strip().lower()
    if row_task not in {"wtq", "crt", "tat", "scitab"}:
        raise ValueError(
            "--task auto requires each dataset row to include task as one of "
            "wtq, crt, tat, or scitab."
        )
    return row_task


def main(args):
    load_dotenv(args.env_file, override=True)
    plan_backend = resolve_backend(args.plan_backend, args.plan_model_name)
    code_backend = resolve_backend(args.code_backend, args.code_model_name)
    codeagent_endpoint = None
    client = None
    if plan_backend == "openai" or code_backend == "openai":
        client = OpenAI(
            api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=args.base_url or os.getenv("OPENAI_BASE_URL"),
            http_client=httpx.Client()
        )

    if plan_backend == "local":
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)
        model = LLM(model=args.model_path)
    else:
        model = None
        tokenizer = None

    if code_backend == "local":
        codeagent_endpoint = sgl.RuntimeEndpoint(
            f"http://localhost:{args.code_endpoint}")
        if "codellama" in args.code_model_name.lower():
            load_codellama_template(codeagent_endpoint)

    with open(args.dataset_path, "r") as f:
        table_dataset = [json.loads(line) for line in f]
    if args.limit is not None:
        table_dataset = table_dataset[:args.limit]

    trial = 0
    agent_cls = ReactAgent
    plan_model_name = args.plan_model_name.split("/")[-1].strip()
    code_model_name = args.code_model_name.split("/")[-1].strip()
    output_path = args.output_path or f"{args.task}_{plan_model_name}_{code_model_name}_{args.as_reward}_{args.plan_sample}_{args.code_sample}_direct_{args.direct_reasoning}_{args.answer_aggregate}.json"
    if args.debug_llm_io and not args.debug_log_path:
        args.debug_log_path = f"{output_path}.llm_debug.jsonl"
    agents = []
    for _, row in enumerate(table_dataset):
        row_task = resolve_row_task(row, args.task)
        table_diagnostics = []
        if row_task == "databench":
            databench_table = get_databench_table(args.table_dir, row["dataset"])
            table = databench_table[0]
            normalized_table = normalize_table_rows(
                databench_table[1], diagnostics=table_diagnostics)
            table_df = table2df(normalized_table)
            df_path = databench_table[2]
        else:
            table = normalize_table_rows(
                row["table_text"], diagnostics=table_diagnostics)
            table_df = table2df(table)
            df_path = None

        agents.append(agent_cls(
            question=row["question"] if "question" in list(row.keys()) else row["statement"],
            table=table,
            table_df=table_df,
            df_path=df_path,
            context=row["text"] if "text" in list(row.keys()) else "",
            key=row["answer"] if "answer" in list(row.keys()) else "none",
            answer="",
            max_steps=args.max_step,
            max_actual_steps=args.max_actual_step,
            plan_model_name=args.plan_model_name,
            code_model_name=args.code_model_name,
            model=model,
            tokenizer=tokenizer,
            task=row_task,
            codeagent_endpoint=codeagent_endpoint,
            as_reward=args.as_reward,
            plan_sample=args.plan_sample,
            code_sample=args.code_sample,
            use_pre_answer=args.use_pre_answer,
            answer_aggrement=args.answer_aggregate,
            direct_reasoning=args.direct_reasoning,
            long_table_op=args.long_table_op,
            debugging=args.debugging,
            client=client,
            plan_backend=plan_backend,
            code_backend=code_backend,
            code_as_observation=args.code_as_observation,
            without_tool=args.without_tool,
            use_router=args.use_router,
            use_verifier=args.use_verifier,
            use_repair=args.use_repair,
            use_code_repair=args.use_code_repair,
            disable_search=args.disable_search,
            disable_calculate=args.disable_calculate,
            disable_coding_agent=args.disable_coding_agent,
            log_router=args.log_router,
            example_id=row.get("id", ""),
            debug_llm_io=args.debug_llm_io,
            debug_full_prompt=args.debug_full_prompt,
            debug_log_path=args.debug_log_path,
            table_diagnostics=table_diagnostics,
            postprocess_pred_answer=args.postprocess_pred_answer,
            openai_max_tokens=args.openai_max_tokens,
            openai_temperature=args.openai_temperature,
        ))
    if args.debugging:
        agents = agents[0:1]
        for idx, agent in enumerate([a for a in agents]):
            print(idx)
            print(agent.question)
            print(agent.table_string)
            agent.run()
            print(f'Answer: {agent.key}, Pred: {agent.answer}')
            print(agent.scratchpad)
            trial += 1
            correct, incorrect, halted = summarize_react_trial(agents)
            print(f'Finished Trial {trial}, Correct: {len(correct)}, \
                    Incorrect: {len(incorrect)}, Halted: {len(halted)}')
    else:
        finished_agents = []
        for idx, agent in enumerate([a for a in agents]):
            try:
                finished_agent = write_to_file(
                    output_path, agent, idx, table_dataset, given_plan=None)
                finished_agents.append(finished_agent)
                trial += 1
                correct, incorrect, halted = summarize_react_trial(
                    finished_agents)
                print(
                    f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')
            except Exception as e:
                print(traceback.format_exc())
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan_model_name',
                        default="", help="name of the planning model.")
    parser.add_argument('--code_model_name',
                        default="", help="name of the coding model.")
    parser.add_argument('--plan_backend', default="auto",
                        choices=["auto", "openai", "local"],
                        help="backend for the planning model.")
    parser.add_argument('--code_backend', default="auto",
                        choices=["auto", "openai", "local"],
                        help="backend for the coding model.")
    parser.add_argument('--api_key', default="",
                        help="OpenAI-compatible API key. Falls back to OPENAI_API_KEY.")
    parser.add_argument('--base_url', default="",
                        help="OpenAI-compatible base URL. Falls back to OPENAI_BASE_URL.")
    parser.add_argument('--env_file', default=default_env_file(),
                        help="env file containing OPENAI_API_KEY and OPENAI_BASE_URL.")
    parser.add_argument('--cache_dir', default="",
                        help="cache dir to load a model from.")
    parser.add_argument('--model_path', type=str,
                        default="", help="model path to the planning model.")
    parser.add_argument('--dataset_path', type=str,
                        default="../datasets/wtq.jsonl", help="dataset path.")
    parser.add_argument('--table_dir', type=str,
                        default="../datasets/databench/data", help="databench table directory")
    parser.add_argument('--limit', type=int, default=None,
                        help="only run the first N dataset examples.")
    parser.add_argument('--output_path', type=str, default="",
                        help="output JSONL path.")
    parser.add_argument('--max_step', type=int, default=6,
                        help="maximum number for valid iterations.")
    parser.add_argument('--max_actual_step', type=int, default=6,
                        help="maximum number for all iterations.")
    parser.add_argument('--task', type=str, default="wtq",
                        choices=["wtq", "crt", "tat", "scitab", "databench", "auto", "mixed"])
    parser.add_argument('--as_reward', type=str, default="consistency",
                        choices=["consistency", "llm", "logp", "rollout", "combined"])
    parser.add_argument('--long_table_op', type=str, default="ignore",
                        choices=["code-agent", "ignore", "short-table"],
                        help="methods to shorten long table. default passing the whole table.")
    parser.add_argument('--plan_sample', type=int, default=5,
                        help="number of actions sampled from a planning model.")
    parser.add_argument('--code_sample', type=int, default=5,
                        help="numbers of trails for generating codes to address an action.")
    parser.add_argument('--use_pre_answer', type=bool, default=True,
                        help="whether use answers from the first iteration as final answers.")
    parser.add_argument('--answer_aggregate', type=float, default=1.,
                        help="agreement threshold for answer selection of use_pre_answer.")
    parser.add_argument('--direct_reasoning', action='store_true',
                        help="whether to use cot and symbolic reasoning directly or not.")
    parser.add_argument('--without_tool', action='store_true')
    parser.add_argument('--code_endpoint', default="11039",
                        help="coding agent port.")
    parser.add_argument('--debugging', action='store_true')
    parser.add_argument('--debug_llm_io', action='store_true',
                        help="write raw LLM call diagnostics as JSONL for debugging.")
    parser.add_argument('--debug_full_prompt', action='store_true',
                        help="include full prompts in --debug_llm_io JSONL logs.")
    parser.add_argument('--debug_log_path', default="",
                        help="path for --debug_llm_io JSONL logs. Defaults to output_path.llm_debug.jsonl.")
    parser.add_argument('--openai_max_tokens', type=int, default=2000,
                        help="max visible output tokens for OpenAI Chat Completions calls.")
    parser.add_argument('--openai_temperature', type=float, default=0.6,
                        help="temperature for OpenAI Chat Completions calls.")
    parser.add_argument('--code_as_observation', action='store_true',
                        help="only use code as the final observations or not.")
    parser.add_argument('--use_router', action='store_true',
                        help="enable question type routing before step-wise planning.")
    parser.add_argument('--use_verifier', action='store_true',
                        help="enable explicit Verify[claim] and final-answer verification.")
    parser.add_argument('--use_code_repair', action='store_true',
                        help="revise generated calculation code once when execution fails.")
    parser.add_argument('--use_repair', action='store_true',
                        help="deprecated alias for --use_code_repair.")
    parser.add_argument('--postprocess_pred_answer', action='store_true',
                        help="enable conservative final-answer postprocessing for evaluator-friendly formatting.")
    parser.add_argument('--disable_search', action='store_true',
                        help="disable Search actions for ablation.")
    parser.add_argument('--disable_calculate', action='store_true',
                        help="disable Calculate and Operate actions for ablation.")
    parser.add_argument('--disable_coding_agent', action='store_true',
                        help="disable LLM code generation tools for ablation.")
    parser.add_argument('--log_router', action='store_true', default=True,
                        help="print the question router profile when --use_router is enabled.")
    args = parser.parse_args()
    main(args)
