# Low-call GPT-5 reasoning baselines

This directory implements three API baselines over the 728 CRT-QA answerable
examples:

- Zero-shot CoT: choice 0 from the shared reasoning generation.
- Self-Consistency-3: answer voting across three reasoning choices.
- Single-pass Self-Correction-1: one compact check-and-revise request applied
  to choice 0.

The last method is intentionally **not** called Self-Refine-1. A standard
Self-Refine iteration separates feedback generation from refinement; this
baseline combines them to reduce API usage.

No few-shot examples, tools, dataset-specific hints, gold answers, or table
truncation are used. The shared prompt is:

```text
TABLE
{table}

Q: {question}
Reason in ≤3 short steps. End with `FINAL_ANSWER: <answer>`.
```

The correction turn is:

```text
Check the answer against the table and fix it if needed.
Give one short check, then `FINAL_ANSWER: <answer>`.
```

`max_tokens=2000` remains aligned with the main experiment so hidden reasoning
does not cause an empty length-limited response. The prompt, rather than a low
token ceiling, asks the model to keep visible reasoning short.

## 1. Check whether the endpoint supports `n=3`

This command sends exactly one API request with a tiny prompt, no retries, and
does not read CRT data:

```bash
python baselines/ReasoningLLM/test_n3_request.py \
  --model_config baselines/DirectLLM/gpt_5.yaml
```

The script prints the HTTP status, returned model, number of choices, compact
choice contents, finish reasons, and usage. It succeeds only when exactly three
choices are returned. It never prints the API key or request headers.

## 2. Run the baselines

If the compatibility check succeeds:

```bash
bash baselines/ReasoningLLM/run_reasoning_llm.sh \
  --multi_choice_mode n
```

If the endpoint does not support `n=3`:

```bash
bash baselines/ReasoningLLM/run_reasoning_llm.sh \
  --multi_choice_mode separate
```

The mode is required. The runner never changes from `n` to `separate`
automatically.

With `n` mode, each example uses one `n=3` CoT request and one correction
request: 1,456 HTTP requests and 2,912 completions for 728 examples, excluding
transport retries. In `separate` mode it uses 2,912 HTTP requests and the same
number of completions.

Use a unique directory when overriding defaults:

```bash
RUN_DIR=baselines/ReasoningLLM/output/my_run \
bash baselines/ReasoningLLM/run_reasoning_llm.sh \
  --multi_choice_mode n
```

## Checkpoint and resume

The append-only checkpoint is:

```text
RUN_DIR/checkpoint.jsonl
```

A successful CoT phase is saved before Self-Correction begins. If correction
fails or the run is interrupted after that checkpoint, resume only the missing
phase:

```bash
RUN_DIR=baselines/ReasoningLLM/output/my_run \
bash baselines/ReasoningLLM/run_reasoning_llm.sh \
  --multi_choice_mode n \
  --resume
```

Do not change the model, prompt parameters, dataset, or multi-choice mode when
resuming the same directory.

The checkpoint is the source of truth. The runner deterministically materializes:

```text
RUN_DIR/zero_shot_cot/results.jsonl
RUN_DIR/self_consistency_3/results.jsonl
RUN_DIR/single_pass_self_correction_1/results.jsonl
```

Each result retains the original dataset fields and exposes `pred_answer` in
the format expected by the existing evaluator. Missing or failed phases are
written with an empty prediction and `execute_status=fail`.

## Evaluation

Run the existing CRT evaluator separately for each result file:

```bash
conda run -n mact python scripts/evaluate_crt_by_type.py \
  --result_jsonl RUN_DIR/zero_shot_cot/results.jsonl

conda run -n mact python scripts/evaluate_crt_by_type.py \
  --result_jsonl RUN_DIR/self_consistency_3/results.jsonl

conda run -n mact python scripts/evaluate_crt_by_type.py \
  --result_jsonl RUN_DIR/single_pass_self_correction_1/results.jsonl
```

Evaluate only after all 728 rows report successful phases. The final comparison
should show both existing Direct GPT-5 results:

- Matched MACT parameters (`temperature=0.6`): Denotation EM 60.47%.
- Deterministic/default parameters (`temperature=0`): Denotation EM 62.81%.

Reporting both avoids selecting only the weaker Direct result.

## Tests

The test module uses mocked HTTP responses and does not require model access:

```bash
python baselines/ReasoningLLM/test_reasoning_llm_baseline.py
```

The implementation was intentionally delivered without running this command,
the compatibility request, the baseline, or the evaluator.

## 4. Repair technical failures and add Self-Refine-1

`self_refine_supplement.py` consumes a completed ReasoningLLM run without
modifying its checkpoint or original three result directories. It only repairs
choices that have empty content, a non-`stop` finish reason, or no extractable
answer. It never uses gold correctness to decide what to retry.

For the `gpt_5_reasoning_crt_answerable_07231634` run, the expected work is:

- Repair 165 invalid CoT choices across 85 examples, grouped into about 85 HTTP
  requests with `n` equal to the number of invalid choices for that example.
- Regenerate 67 single-pass corrections whose output is invalid or whose
  choice 0 changed.
- Run one feedback request and one refinement request for each of the 715
  examples with a recoverable source CoT.
- Keep the 13 prompt-level HTTP 400 content-filter examples failed without
  retrying or altering their tables.

This is about 1,582 HTTP requests in total: 152 for technical repair and 1,430
for Self-Refine-1, excluding transport and empty-length retries.

The separated prompts are:

```text
Check the answer against the table. Give brief feedback only.
```

```text
Revise using the feedback. Reason in ≤3 short steps. End with `FINAL_ANSWER: <answer>`.
```

Run it with:

```bash
bash baselines/ReasoningLLM/run_self_refine_supplement.sh \
  --source_run_dir \
  /home/zhangyunhe/nas/code/table/MACT/baselines/ReasoningLLM/output/gpt_5_reasoning_crt_answerable_07231634
```

Resume only missing or failed supplemental phases with:

```bash
bash baselines/ReasoningLLM/run_self_refine_supplement.sh \
  --source_run_dir \
  /home/zhangyunhe/nas/code/table/MACT/baselines/ReasoningLLM/output/gpt_5_reasoning_crt_answerable_07231634 \
  --resume
```

The source checkpoint is hashed per example when supplemental work begins.
Resume aborts if a source state or matched generation parameter has changed.
New files are written to:

```text
RUN_DIR/self_refine_supplement_checkpoint.jsonl
RUN_DIR/self_refine_supplement_summary.json
RUN_DIR/repaired_baselines/zero_shot_cot/results.jsonl
RUN_DIR/repaired_baselines/self_consistency_3/results.jsonl
RUN_DIR/repaired_baselines/single_pass_self_correction_1/results.jsonl
RUN_DIR/self_refine_1/results.jsonl
```

CoT repair uses `max_tokens=4000` directly because those choices already
failed at 2000. Correction, feedback, and refinement start at 2000 and retry
once at 4000 only for an empty `finish_reason=length` response. All other
sampling parameters remain matched to the source run.

Evaluate the four supplemental result files separately:

```bash
conda run -n mact python scripts/evaluate_crt_by_type.py \
  --result_jsonl RUN_DIR/repaired_baselines/zero_shot_cot/results.jsonl

conda run -n mact python scripts/evaluate_crt_by_type.py \
  --result_jsonl RUN_DIR/repaired_baselines/self_consistency_3/results.jsonl

conda run -n mact python scripts/evaluate_crt_by_type.py \
  --result_jsonl RUN_DIR/repaired_baselines/single_pass_self_correction_1/results.jsonl

conda run -n mact python scripts/evaluate_crt_by_type.py \
  --result_jsonl RUN_DIR/self_refine_1/results.jsonl
```

The supplemental mock tests are:

```bash
python baselines/ReasoningLLM/test_self_refine_supplement.py
```

As with the original delivery, API requests, baselines, evaluation, and tests
are left for the user to run.
