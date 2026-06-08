# Custom API/Base URL/Model and Dataset Limit Plan

## Summary

The project now supports OpenAI-compatible API calls with a custom `api_key`, `base_url`, and model names supplied through command-line arguments. It also supports running only the first N examples of a dataset through `--limit`.

## Implemented Changes

- Added explicit model backends in `code/tqa.py`:
  - `--plan_backend auto|openai|local`
  - `--code_backend auto|openai|local`
- Added OpenAI-compatible API options:
  - `--api_key`, falling back to `OPENAI_API_KEY`
  - `--base_url`, falling back to `OPENAI_BASE_URL`
- Kept custom model names on the existing arguments:
  - `--plan_model_name`
  - `--code_model_name`
- Added dataset truncation:
  - `--limit N` runs only the first N examples.
- Added output override:
  - `--output_path path/to/output.jsonl`
- Updated `code/agents.py` so API calls use the selected custom model names instead of hard-coded Azure/GPT behavior.

## User Work

Activate the existing conda environment before running code:

```bash
conda activate mact
```

Set API credentials:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="your_base_url"
```

Run any first-N test, for example the first 10 TAT examples:

```bash
cd /home/zhangyunhe/nas/code/table/MACT/code

python tqa.py \
  --plan_backend openai \
  --code_backend openai \
  --plan_model_name "your_plan_model" \
  --code_model_name "your_code_model" \
  --dataset_path ../datasets_examples/tat.jsonl \
  --task tat \
  --limit 10 \
  --plan_sample 1 \
  --code_sample 1 \
  --output_path ../tat_top10_test.jsonl
```

Change `--limit 10` to any number you want.

## Verification

```bash
conda activate mact
cd /home/zhangyunhe/nas/code/table/MACT
python -m py_compile code/agents.py code/tqa.py
```
