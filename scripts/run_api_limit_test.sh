#!/usr/bin/env bash

cd ../code

python tqa.py \
  --env_file ../.env \
  --plan_backend openai \
  --code_backend openai \
  --plan_model_name gpt-5.4 \
  --code_model_name gpt-5.4 \
  --dataset_path ../datasets_examples/tat.jsonl \
  --task wtq \
  --limit 2 \
  --plan_sample 1 \
  --code_sample 1 \
  --output_path ../output/tat_limit_test.jsonl
