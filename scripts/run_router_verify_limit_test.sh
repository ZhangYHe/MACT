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
  --limit 1 \
  --plan_sample 1 \
  --code_sample 1 \
  --use_router \
  --use_verify_action \
  --use_repair \
  --output_path ../output/router_verify_limit_test.jsonl
