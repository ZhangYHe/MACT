# TableLlama baseline on CRT-QA

This baseline runs the local `osunlp/TableLlama` 7B checkpoint on the 728
CRT-QA answerable examples. It performs inference only: CRT training data,
gold answers, and CRT annotations are never added to the model prompt.

The existing `mact` environment is used without installing or changing any
package. vLLM loads the local PyTorch shards with `dtype=auto`, which resolves
to the checkpoint's configured BF16. All model access is offline.

## CPU-only validation

This checks the download, input schema, and all 728 prompts without loading
model weights or using CUDA:

```bash
conda run -n mact python \
  /home/zhangyunhe/nas/code/table/MACT/baselines/TableLlama/run_tablellama_crt.py \
  --validate_only
```

The current CRT input has a maximum prompt length below 3000 tokens, so the
4096-token inference setting keeps every table row. Row dropping remains as a
safety boundary for other inputs.

## RTX 3090 execution

Use one otherwise-free 24 GB RTX 3090 with TP=1 and batch 16. Start with a
representative smoke test:

```bash
CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=16 \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/TableLlama/run_crt.sh --limit 16
```

Full run:

```bash
CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=16 \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/TableLlama/run_crt.sh
```

If generation OOMs, retry with batch 8, then 4. If model initialization fails
on an otherwise-free 3090, two GPUs can split the checkpoint:

```bash
CUDA_VISIBLE_DEVICES=2,3 TENSOR_PARALLEL_SIZE=2 BATCH_SIZE=16 \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/TableLlama/run_crt.sh
```

Tensor parallel 2 is a capacity fallback and is not guaranteed to be faster
than one 3090. Useful overrides are `BATCH_SIZE`, `MAX_MODEL_LENGTH`,
`MAX_INPUT_TOKENS`, `MAX_NEW_TOKENS`, `GPU_MEMORY_UTILIZATION`,
`TENSOR_PARALLEL_SIZE`, `ENFORCE_EAGER`, `MODEL_PATH`, `DATASET_PATH`,
`CRT_DATASET_PATH`, and `RUN_DIR`.

## Resume and outputs

Resume only after the old process has stopped, using the exact existing run
directory:

```bash
CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=8 \
RUN_DIR=/absolute/path/to/the/existing/run \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/TableLlama/run_crt.sh --resume
```

Each run writes to
`output/crt_tablellama-7b_<timestamp>/`. It contains `results.jsonl`,
`run_config.json`, `run.log`, `crt_type_metrics.json`, `crt_type_metrics.md`,
and `crt_type_details.jsonl`. Only `scripts/evaluate_crt_by_type.py` is run.
A full run must contain 728 results, 726 evaluated examples, and 2 invalid
empty gold answers.
