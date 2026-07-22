# TAPEX baseline on CRT-QA

This baseline runs the local `microsoft/tapex-large-finetuned-wtq` checkpoint
on the existing CRT-QA answerable data. It performs inference only: CRT answers
are retained for evaluation but are never passed to the tokenizer or model.

## Paths and environment

- Model: `/home/zhangyunhe/nas/model/tapex-large-finetuned-wtq`
- Input: `/home/zhangyunhe/nas/code/table/MACT/output/crt_answerable.jsonl`
- Conda environment: `mact`
- Outputs: `baselines/TAPEX/output/crt_tapex-large-wtq_<timestamp>/`

The model and tokenizer are loaded with `local_files_only=True`, and the model
is loaded from `model.safetensors`. No `torch_dtype` override or manual dtype
conversion is applied, so inference uses the checkpoint/PyTorch default
precision. The actual loaded dtype is recorded in `run_config.json`. The script
does not download anything.

## Validate without loading the model

This checks the local model files, all input rows, IDs, table shapes, and
DataFrame conversion. It does not initialize CUDA or load model weights.

```bash
conda run -n mact python \
  /home/zhangyunhe/nas/code/table/MACT/baselines/TAPEX/run_tapex_crt.py \
  --validate_only
```

## Choose a GPU and run

`CUDA_VISIBLE_DEVICES` must be set explicitly. Inside the process, the selected
physical GPU is exposed as `cuda:0`.

Five-example smoke test:

```bash
CUDA_VISIBLE_DEVICES=2 \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/TAPEX/run_crt.sh --limit 5
```

Full 728-example run:

```bash
CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=4 \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/TAPEX/run_crt.sh
```

If batch size 4 does not fit the selected GPU, use `BATCH_SIZE=2` or
`BATCH_SIZE=1`. The baseline does not automatically lower model precision.

## Resume an interrupted run

Completed batches are appended to `results.jsonl` and flushed to disk. Resume
requires the exact existing run directory:

```bash
CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=2 \
RUN_DIR=/home/zhangyunhe/nas/code/table/MACT/baselines/TAPEX/output/crt_tapex-large-wtq_YYYYMMDD_HHMMSS \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/TAPEX/run_crt.sh --resume
```

The resume check requires the saved result IDs to be an exact prefix of the
current input. It refuses mismatched, duplicate, or incomplete rows.

## Configuration overrides

The wrapper accepts the following environment variables:

- `BATCH_SIZE` (default `4`)
- `MODEL_PATH`
- `DATASET_PATH`
- `CRT_DATASET_PATH`
- `MAX_SOURCE_LENGTH` (default `1024`)
- `MAX_NEW_TOKENS` (default `64`)
- `SEED` (default `42`; TAPEX row truncation is seeded independently per ID)
- `RUN_DIR`

Additional arguments are forwarded to `run_tapex_crt.py`.

After inference, the wrapper only runs `scripts/evaluate_crt_by_type.py`. It
produces `crt_type_metrics.json`, `crt_type_metrics.md`, and
`crt_type_details.jsonl` beside `results.jsonl`. A full default run must contain
728 result rows, of which 726 have valid gold answers and 2 have empty golds.
