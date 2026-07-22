# TAPAS baseline on CRT-QA

This baseline runs the local `google/tapas-large-finetuned-wtq` checkpoint on
the existing CRT-QA answerable data. It performs inference only: CRT gold
answers are retained for evaluation but are never passed to the tokenizer or
model.

## Paths and environment

- Model: `/home/zhangyunhe/nas/model/tapas-large-finetuned-wtq`
- Input: `/home/zhangyunhe/nas/code/table/MACT/output/crt_answerable.jsonl`
- Conda environment: `mact`
- Outputs: `baselines/TAPAS/output/crt_tapas-large-wtq_<timestamp>/`

The model and tokenizer use `local_files_only=True`. The model is loaded from
`model.safetensors` without a `torch_dtype` override, so inference uses the
checkpoint/PyTorch default precision. The script does not download anything
and does not use `pytorch_model.bin` or `tf_model.h5`.

## Validate without loading model weights

Run this after `model.safetensors` has finished downloading. It validates all
728 inputs and their TAPAS tokenization, but does not load model weights or
initialize CUDA.

```bash
conda run -n mact python \
  /home/zhangyunhe/nas/code/table/MACT/baselines/TAPAS/run_tapas_crt.py \
  --validate_only
```

Run the no-weight unit tests (postprocessing, different-table padding, mock
logits, output schema, and resume validation):

```bash
conda run -n mact python -m unittest \
  /home/zhangyunhe/nas/code/table/MACT/baselines/TAPAS/test_run_tapas_crt.py
```

## Choose a GPU and run

`CUDA_VISIBLE_DEVICES` must be set explicitly. Inside the process, the selected
physical GPU is exposed as `cuda:0`.

Five-example smoke test:

```bash
CUDA_VISIBLE_DEVICES=2 \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/TAPAS/run_crt.sh --limit 5
```

Full 728-example run:

```bash
CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=2 \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/TAPAS/run_crt.sh
```

If the default precision does not fit with batch size 2, set `BATCH_SIZE=1`.
The baseline never changes model precision automatically.

## TAPAS answer conversion

TAPAS predicts selected cells plus one aggregation label. `pred_answer` stores
the final denotation used by the CRT evaluator:

- `NONE`: one selected cell is a string; multiple selected cells are a list.
- `COUNT`: number of selected cells.
- `SUM` and `AVERAGE`: numeric values parsed with TAPAS's own numeric parser,
  followed by the predicted aggregation.
- Empty selections or non-numeric aggregation inputs produce an empty
  prediction with the reason preserved in `tapas_metadata`.

The raw pipeline-style value such as `SUM > 87, 53, 69`, selected coordinates,
cells, aggregation, input length, and truncation diagnostics remain available
in `tapas_metadata`.

## Resume an interrupted run

Completed batches are appended to `results.jsonl` and flushed to disk. Resume
requires the exact existing run directory:

```bash
CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=1 \
RUN_DIR=/home/zhangyunhe/nas/code/table/MACT/baselines/TAPAS/output/crt_tapas-large-wtq_YYYYMMDD_HHMMSS \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/TAPAS/run_crt.sh --resume
```

The resume check requires saved result IDs to be an exact prefix of the current
input. It refuses mismatched, duplicate, or incomplete rows.

## Configuration overrides

- `BATCH_SIZE` (default `2`)
- `MODEL_PATH`
- `DATASET_PATH`
- `CRT_DATASET_PATH`
- `MAX_SOURCE_LENGTH` (default `1024`)
- `CELL_CLASSIFICATION_THRESHOLD` (default `0.5`)
- `RUN_DIR`

Additional arguments are forwarded to `run_tapas_crt.py`.

After inference, the wrapper only runs `scripts/evaluate_crt_by_type.py`. It
produces `crt_type_metrics.json`, `crt_type_metrics.md`, and
`crt_type_details.jsonl` beside `results.jsonl`. A full default run must contain
728 result rows, of which 726 have valid gold answers and 2 have empty golds.
