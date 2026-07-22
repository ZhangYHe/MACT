# OmniTab baseline on CRT-QA

This baseline runs the local `neulab/omnitab-large-finetuned-wtq` checkpoint
on the 728 CRT-QA answerable examples. It performs inference only and never
passes gold answers to the tokenizer or model.

The official checkpoint uses `pytorch_model.bin`. It is loaded locally without
a `torch_dtype` override, so its configured/PyTorch default precision is used
and the actual dtype is written to `run_config.json`. No package is installed
into the existing `mact` environment.

## Validation and execution

After the model download finishes, validate without loading weights or CUDA:

```bash
conda run -n mact python \
  /home/zhangyunhe/nas/code/table/MACT/baselines/OmniTab/run_omnitab_crt.py \
  --validate_only
```

## RTX 3090 parameter recommendations

OmniTab is a BART-large encoder-decoder checkpoint (about 776 MB on disk) and
uses PyTorch FP32 according to its config. On a 24 GB RTX 3090, start with
`BATCH_SIZE=16`, `MAX_SOURCE_LENGTH=1024`, and `MAX_NEW_TOKENS=64`. A
representative smoke test is:

```bash
CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=16 \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/OmniTab/run_crt.sh --limit 32
```

Full run:

```bash
CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=16 \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/OmniTab/run_crt.sh
```

If generation OOMs, retry with batch 8, then 4; batch 1--2 should only be
needed when the GPU is shared with another large process. Batch 24 can be
tested after the batch-16 smoke test, but 16 is the recommended full-run value
because CRT examples reach the full 1024-token encoder length. Keep source
length 1024 and generation length 64 for comparable results.

This implementation is single-GPU. Setting `CUDA_VISIBLE_DEVICES=2,3` does not
enable data or tensor parallelism: the script still uses only logical
`cuda:0`, so exposing a second GPU provides no speedup. Run one baseline per
free GPU instead if independent jobs are desired.

Resume after an OOM or interruption with the same output directory:

```bash
CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=8 \
RUN_DIR=/absolute/path/to/the/existing/run \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/OmniTab/run_crt.sh --resume
```

The wrapper only runs `scripts/evaluate_crt_by_type.py` and requires 728
results, 726 evaluated examples, and 2 invalid empty gold answers.
