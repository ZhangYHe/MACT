# ProTrix baseline on CRT-QA

This baseline runs the local `pkupie/ProTrix` 7B checkpoint on the 728 CRT-QA
answerable examples without CRT training data or gold-answer input. It follows
the official Plan-then-Reason structure: generate a plan and SQL, execute safe
read-only SQL in memory, then generate the final answer from the execution
result.

The existing `mact` environment is used without installing or changing any
package. vLLM loads the checkpoint with `dtype=auto`, which resolves to the
model's configured BF16. The scripts are fully offline after download.

## Validation

After all three safetensors shards finish downloading:

```bash
conda run -n mact python \
  /home/zhangyunhe/nas/code/table/MACT/baselines/ProTrix/run_protrix_crt.py \
  --validate_only
```

## RTX 3090 parameter recommendations

This server has 24 GB RTX 3090 GPUs. The recommended configuration for a
comparable full baseline is **one otherwise-free 3090**, tensor parallel 1,
`BATCH_SIZE=16`, `GPU_MEMORY_UTILIZATION=0.90`, and
`MAX_NEW_TOKENS=1024`. The 13.5 GB BF16 checkpoint and a 4096-token KV cache
should fit, but select a card with about 22 GB free before launch. Start with a
16-example smoke test so that throughput and peak memory are representative:

```bash
CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=16 \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/ProTrix/run_crt.sh --limit 16
```

Full run:

```bash
CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=16 \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/ProTrix/run_crt.sh
```

If this OOMs during generation, try `BATCH_SIZE=8`, then 4. If model
initialization itself cannot fit, make sure the selected GPU is not occupied;
only then use two 3090s:

```bash
CUDA_VISIBLE_DEVICES=2,3 TENSOR_PARALLEL_SIZE=2 BATCH_SIZE=16 \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/ProTrix/run_crt.sh
```

For this server, vLLM's P2P test for the tested GPU pair failed and it fell
back to NCCL with custom all-reduce disabled. Consequently tensor parallel 2
splits memory but is not guaranteed to be faster than one 3090. Use TP=2 for
capacity/stability, not as the first throughput optimization. With TP=2,
`BATCH_SIZE=16` is the recommended starting point; test 24 before using it for
a full run. Do not start at 32 because long 3072-token prompts can exhaust the
KV cache and trigger recomputation.

`MAX_NEW_TOKENS=1024` is the accuracy-preserving setting used by default.
Some ProTrix responses run to this limit (including occasional long whitespace
tails), so individual batches can take much longer than their neighbors. For
exploration only, `MAX_NEW_TOKENS=768` is a balanced speed setting and 512 is a
faster setting; both may truncate the plan, SQL, or final answer and should not
be mixed with the official baseline result. Keep `MAX_INPUT_TOKENS=3072`,
`MAX_MODEL_LENGTH=4096`, and `ENFORCE_EAGER=0`. Enable eager mode only to work
around CUDA-graph memory problems, since it normally reduces throughput.

Increasing outer `BATCH_SIZE` increases the number of requests vLLM can
schedule concurrently. It does not change model precision or allocate model
weights again. vLLM reserves most of `GPU_MEMORY_UTILIZATION` for weights and
KV cache up front, so high memory use (around 22--23 GB at 0.90) is expected and
is not by itself an OOM.

Resume an interrupted run by setting the exact `RUN_DIR` and adding `--resume`.
The wrapper only runs `scripts/evaluate_crt_by_type.py`; a full run must contain
728 results, 726 evaluated examples, and 2 invalid empty gold answers.

Useful overrides are `BATCH_SIZE`, `MAX_INPUT_TOKENS`, `MAX_NEW_TOKENS`,
`GPU_MEMORY_UTILIZATION`, `TENSOR_PARALLEL_SIZE`, `ENFORCE_EAGER`, `MODEL_PATH`,
`DATASET_PATH`, `CRT_DATASET_PATH`, and `RUN_DIR`.

Example resume with a larger batch (run this only after the old process has
stopped):

```bash
CUDA_VISIBLE_DEVICES=2,3 TENSOR_PARALLEL_SIZE=2 BATCH_SIZE=16 \
RUN_DIR=/absolute/path/to/the/existing/run \
bash /home/zhangyunhe/nas/code/table/MACT/baselines/ProTrix/run_crt.sh --resume
```
