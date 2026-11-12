#!/usr/bin/env bash
# Extra chunked-prefill passes at the chunk sizes vLLM actually runs (a 2048-token
# budget minus the decodes sharing the step), out to ~1M cached tokens.
set -u
cd /workspace/Frontier
PY=/venv/frontier-profiling/bin/python
OUT=/workspace/prof
M=llama3.1-8b-1m
export FRONTIER_MAX_PREFILL_CHUNK=8192 FRONTIER_FLASHINFER_WORKSPACE_GB=8
for C in 2048 1792 1536 512; do
  echo "=== STEP prefill_chunk_$C START $(date +%T)"
  $PY -m frontier.profiling.attention.main --disable_ray --yes --models $M --device h200 --num_gpus 1 --num_tensor_parallel_workers 1 \
    --attention_backend FLASHINFER --profile_method cuda_event --max_seq_len 1048576 --max_model_len 1048576 \
    --min_batch_size 1 --max_batch_size 1 --profile_only_prefill --enable_chunked_prefill_grid_search --fixed_chunked_prefill_size $C \
    --output_dir $OUT/prefill_chunk_$C
  echo "=== STEP prefill_chunk_$C DONE rc=$? $(date +%T)"
done
echo "=== ALL DONE $(date +%T)"
