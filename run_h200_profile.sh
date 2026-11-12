#!/usr/bin/env bash
# H200 NVL + Llama-3.1-8B architecture profile out to ~1M context.
# Every pass writes to its own directory under $OUT; merge afterwards.
set -u
cd /workspace/Frontier
PY=/venv/frontier-profiling/bin/python
OUT=/workspace/prof
M=llama3.1-8b-1m
COMMON="--disable_ray --yes --models $M --device h200 --num_gpus 1 --num_tensor_parallel_workers 1"
ATTN="$COMMON --attention_backend FLASHINFER"
export FRONTIER_MAX_PREFILL_CHUNK=8192 FRONTIER_FLASHINFER_WORKSPACE_GB=8
step() { echo "=== STEP $1 START $(date +%T)"; }
done_() { echo "=== STEP $1 DONE rc=$2 $(date +%T)"; }

step linear_cuda_event
$PY -m frontier.profiling.linear_op.main $COMMON --profile_method cuda_event --max_tokens 16384 --output_dir $OUT/linear_cuda_event; done_ linear_cuda_event $?
step linear_kernel_only
$PY -m frontier.profiling.linear_op.main $COMMON --profile_method record_function --max_tokens 16384 --output_dir $OUT/linear_kernel_only; done_ linear_kernel_only $?

step decode_1m
KV=$(seq -s' ' 131328 2048 1030000)
$PY -m frontier.profiling.attention.main $ATTN --profile_method cuda_event --max_seq_len 1048576 --max_model_len 1048576 \
  --min_batch_size 1 --max_batch_size 256 --profile_only_decode --decode_kv_cache_size_list $KV --output_dir $OUT/decode_1m; done_ decode_1m $?

step standard_128k
export FRONTIER_EXTRA_SEQ_LENGTHS=$(seq -s, 65792 256 131072)
$PY -m frontier.profiling.attention.main $ATTN --profile_method cuda_event --max_seq_len 131072 --max_model_len 131072 \
  --min_batch_size 1 --max_batch_size 256 --enable_chunked_prefill_grid_search --output_dir $OUT/standard_128k; done_ standard_128k $?
unset FRONTIER_EXTRA_SEQ_LENGTHS

for C in 8192 4096 1024 256; do
  step prefill_chunk_$C
  $PY -m frontier.profiling.attention.main $ATTN --profile_method cuda_event --max_seq_len 1048576 --max_model_len 1048576 \
    --min_batch_size 1 --max_batch_size 1 --profile_only_prefill --enable_chunked_prefill_grid_search --fixed_chunked_prefill_size $C \
    --output_dir $OUT/prefill_chunk_$C; done_ prefill_chunk_$C $?
done

step decode_kernel_only
export FRONTIER_EXTRA_SEQ_LENGTHS=$(seq -s, 65792 256 131072),$(seq -s, 131328 2048 1030000)
$PY -m frontier.profiling.attention.main $ATTN --profile_method record_function --profile_only_decode --max_seq_len 1048576 --max_model_len 1048576 \
  --min_batch_size 1 --max_batch_size 256 --output_dir $OUT/decode_kernel_only; done_ decode_kernel_only $?
unset FRONTIER_EXTRA_SEQ_LENGTHS

for PKV in 0 2048 8192 32768 65536 98304 120000 262144 524288 786432; do
  step mixed_$PKV
  $PY -m frontier.profiling.attention.main $ATTN --profile_method cuda_event --max_seq_len 64 --max_model_len 1048576 --enable_true_mixed \
    --true_mixed_prefill_batch_sizes 1 2 3 --true_mixed_prefill_chunk_sizes 64 128 256 512 1024 2048 4096 8192 \
    --true_mixed_decode_batch_sizes 1 2 4 8 12 16 24 32 48 64 96 125 \
    --true_mixed_decode_kv_cache_sizes 128 256 512 1024 2048 4096 8192 12288 16384 24576 32768 49152 65536 98304 131071 262144 524288 \
    --true_mixed_prefill_kv_cache_size $PKV --output_dir $OUT/mixed_$PKV; done_ mixed_$PKV $?
done
echo "=== ALL DONE $(date +%T)"
