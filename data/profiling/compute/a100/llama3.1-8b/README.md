# A100 80GB PCIe, Llama-3.1-8B (llama3.1-8b), TP1

Collected 2026-11-04 on a Vast.ai instance (2x NVIDIA A100 80GB PCIe, driver
595.99.02, CUDA 12.8, torch 2.8.0, vLLM 0.10.2, FlashInfer 0.3.1), from this
repository's `environment_profiling.yml`. Every table is timed with CUDA
events (`cuda_event`) except the `*_kernel_only` tables (`record_function`).

| File | Rows | Contents |
|---|---|---|
| `attention.csv` | 59,677 | 37,760 standard rows (decode, prefill and chunked prefill, context to 131,072 every 256 tokens above 4k, batch 1-256 where the KV cache fits, prefill chunks up to 8,192 tokens on cached prefixes up to 131k) plus 21,917 true mixed prefill+decode rows (7 passes with cached prefix 0, 2k, 8k, 32k, 64k, 96k, 120k) |
| `attention_kernel_only.csv` | | decode attention to 131k, for CUDA-graph decode |
| `linear_op.csv`, `linear_op_kernel_only.csv` | | linear operators up to 16,384 tokens per step |
| `kernel_gap.csv` | 2 | 1.55 us between eager kernels, 0.41 us under CUDA graphs |

## How it was produced

```bash
export FRONTIER_MAX_PREFILL_CHUNK=8192 FRONTIER_FLASHINFER_WORKSPACE_GB=8 \
       FRONTIER_EXTRA_SEQ_LENGTHS=$(seq -s, 65792 256 131072)
python -m frontier.profiling.attention.main --disable_ray --yes --models llama3.1-8b \
  --device a100 --num_gpus 2 --num_tensor_parallel_workers 1 --attention_backend FLASHINFER \
  --profile_method cuda_event --max_seq_len 131072 --max_model_len 131072 \
  --min_batch_size 1 --max_batch_size 256 --enable_chunked_prefill_grid_search
# true mixed passes, one per cached-prefix length PKV, merged into attention.csv:
python -m frontier.profiling.attention.main ... --max_seq_len 64 --enable_true_mixed \
  --true_mixed_prefill_batch_sizes 1 2 3 \
  --true_mixed_prefill_chunk_sizes 64 128 256 512 1024 2048 4096 8192 \
  --true_mixed_decode_batch_sizes 1 2 4 8 12 16 24 32 48 64 96 125 \
  --true_mixed_decode_kv_cache_sizes 128 256 512 1024 2048 4096 8192 12288 16384 24576 32768 49152 65536 98304 131071 \
  --true_mixed_prefill_kv_cache_size $PKV
python -m frontier.profiling.linear_op.main ... --profile_method cuda_event --max_tokens 16384
python -m frontier.profiling.linear_op.main ... --profile_method record_function --max_tokens 16384
python -m frontier.profiling.attention.main ... --profile_method record_function --profile_only_decode \
  --max_seq_len 131072 --max_model_len 131072 --max_batch_size 256
python -m frontier.profiling.kernel_gap.main --device a100 --model llama3.1-8b \
  --num_tensor_parallel_workers 1 --num_tokens 16 256 2048 \
  --vllm_engine_args engine_args.json --vllm_attention_backend FLASHINFER
```

No CPU-overhead tables: they need Frontier's instrumented vLLM, which is not
public. Run with `--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling`.

## Simulator settings that match real vLLM

vLLM 0.10.2 V1, `VLLM_ATTENTION_BACKEND=FLASHINFER`, `--max-num-batched-tokens 2048
--max-num-seqs 256 --max-model-len 131072 --no-enable-prefix-caching`, which gives a
453,088-token KV cache and piecewise CUDA graphs. In Frontier:

```
--replica_config_model_name llama3.1-8b --replica_config_device a100
--replica_config_network_device a100_dgx --vllm_v1_scheduler_config_max_tokens_in_batch 2048
--vllm_v1_scheduler_config_batch_size_cap 256 --vllm_v1_scheduler_config_max_model_len 131072
--vllm_v1_scheduler_config_num_blocks_mode explicit --vllm_v1_scheduler_config_num_blocks 28318
--decode_cuda_graph_mode piecewise
--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 131072
--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 2048
--random_forrest_execution_time_predictor_config_prediction_max_batch_size 256
```

## Validation against real vLLM (same machine, FCFS, `frontier/rl/vllm_replay.py`)

| Slice | Metric | vLLM | Frontier |
|---|---|---|---|
| Azure conv, 4 qps, 300 req | mean / p99 TTFT | 0.200 / 0.752 s | 0.198 / 0.754 s |
| | mean e2e, mean TBT | 4.87 s, 23.6 ms | 4.35 s, 20.9 ms |
| Azure conv, 8 qps, 400 req | mean TTFT, mean e2e, mean TBT | 0.61 s, 14.6 s, 81.0 ms | 0.51 s, 12.7 s, 70.6 ms |
| Mooncake long context (8k-119k prompts), 0.5 qps, 80 req | median TTFT, mean e2e, mean TBT | 57.0 s, 137 s, 180 ms | 50.4 s, 158 s, 191 ms |

Per-request e2e correlation 0.98-0.997, median per-request error 6-12%. Decode
steps run about 11-13% fast on chat traffic, about 2.7 ms per step, consistent
with the unmodeled CPU overhead.
