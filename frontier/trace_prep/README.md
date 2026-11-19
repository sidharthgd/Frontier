# Session traces for prefix caching

`python -m frontier.trace_prep` turns chat and agentic traces into **session
traces**: one row per LLM call, with the session it belongs to and the chained
hash ids of its prompt's 16-token KV blocks, which is what Frontier's prefix
cache matches (`--vllm_v1_scheduler_config_enable_prefix_caching`, vLLM V1 and
SGLang schedulers). It also samples training workloads from them.

```
arrived_at,completed_at,session_id,step,num_prefill_tokens,num_decode_tokens,block_hash_ids,ids_cover_output
0.000,4.787,0,0,2645,104,0-171,1
4.788,6.630,0,1,2859,43,0-180,1
6.632,15.963,0,2,3478,365,0-239,1
```

`block_hash_ids` name the call's prompt blocks and then the full blocks its
output completes ((prompt + output) // 16 ids, `ids_cover_output`). Frontier
caches the output's blocks under them as they are decoded, as vLLM caches
generated tokens, so the session's next call, whose prompt holds this
output, hits them too.

Sampled workloads add `think_time` (see Training workloads), which Frontier's
trace replay uses to hold each session's calls until the previous one completes.

`block_hash_ids` are runs of consecutive ids (`a-b|c`,
`frontier/utils/block_hash_ids.py`). Agentic prompts are ~10k blocks, but a
prompt is a handful of runs, which makes these files ~1000x smaller than
listing ids. Trace replay reads both forms, and `Request` keeps the ids as an
int64 array. Ids are chained, so two prompts share a k-block prefix exactly
when their first k ids agree. Blocks of 16·m tokens are `ids[m-1::m]`.

## Sources

| Command | Source | Block ids | Default `--max-tokens` |
|---|---|---|---|
| `freeinference` | [FreeInference agentic trace](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace) `release/<week>/traces/*.jsonl.zst` (CC BY 4.0) | Released content hashes (`block_ids`, 16-token o200k blocks), namespaced per week. Subagent sessions become their own sessions. | 131072 |
| `tracelab` | [TraceLab](https://github.com/uw-syfi/TraceLab) release `syfi_coding_trace.jsonl.gz` (CC BY 4.0) | **Synthesized** from the provider's cache read (`prefix_tokens`): a round shares `prefix_tokens // 16` blocks with the previous round of its session (a session's first round: with the user's latest earlier round). | 131072 |
| `sharegpt` | `ShareGPT_V3_unfiltered_cleaned_split.json` (needs `tokenizers`) | Real content hashes of the Llama-3 chat template over the whole conversation. Every turn is a request, and turn k+1's prompt extends turn k's. | 4096 |

The processed traces (`data/processed_traces/freeinference_2026-08-30_128k.csv`,
`tracelab_coding_128k.csv`, `sharegpt_multiturn_llama3_4k.csv`) are in git. The
raw downloads are not (the TraceLab file is over GitHub's 100 MB limit):
download them from the links above into `data/raw_traces/freeinference_2026-08-30/`
and `data/raw_traces/tracelab/` to rebuild.

Agentic prompts are longer than a 128k model's window: median ~140k
(FreeInference) and ~130k (TraceLab) tokens, half the calls. Converters keep
every call and compact sessions instead (`common.finalize_trace`), as coding
agents do near their window: when a call's prompt + output would pass
`--compact-at` (120000), the context restarts as `--compact-to` (16384) tokens,
its system prompt and tools (kept, shared across sessions) plus a fresh
summary, plus the call's newly appended tokens; later calls extend the
compacted context by what they append in the source. `--max-tokens` (the
model's `max_model_len`) caps `--compact-at`; `--compact-at 0` drops calls over
`--max-tokens` instead. `finalize` runs the same pass (output ids, compaction)
on an existing session trace.

Built files in `data/processed_traces/`:

| File | Calls | Sessions | Blocks shared with the session's previous call |
|---|---|---|---|
| `freeinference_2026-08-30_128k.csv` (week of 2026-08-30; 5,235 compactions) | 190,382 | 2,043 | 95.4% |
| `tracelab_coding_128k.csv` (28,579 compactions) | 665,166 | 8,040 | 92.2% |
| `sharegpt_multiturn_llama3_4k.csv` (output ids by `finalize`; 4k cut) | 200,441 | 58,606 | |

(Shared: the previous call's prompt and output. Compaction summaries are cold.)

Turn 0 of the ShareGPT file is `sharegpt_llama3_tokenizer.csv` (the first-turn
lengths): the same tokenizer, template and filters, with matching lengths.

## Training workloads

A real trace runs far below a simulator's load, with sessions lasting hours. If
you speed up its clock, tool and think time shrink too, until a session's next
call arrives before its previous one has finished. `sessions.sample_session_workload`
instead overlays many sessions, each with its own call gaps. It draws sessions
in proportion to duration + window, places each one at a random phase, and
keeps the calls that land in a window of num_requests / qps seconds. In
expectation, the window's calls are a uniform sample of the trace's calls,
spread uniformly in time. ShareGPT has no times, so its turns get exponential
think time (`turn_gap`, default 30 s).

```bash
python -m frontier.trace_prep sample data/processed_traces/tracelab_coding_128k.csv \
    -n 500 --qps 2 --seed 0 -o workload.csv
python -m frontier.trace_prep describe workload.csv
```

In the RL environment (frontier/rl, a workload in configs/rl/*.yaml):

```yaml
- name: tracelab_qps2
  trace: tracelab            # or freeinference, sharegpt_multiturn, or a path
  arrivals: sessions
  qps: 2
  num_requests: 500
  simulation: {max_tokens: 131072}
  backend: frontier
  frontier: {enable_prefix_caching: true, model: llama3.1-8b, device: a100, ...}
```

## Caveats

- Sessions are closed loop by default. A session's first call in the window
  arrives at its sampled time, and each later call arrives `think_time` after
  its previous call completes in the simulation. `think_time` is the trace's
  gap between the end of one call and the start of the next (`completed_at`);
  ShareGPT turns get sampled think time. Different sessions still arrive
  independently, so `qps` sets the offered load, and the achieved call rate
  falls below it when the server falls behind, as with real agents. Use
  `closed_loop=False` (`--open-loop`, `session_closed_loop: false`) for fixed
  arrival times. Subagent sessions are independent sessions: a parent's next
  call waits on the trace's tool latency, not on the simulated subagent.
- Output blocks take the ids of the next prompt's blocks at the same positions
  when that prompt holds the call's whole prompt (chat history is append-only);
  a model that drops earlier reasoning from the prompt would reuse less (Codex
  output counts reasoning tokens). The block holding the last output token is
  never cached (vLLM does not write that token's KV).
- TraceLab: a call that continues the previous one (prompt >= previous prompt +
  output, cache read >= half the previous prompt) shares its prompt and output;
  otherwise it shares the provider's cache read, a lower bound. Every context
  starts with its provider's system prompt chain (Claude 722 blocks, Codex 397:
  the 10th percentile of first-call prompts, an assumption, as TraceLab does not
  give it). FreeInference's cross-session sharing is its real content hashes;
  its compactions keep the leading blocks a session's first prompt shares with
  another session's. FreeInference token counts are o200k, and its outputs use
  the provider's tokenizer.
- Compaction is synthetic: summaries are cold fresh blocks, and new tokens past
  what fits are cut (from calls whose source context was already far longer).
- Ids are unique within a file only. Do not concatenate traces from different
  sources.
