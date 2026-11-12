"""
Closed-loop agent sessions against a real vLLM server, for checking the
simulator with prefix caching on:

    vllm serve ... --enable-prefix-caching --enable-prompt-tokens-details
    python -m frontier.rl.vllm_sessions --num-sessions 4 --output real_sessions.csv

The sessions are the ones the simulator would run (SessionWorkload.build with
the same seed). Every 16-token block of a prompt gets deterministic token ids
from its hash id, so blocks with equal hash ids are equal text and vLLM's
prefix cache hits where Frontier's does. A session's next call is sent when
its previous call has finished plus the trace's tool gap.

One difference from the simulator: the tokens a call generated are the
model's, not the synthetic ones, so a later call re-reads the earlier prompt
from the cache but not the earlier output's blocks (a few hundred tokens per
call against tens of thousands of prompt tokens).
"""

import argparse
import asyncio
import json
import time
from functools import lru_cache
from math import ceil

import numpy as np
import pandas as pd

BLOCK_SIZE = 16


@lru_cache(maxsize=None)
def block_tokens(hash_id: int, vocab: int = 120_000) -> tuple:
    rng = np.random.default_rng([hash_id % (1 << 32), (hash_id >> 32) % (1 << 32)])
    return tuple(int(t) for t in rng.integers(1000, vocab, size=BLOCK_SIZE))


def prompt_token_ids(request) -> list:
    n = int(request.num_prefill_tokens)
    ids = request.block_hash_ids[: ceil(n / BLOCK_SIZE)]
    tokens = [t for h in ids for t in block_tokens(int(h))]
    return tokens[:n]


async def send(session, url, model, request, origin):
    import aiohttp

    n_out = int(request.num_decode_tokens)
    body = {
        "model": model, "prompt": prompt_token_ids(request),
        "max_tokens": n_out, "min_tokens": n_out, "ignore_eos": True,
        "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
    }
    sent = time.perf_counter()
    stamps, usage = [], None
    try:
        async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=None)) as resp:
            if resp.status != 200:
                return {"error": f"HTTP {resp.status}: {(await resp.text())[:200]}"}, sent - origin
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("choices") and chunk["choices"][0].get("text") is not None:
                    stamps.append(time.perf_counter())
    except Exception as exc:  # noqa: BLE001 - recorded per call
        return {"error": repr(exc)}, sent - origin
    gaps = np.diff(stamps) if len(stamps) > 1 else np.array([])
    details = (usage or {}).get("prompt_tokens_details") or {}
    return {
        "ttft": stamps[0] - sent if stamps else float("nan"),
        "e2e": stamps[-1] - sent if stamps else float("nan"),
        "mean_tbt": float(gaps.mean()) if len(gaps) else float("nan"),
        "cached_tokens": details.get("cached_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "error": "",
    }, sent - origin


async def run_session(session, url, model, first, next_call, origin, results):
    request, start = first, first.arrived_at
    await asyncio.sleep(max(0.0, origin + start - time.perf_counter()))
    while request is not None:
        record, sent_at = await send(session, url, model, request, origin)
        results.append({
            "session": request.session_id, "call_index": request._rl_call_index,
            "sent_at": sent_at, "prompt": int(request.num_prefill_tokens),
            "output": int(request.num_decode_tokens), **record,
        })
        if record["error"] or request.id not in next_call:
            return
        request, gap = next_call[request.id]
        await asyncio.sleep(gap)


async def run(first_calls, next_call, url, model, lead_s):
    import aiohttp

    results = []
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector, read_bufsize=1 << 26) as session:
        origin = time.perf_counter() + lead_s
        await asyncio.gather(*[
            run_session(session, url, model, first, next_call, origin, results) for first in first_calls
        ])
    return pd.DataFrame(results).sort_values(["session", "call_index"])


def main() -> None:
    from frontier.rl.agent_trace import DEFAULT_SHARED_PREFIX, SessionWorkload

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--calls", default="rl_runs/agent_traces/tracelab_calls.csv")
    parser.add_argument("--num-sessions", type=int, default=4)
    parser.add_argument("--calls-per-session", type=int, default=10)
    parser.add_argument("--session-rate", type=float, default=None, help="sessions/s (default: all start at once)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shared-prefix", action="store_true")
    parser.add_argument("--max-prompt", type=int, default=None, help="skip sessions whose prompts exceed this")
    parser.add_argument("--url", default="http://localhost:8000/v1/completions")
    parser.add_argument("--model", default="nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct")
    parser.add_argument("--lead-s", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workload = SessionWorkload(
        args.calls, num_sessions=args.num_sessions, calls_per_session=args.calls_per_session,
        session_rate=args.session_rate,
        shared_prefix_tokens=DEFAULT_SHARED_PREFIX if args.shared_prefix else (),
    )
    first_calls, next_call, _ = workload.build(args.seed)
    out = asyncio.run(run(first_calls, next_call, args.url, args.model, args.lead_s))
    out.to_csv(args.output, index=False)
    ok = out[out.error == ""]
    print(f"{len(ok)}/{len(out)} calls ok; mean TTFT {ok.ttft.mean():.2f}s, mean e2e {ok.e2e.mean():.1f}s, "
          f"cached {ok.cached_tokens.sum()}/{ok.prompt.sum()} prompt tokens")
    if len(ok) < len(out):
        print("first error:", out[out.error != ""].error.iloc[0])


if __name__ == "__main__":
    main()
