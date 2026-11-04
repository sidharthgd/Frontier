"""
Replay a request trace against a running vLLM OpenAI server and record
per-request timings, for checking Frontier's predictions against real vLLM:

    vllm serve meta-llama/Llama-3.1-8B-Instruct ...   # see rl/README
    python -m frontier.rl.vllm_replay --trace slice.csv --output real.csv

The trace is a CSV with arrived_at (seconds from the start),
num_prefill_tokens and num_decode_tokens, the format Frontier's trace replay
reads. Each row becomes one /v1/completions request sent at its arrival
offset, with a prompt of exactly num_prefill_tokens token ids (deterministic,
distinct per request so prefix caching cannot hit) and exactly
num_decode_tokens output tokens (min_tokens = max_tokens, ignore_eos).
Responses are streamed; the time of every chunk gives TTFT, the gaps between
tokens and the end-to-end time, measured at the client.

Client-side TTFT includes the HTTP and detokenization path, so it is a little
above Frontier's TTFT (arrival to prefill completion on the server).
"""

import argparse
import asyncio
import json
import time

import numpy as np
import pandas as pd


def prompt_ids(index: int, length: int, vocab: int = 120_000) -> list:
    # distinct per request and per position, avoiding special token ids
    rng = np.random.default_rng(index)
    return (rng.integers(1000, vocab, size=length)).tolist()


async def send(session, url, model, row, index, origin, results):
    import aiohttp

    await asyncio.sleep(max(0.0, origin + row.arrived_at - time.perf_counter()))
    body = {
        "model": model,
        "prompt": prompt_ids(index, int(row.num_prefill_tokens)),
        "max_tokens": int(row.num_decode_tokens),
        "min_tokens": int(row.num_decode_tokens),
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    sent = time.perf_counter()
    token_times, usage = [], None
    try:
        async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=None)) as resp:
            if resp.status != 200:
                results.append({"index": index, "error": f"HTTP {resp.status}: {await resp.text()}"})
                return
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("choices") and chunk["choices"][0].get("text") is not None:
                    token_times.append(time.perf_counter())
    except Exception as exc:  # noqa: BLE001 - recorded per request
        results.append({"index": index, "error": repr(exc)})
        return
    gaps = np.diff(token_times) if len(token_times) > 1 else np.array([])
    results.append({
        "index": index,
        "arrived_at": row.arrived_at,
        "num_prefill_tokens": int(row.num_prefill_tokens),
        "num_decode_tokens": int(row.num_decode_tokens),
        "sent_at": sent - origin,
        "ttft": token_times[0] - sent if token_times else float("nan"),
        "e2e": token_times[-1] - sent if token_times else float("nan"),
        "mean_tbt": float(gaps.mean()) if len(gaps) else float("nan"),
        "p99_tbt": float(np.quantile(gaps, 0.99)) if len(gaps) else float("nan"),
        "num_chunks": len(token_times),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "error": "",
    })


async def replay(trace: pd.DataFrame, url: str, model: str, lead_s: float):
    import aiohttp

    results = []
    connector = aiohttp.TCPConnector(limit=0)
    # streamed lines can exceed aiohttp's default 512 KB line limit
    async with aiohttp.ClientSession(connector=connector, read_bufsize=1 << 26) as session:
        origin = time.perf_counter() + lead_s
        await asyncio.gather(*[
            send(session, url, model, row, i, origin, results)
            for i, row in enumerate(trace.itertuples())
        ])
    return pd.DataFrame(results).sort_values("index")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--url", default="http://localhost:8000/v1/completions")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--lead-s", type=float, default=2.0)
    args = parser.parse_args()

    trace = pd.read_csv(args.trace)
    trace = trace.sort_values("arrived_at").reset_index(drop=True)
    trace["arrived_at"] -= trace["arrived_at"].iloc[0]
    out = asyncio.run(replay(trace, args.url, args.model, args.lead_s))
    out.to_csv(args.output, index=False)
    ok = out[out.error == ""]
    print(f"{len(ok)}/{len(out)} requests ok; mean TTFT {ok.ttft.mean():.3f}s, "
          f"mean e2e {ok.e2e.mean():.2f}s, mean TBT {1000 * ok.mean_tbt.mean():.1f}ms")
    bad = out[out.error != ""]
    if len(bad):
        print("first error:", bad.error.iloc[0][:300])


if __name__ == "__main__":
    main()
