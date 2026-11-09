"""
Agent sessions from TraceLab (uw-syfi/TraceLab, coding-agent traces from
Claude Code and Codex) for the RL environment.

    python -m frontier.rl.agent_trace data/raw_traces/syfi_coding_trace.jsonl.gz \\
        rl_runs/agent_traces/tracelab_calls.csv

Conversion (one row per LLM call, in session order):

- Context. A call's prompt is the session's context: the previous call's
  prompt plus everything appended since (the previous output and the tool
  results), so consecutive calls share a prefix and continue one hash chain.
  TraceLab's own input totals give the growth; a call whose input is less
  than half of the previous one starts a new context (the agent compacted or
  started over), as does a session's first call.
- Context limit. Contexts are kept under `compact_at` tokens (prompt +
  output), below the model's 131,072. A call that would exceed it compacts:
  the context restarts as a `compact_to`-token summary plus the call's new
  tokens, with a new hash chain, as coding agents compact near their window.
  TraceLab contexts are far longer (median input 132k; Claude has a 1M
  window), so with a 128k model sessions run near the limit and compact.
- Output. output_tokens (Codex counts reasoning in it), clipped to
  `max_output`.
- Tool gap. The time from the call's last event to the next call's first
  event (tool execution, or a human's reply), clipped to [0, `max_gap_s`].

SessionWorkload samples sessions from the converted calls and builds Frontier
requests: each session's first call arrives at its start time and each later
call arrives when the previous one finishes plus its tool gap (closed loop).
Block hash ids identify 16-token blocks of a hash chain, so the prefix cache
recognizes a session's earlier context.
"""

import argparse
import gzip
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BLOCK_SIZE = 16
# hash id = chain * CHAIN_STRIDE + block index
CHAIN_STRIDE = 1 << 24
# hash ids of a provider's shared system prompt (above every context chain)
SHARED_BASE = 1 << 50
# Leading tokens every session of a provider shares (system prompt and tool
# definitions). TraceLab does not give them; these are the 10th percentile of
# each provider's first-call input, a lower bound on what it carries.
DEFAULT_SHARED_PREFIX = (("claude", 11_520), ("codex", 6_336))


def _ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def convert_tracelab(
    src: str,
    dst: str,
    compact_at: int = 120_000,
    compact_to: int = 16_384,
    max_output: int = 16_384,
    max_gap_s: float = 120.0,
) -> pd.DataFrame:
    sessions: Dict[str, list] = {}
    with gzip.open(src, "rt") as f:
        for line in f:
            r = json.loads(line)
            events = [_ts(e["timestamp"]) for e in r["timing_events"] if e.get("timestamp")]
            if not events:
                continue
            sessions.setdefault(r["session_id"], []).append((
                r["round_index"], r["provider"], int(r["input_tokens_total"] or 0),
                int(r["output_tokens"] or 0), min(events), max(events),
            ))

    rows = []
    chain = 0
    for session_index, calls in enumerate(sessions.values()):
        calls.sort()
        prompt = prev_input = prev_output = None
        for k, (_, provider, real_input, output, start, end) in enumerate(calls):
            output = int(np.clip(output, 1, max_output))
            if prompt is None or real_input < prev_input / 2:
                chain += 1
                prompt = min(max(real_input, 1), compact_at - output)
            else:
                prompt = prompt + prev_output + max(0, real_input - prev_input - prev_output)
                if prompt + output > compact_at:
                    chain += 1
                    new_tokens = max(0, real_input - prev_input)
                    prompt = min(compact_to + new_tokens, compact_at - output)
            gap = (
                float(np.clip(calls[k + 1][4] - end, 0.0, max_gap_s))
                if k + 1 < len(calls) else float("nan")
            )
            rows.append((session_index, k, provider, chain, prompt, output, gap, real_input))
            prev_input, prev_output = real_input, output

    df = pd.DataFrame(rows, columns=[
        "session", "call", "provider", "chain", "prompt_tokens", "output_tokens",
        "gap_after_s", "real_input_tokens",
    ])
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    df.to_csv(dst, index=False)
    return df


_CALLS: Dict[str, pd.DataFrame] = {}


def _load(path: str) -> pd.DataFrame:
    if path not in _CALLS:
        _CALLS[path] = pd.read_csv(path)
    return _CALLS[path]


@dataclass(frozen=True)
class SessionWorkload:
    """
    num_sessions sessions, each a window of up to calls_per_session
    consecutive calls (at a random offset). Sessions start as a Poisson
    process at session_rate per second (None: all at time 0).
    """

    calls_csv: str
    num_sessions: int = 16
    calls_per_session: int = 30
    min_calls: int = 10
    session_rate: Optional[float] = None
    provider: Optional[str] = None  # "claude" | "codex" | None for both
    # (provider, tokens): the leading tokens of every context of that provider
    # are one shared hash chain, so sessions (and compactions) reuse each
    # other's system prompt in the prefix cache. () gives every context its own.
    shared_prefix_tokens: Tuple[Tuple[str, int], ...] = ()

    def plan(self, seed: int) -> List[pd.DataFrame]:
        calls = _load(self.calls_csv)
        if self.provider is not None:
            calls = calls[calls.provider == self.provider]
        counts = calls.groupby("session").size()
        eligible = counts[counts >= self.min_calls].index.to_numpy()
        rng = np.random.default_rng(seed)
        chosen = rng.choice(eligible, size=self.num_sessions, replace=False)
        by_session = dict(tuple(calls[calls.session.isin(chosen)].groupby("session")))
        windows = []
        for session in chosen:
            rows = by_session[session].sort_values("call")
            start = int(rng.integers(0, max(1, len(rows) - self.calls_per_session + 1)))
            windows.append(rows.iloc[start:start + self.calls_per_session])
        return windows

    def build(self, seed: int):
        """
        Returns (first_calls, next_call, all_requests): first_calls arrive at
        their sessions' start times; next_call maps a request id to the next
        call of its session and the tool gap before it.
        """
        from frontier.entities import Request

        rng = np.random.default_rng(seed + 1)
        if self.session_rate:
            starts = np.cumsum(rng.exponential(1.0 / self.session_rate, self.num_sessions))
            starts -= starts[0]
        else:
            starts = np.zeros(self.num_sessions)

        shared = {name: tokens // BLOCK_SIZE for name, tokens in self.shared_prefix_tokens}
        provider_index = {name: i for i, (name, _) in enumerate(self.shared_prefix_tokens)}
        first_calls, next_call, all_requests = [], {}, []
        for session_index, (window, start) in enumerate(zip(self.plan(seed), starts)):
            previous: Optional[Tuple[Request, float]] = None
            for call_index, row in enumerate(window.itertuples()):
                total = int(row.prompt_tokens) + int(row.output_tokens)
                num_blocks = -(-total // BLOCK_SIZE)
                base = int(row.chain) * CHAIN_STRIDE
                num_shared = min(shared.get(row.provider, 0), num_blocks)
                shared_base = SHARED_BASE + provider_index.get(row.provider, 0) * CHAIN_STRIDE
                request = Request(
                    arrived_at=float(start),
                    num_prefill_tokens=int(row.prompt_tokens),
                    num_decode_tokens=int(row.output_tokens),
                    block_hash_ids=(
                        list(range(shared_base, shared_base + num_shared))
                        + list(range(base + num_shared, base + num_blocks))
                    ),
                    session_id=session_index,
                )
                request._rl_call_index = call_index
                all_requests.append(request)
                if previous is None:
                    first_calls.append(request)
                else:
                    next_call[previous[0].id] = (request, previous[1])
                gap = row.gap_after_s
                previous = (request, 0.0 if np.isnan(gap) else float(gap))
        return first_calls, next_call, all_requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--compact-at", type=int, default=120_000)
    parser.add_argument("--compact-to", type=int, default=16_384)
    parser.add_argument("--max-output", type=int, default=16_384)
    parser.add_argument("--max-gap-s", type=float, default=120.0)
    args = parser.parse_args()
    df = convert_tracelab(args.src, args.dst, args.compact_at, args.compact_to,
                          args.max_output, args.max_gap_s)
    q = lambda c: [int(np.percentile(df[c], p)) for p in (10, 50, 90, 99)]
    print(f"{len(df)} calls in {df.session.nunique()} sessions, {df.chain.nunique()} contexts")
    print("prompt tokens p10/50/90/99", q("prompt_tokens"))
    print("output tokens p10/50/90/99", q("output_tokens"))
    print("calls per context, median", int(df.groupby("chain").size().median()))
    print("gap s p10/50/90/99", [round(float(np.nanpercentile(df.gap_after_s, p)), 1) for p in (10, 50, 90, 99)])


if __name__ == "__main__":
    main()
