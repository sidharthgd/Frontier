"""TraceLab coding-agent trace -> session trace.

Source: github.com/uw-syfi/TraceLab releases, syfi_coding_trace.jsonl.gz
(CC BY 4.0): one LLM round of Claude Code or Codex per line. No content or
block hashes are released, only the provider's cache accounting:

    input_tokens_total = prefix_tokens + newly_append_tokens

where prefix_tokens is the provider's cache read. TraceLab's
docs/prompt_cache_accounting.md finds next.prefix_tokens ~= last
input_tokens_total for adjacent rounds of a session (median error -12
tokens), i.e. a round's cached prefix is the previous round's prompt. Block
ids are synthesized on that model, for each round's prompt and output:

- A round that continues its predecessor (its prompt is at least the
  previous prompt + output, and the cache read covers half the previous
  prompt) shares the previous prompt and output: chat history is
  append-only, and vLLM caches the output's full blocks too
  (continues_through_output). Otherwise it shares prefix_tokens // 16 blocks
  (the provider's read: a restart, an edited context).
- Every context starts with its provider's system prompt and tools, one
  chain per provider shared by all users (provider_prefix_blocks: the 10th
  percentile of first-round prompts, as TraceLab does not give them).
- A session's first round with a cache read longer than that shares it with
  the user's latest earlier round with the same provider.

The provider's cache read is a lower bound on what the content shares (a
miss from TTL expiry or a moved boundary is a miss here too) except where a
round continues its predecessor.

Ids are numbered so a prompt is a few runs: each session has a trunk of
consecutive ids by block position, and a round's blocks past what the next
round shares (its tail) take ids from a separate range, so the next round
extends the trunk instead of starting a run. A round that shares less than
its parent kept forks the trunk.

The raw trace then goes through common.finalize_trace: contexts longer than
the model's (median input ~130k tokens) are compacted, keeping the system
prompt chain.

Arrival is the round's last input event (user message or tool result), else
its first event; completion is its last output event. Output is output_tokens, which for Codex includes reasoning
tokens (generated, so they cost decode steps).
"""

from __future__ import annotations

import bisect
import datetime as dt
import gzip
import json
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple, Optional, Tuple

from frontier.trace_prep.common import (
    BLOCK_TOKENS,
    LOCAL_BITS,
    BlockRuns,
    SessionTraceWriter,
)

INPUT_EVENTS = ("user_message", "tool_result")
# tails take ids from the top half of the id space
TAIL_BASE = 1 << (LOCAL_BITS - 1)


class Round(NamedTuple):
    arrived_at: float
    completed_at: Optional[float]
    session: str
    user: str
    provider: str
    prompt: int
    prefix: int
    completion: int


def _seconds(timestamp: str) -> float:
    return dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()


def arrival_time(round_: dict) -> Optional[float]:
    events = round_.get("timing_events") or []
    inputs = [e["timestamp"] for e in events if e.get("event_type") in INPUT_EVENTS and e.get("timestamp")]
    if inputs:
        return max(_seconds(t) for t in inputs)
    others = [e["timestamp"] for e in events if e.get("timestamp")]
    return min(_seconds(t) for t in others) if others else None


def completion_time(round_: dict, arrived_at: float) -> Optional[float]:
    """The round's last output event (text, tool call, reasoning, usage)."""
    events = round_.get("timing_events") or []
    outputs = [
        _seconds(e["timestamp"]) for e in events
        if e.get("event_type") not in INPUT_EVENTS and e.get("timestamp")
    ]
    outputs = [t for t in outputs if t >= arrived_at]
    return max(outputs) if outputs else None


def read_rounds(path: str, stats: Counter) -> Dict[str, List[Round]]:
    """Rounds by session, each session in time order."""
    sessions: Dict[str, List[Tuple[float, int, Round]]] = defaultdict(list)
    with gzip.open(path, "rt") as file:
        for line in file:
            r = json.loads(line)
            stats["rounds"] += 1
            arrived_at = arrival_time(r)
            if arrived_at is None:
                stats["dropped_no_time"] += 1
                continue
            sessions[r["session_id"]].append(
                (
                    arrived_at,
                    int(r.get("round_index") or 0),
                    Round(
                        arrived_at,
                        completion_time(r, arrived_at),
                        r["session_id"],
                        r.get("user") or "",
                        r.get("provider") or "",
                        int(r.get("input_tokens_total") or 0),
                        int(r.get("prefix_tokens") or 0),
                        int(r.get("output_tokens") or 0),
                    ),
                )
            )
    return {s: [r for _, _, r in sorted(v)] for s, v in sessions.items()}


class _UserHistory:
    """Each user's rounds by time, to find a first round's parent."""

    def __init__(self) -> None:
        self.times: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.runs: Dict[Tuple[str, str], List[BlockRuns]] = defaultdict(list)

    def add(self, key, time: float, runs: BlockRuns) -> None:
        index = bisect.bisect_right(self.times[key], time)
        self.times[key].insert(index, time)
        self.runs[key].insert(index, runs)

    def latest_before(self, key, time: float) -> Optional[BlockRuns]:
        index = bisect.bisect_left(self.times.get(key, []), time)
        return self.runs[key][index - 1] if index else None


def provider_prefix_blocks(sessions: Dict[str, List[Round]], quantile: float = 0.1) -> Dict[str, int]:
    """Blocks of the system prompt and tool definitions every session of a
    provider starts with. TraceLab does not give them; this is the 10th
    percentile of the provider's first-round prompts (a lower bound on what
    they carry): Claude ~11.6k tokens, Codex ~6.4k."""
    import numpy as np

    firsts: Dict[str, List[int]] = defaultdict(list)
    for rounds in sessions.values():
        firsts[rounds[0].provider].append(rounds[0].prompt)
    return {
        provider: int(np.quantile(prompts, quantile)) // BLOCK_TOKENS
        for provider, prompts in firsts.items()
        if provider and len(prompts) >= 10
    }


def continues_through_output(previous: Round, r: Round) -> bool:
    """Whether r's prompt is previous's prompt and output plus what came
    after (chat history is append-only), as vLLM's cache would see it. The
    provider's cache read (prefix) often stops short of the previous prompt's
    end (median 12 tokens, p90 ~1.2k: where its cache breakpoint sat), so a
    round counts as continuing when it is at least that long and its cache
    read covers half the previous prompt (not a restart)."""
    return r.prompt >= previous.prompt + previous.completion and r.prefix * 2 >= previous.prompt


def convert(
    path: str,
    output: str,
    max_tokens: Optional[int] = 131072,
    compact_at: Optional[int] = 120_000,
    compact_to: int = 16_384,
) -> Counter:
    import os

    from frontier.trace_prep.common import finalize_trace

    stats: Counter = Counter()
    sessions = read_rounds(path, stats)
    order = sorted(sessions, key=lambda s: sessions[s][0].arrived_at)

    users = _UserHistory()
    next_trunk, next_tail = 0, TAIL_BASE
    # one chain per provider for the shared system prompt
    system_blocks = provider_prefix_blocks(sessions)
    system: Dict[str, BlockRuns] = {}
    for provider, n in sorted(system_blocks.items()):
        system[provider] = BlockRuns()
        system[provider].extend(next_trunk, n)
        next_trunk += n
    raw = f"{output}.raw.{os.getpid()}.csv"
    writer = SessionTraceWriter(raw)
    keep_blocks: Dict[int, int] = {}
    for session_id, session in enumerate(order):
        rounds = sessions[session]
        # blocks of each round's prompt and of its prompt + output
        prompt_blocks = [r.prompt // BLOCK_TOKENS for r in rounds]
        blocks = [(r.prompt + r.completion) // BLOCK_TOKENS for r in rounds]
        first = rounds[0]
        head = system.get(first.provider, BlockRuns())
        keep_blocks[session_id] = head.num_blocks
        parent = (
            users.latest_before((first.user, first.provider), first.arrived_at)
            if first.prefix > 0
            else None
        )
        parent_shared = min(first.prefix // BLOCK_TOKENS, prompt_blocks[0], parent.num_blocks) if parent else 0
        if parent_shared > min(head.num_blocks, prompt_blocks[0]):
            stats["first_round_parent_from_user"] += 1
            trunk = parent.prefix(parent_shared)
        else:
            trunk = head.prefix(prompt_blocks[0])
        # shared[k]: blocks round k's prompt shares with round k-1 (its prompt
        # and output); every context keeps the system prompt
        shared = [trunk.num_blocks]
        for k, r in enumerate(rounds[1:], start=1):
            previous = rounds[k - 1]
            if continues_through_output(previous, r):
                n = min(blocks[k - 1], prompt_blocks[k])
                stats["continues_through_output"] += 1
            else:
                n = min(r.prefix // BLOCK_TOKENS, prompt_blocks[k], blocks[k - 1])
            shared.append(max(n, min(head.num_blocks, prompt_blocks[k], prompt_blocks[k - 1])))
        shared.append(0)

        # ids are made for every round, kept or not, so the chain of content
        # follows the real session
        for k, r in enumerate(rounds):
            trunk = trunk.prefix(shared[k])
            on_trunk = min(blocks[k], max(shared[k], shared[k + 1]))
            # the system prompt covers prompt blocks only, never an output
            if trunk.num_blocks < min(head.num_blocks, on_trunk, prompt_blocks[k]):
                trunk = head.prefix(min(on_trunk, prompt_blocks[k]))
            grow = on_trunk - trunk.num_blocks
            trunk.extend(next_trunk, grow)
            next_trunk += max(grow, 0)
            runs = trunk.prefix(on_trunk)
            runs.extend(next_tail, blocks[k] - on_trunk)
            next_tail += max(blocks[k] - on_trunk, 0)
            users.add((r.user, r.provider), r.arrived_at, runs)

            if r.prompt <= 0 or r.completion <= 0:
                stats["dropped_empty"] += 1
                continue
            writer.write(
                arrived_at=r.arrived_at,
                completed_at=r.completed_at,
                session_id=session_id,
                step=k,
                num_prefill_tokens=r.prompt,
                num_decode_tokens=r.completion,
                block_hash_ids=runs.encode() if blocks[k] else "",
                ids_cover_output=True,
            )
    assert next_trunk < TAIL_BASE and next_tail < (1 << LOCAL_BITS)
    writer.close()
    stats["sessions"] = len(order)
    try:
        stats.update(
            finalize_trace(raw, output, max_tokens, compact_at, compact_to, keep_blocks)
        )
    finally:
        os.remove(raw)
    return stats
