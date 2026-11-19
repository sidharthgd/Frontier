"""The session trace format the converters write, and helpers they share.

A session trace is a CSV with one row per LLM call:

    arrived_at          seconds (absent for ShareGPT, which has no times)
    completed_at        when the source finished the call, same clock (empty if
                        unknown); the sampler turns it into think_time, the gap
                        before the session's next call, for closed-loop replay
    session_id          int; calls of one conversation / agent session
    step                position of the call in its session, from 0
    num_prefill_tokens  prompt tokens
    num_decode_tokens   output tokens
    block_hash_ids      chained ids of the call's full 16-token blocks, as
                        runs (frontier/utils/block_hash_ids.py): its prompt's,
                        then (ids_cover_output) those its output completes
    ids_cover_output    1 when block_hash_ids also name the output's blocks:
                        (num_prefill_tokens + num_decode_tokens) // 16 ids, of
                        which the first num_prefill_tokens // 16 are the
                        prompt's; 0: the prompt's only

Frontier's Request.prompt_block_hash_ids covers the whole prompt, and with
ids_cover_output Frontier binds the output's full blocks to the rest as they
are decoded (vLLM caches generated tokens' full blocks too), so a session's
next call, whose prompt holds this one's output, can hit them. With
arrived_at the file replays directly (trace_replay, prefix caching on); every
file can also be sampled into training workloads with
frontier.trace_prep.sessions.

finalize_trace turns a converter's raw trace into the built file: it gives
every call output ids and compacts sessions whose context outgrows the model
(see its docstring).

Block ids are chained: id k names the whole prefix through block k, so two
prompts share a prefix of k blocks exactly when their first k ids agree, and
coarser blocks of 16*m tokens are ids[m-1::m]. Ids are unique within one
file only (FreeInference weeks get namespaces of 2**36 ids), so do not
concatenate files from different sources without renumbering.

`step` is the call's index in its session in the source, so it has gaps
where calls were dropped (empty calls, or calls compaction cannot fit).

A call under 16 tokens has no full block; its row carries one placeholder
id (PlaceholderIds) so that prefix caching, which needs every row to have
ids, accepts the file. Frontier looks up only the first
num_prefill_tokens // 16 ids, so a placeholder names nothing.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np

from frontier.utils.block_hash_ids import encode_block_hash_ids

BLOCK_TOKENS = 16

COLUMNS = [
    "arrived_at",
    "completed_at",
    "session_id",
    "step",
    "num_prefill_tokens",
    "num_decode_tokens",
    "block_hash_ids",
    "ids_cover_output",
]

# compacted contexts and outputs nothing reuses get ids from here (above every
# source's, below the placeholders)
FRESH_BASE = 1 << 48

# FreeInference block ids are local to a week: week w's ids are w << 36 | id
LOCAL_BITS = 36


def namespace_base(sub: int = 0) -> int:
    assert 0 <= sub < (1 << 20)
    return sub << LOCAL_BITS


class PlaceholderIds:
    """Unique ids, above every source's, for prompts with no full block."""

    BASE = 1 << 50

    def __init__(self) -> None:
        self._next = self.BASE

    def next(self) -> str:
        self._next += 1
        return str(self._next - 1)


def check_local_ids(ids: np.ndarray) -> None:
    if ids.size and (ids.min() < 0 or ids.max() >= (1 << LOCAL_BITS)):
        raise ValueError("block ids exceed the 36-bit local namespace")


@dataclass
class BlockRuns:
    """A prompt's block ids as runs of consecutive ids (start, length)."""

    starts: List[int] = field(default_factory=list)
    lengths: List[int] = field(default_factory=list)

    @property
    def num_blocks(self) -> int:
        return sum(self.lengths)

    def prefix(self, num_blocks: int) -> "BlockRuns":
        out = BlockRuns()
        remaining = num_blocks
        for start, length in zip(self.starts, self.lengths):
            if remaining <= 0:
                break
            take = min(length, remaining)
            out.starts.append(start)
            out.lengths.append(take)
            remaining -= take
        return out

    def extend(self, start: int, length: int) -> None:
        if length <= 0:
            return
        if self.starts and self.starts[-1] + self.lengths[-1] == start:
            self.lengths[-1] += length
        else:
            self.starts.append(start)
            self.lengths.append(length)

    def encode(self) -> str:
        return "|".join(
            str(s) if n == 1 else f"{s}-{s + n - 1}"
            for s, n in zip(self.starts, self.lengths)
        )


def prompt_tokens_for_blocks(num_blocks: int, num_tokens: Optional[int]) -> int:
    """A prompt length consistent with num_blocks full blocks: num_tokens when
    it agrees, else the nearest length that does."""
    low = num_blocks * BLOCK_TOKENS
    if num_tokens is None:
        return low
    return int(min(max(num_tokens, low), low + BLOCK_TOKENS - 1))


class SessionTraceWriter:
    """Streams rows to a session trace CSV, then sorts it by arrival."""

    def __init__(self, path: str, with_arrivals: bool = True) -> None:
        self.path = path
        self.columns = COLUMNS if with_arrivals else COLUMNS[2:]
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._temporary = f"{path}.{os.getpid()}.tmp"
        self._file = open(self._temporary, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.columns)
        self.num_rows = 0

    def write(
        self,
        *,
        session_id: int,
        step: int,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        block_hash_ids: str,
        arrived_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        ids_cover_output: bool = False,
    ) -> None:
        row = [
            session_id, step, num_prefill_tokens, num_decode_tokens, block_hash_ids,
            int(bool(ids_cover_output)),
        ]
        if "arrived_at" in self.columns:
            row[:0] = [f"{arrived_at:.3f}", "" if completed_at is None else f"{completed_at:.3f}"]
        self._writer.writerow(row)
        self.num_rows += 1

    def close(self) -> None:
        import pandas as pd

        self._file.close()
        frame = pd.read_csv(self._temporary, dtype={"block_hash_ids": str})
        keys = ["arrived_at", "session_id", "step"] if "arrived_at" in frame else ["session_id", "step"]
        if "arrived_at" in frame:
            origin = frame["arrived_at"].min()
            frame["arrived_at"] -= origin
            frame["completed_at"] -= origin
        frame = frame.sort_values(keys, kind="stable")
        frame.to_csv(self._temporary, index=False, float_format="%.3f")
        os.replace(self._temporary, self.path)


def describe(path: str) -> str:
    """Summary of a written session trace, including the fraction of prompt
    blocks each call shares with the previous call of its session (the reuse
    a cache that keeps every session's last prompt would see)."""
    import pandas as pd

    from frontier.utils.block_hash_ids import decode_block_hash_ids

    frame = pd.read_csv(path, dtype={"block_hash_ids": str})
    hit = total = 0
    by_session = frame.sort_values(["session_id", "step"], kind="stable")
    last_session, last = None, None
    for session_id, text, prompt in zip(
        by_session["session_id"], by_session["block_hash_ids"], by_session["num_prefill_tokens"]
    ):
        ids = decode_block_hash_ids(text)
        query = ids[: prompt // BLOCK_TOKENS]
        if session_id == last_session:
            hit += common_prefix(last, query)
        total += len(query)
        last_session, last = session_id, ids
    sessions = frame.groupby("session_id").size()
    lines = [
        f"{path}: {len(frame)} requests in {len(sessions)} sessions "
        f"(requests/session p50 {sessions.median():.0f}, p99 {sessions.quantile(0.99):.0f})",
        frame[["num_prefill_tokens", "num_decode_tokens"]]
        .describe(percentiles=[0.5, 0.9, 0.99])
        .round(0)
        .to_string(),
        f"prompt blocks shared with the session's previous call (its prompt and output): {hit / max(total, 1):.1%}",
    ]
    if "arrived_at" in frame:
        span = frame["arrived_at"].max() - frame["arrived_at"].min()
        lines.append(f"span {span:.0f} s, native rate {len(frame) / max(span, 1e-9):.3f} qps")
    return "\n".join(lines)


def common_prefix(a: np.ndarray, b: np.ndarray) -> int:
    """Number of leading ids a and b share."""
    n = min(len(a), len(b))
    differ = np.flatnonzero(a[:n] != b[:n])
    return int(differ[0]) if differ.size else n


class FreshIds:
    """Unique ids for blocks no source names: compacted contexts, outputs."""

    def __init__(self, base: int = FRESH_BASE) -> None:
        self._next = base

    def take(self, n: int) -> np.ndarray:
        ids = np.arange(self._next, self._next + max(n, 0), dtype=np.int64)
        self._next += max(n, 0)
        assert self._next < PlaceholderIds.BASE
        return ids


@dataclass
class Call:
    """One call of a session while finalize_trace works on it."""

    row: dict  # the raw trace row (arrived_at, completed_at, step, ...)
    prompt: int
    output: int
    ids: np.ndarray  # prompt blocks, then output blocks once they have ids


def add_output_ids(calls: List[Call], fresh: FreshIds) -> None:
    """Give each call's output blocks ids, in place: a call's prompt plus
    output fill (prompt + output) // 16 blocks. When the session's next call
    starts with this call's whole prompt, its output is taken to follow
    verbatim (chat history is append-only), so those blocks take the next
    prompt's ids at the same positions, as far as the next prompt reaches;
    the rest get fresh ids. Calls whose ids already cover their output are
    left alone."""
    for k, call in enumerate(calls):
        prompt_blocks = call.prompt // BLOCK_TOKENS
        total = (call.prompt + call.output) // BLOCK_TOKENS
        if len(call.ids) >= total:
            call.ids = call.ids[:total]
            continue
        own = call.ids[:prompt_blocks]
        tail = np.zeros(0, dtype=np.int64)
        if k + 1 < len(calls):
            following = calls[k + 1].ids[: calls[k + 1].prompt // BLOCK_TOKENS]
            if common_prefix(own, following) == prompt_blocks:
                tail = following[prompt_blocks:total]
        call.ids = np.concatenate([own, tail, fresh.take(total - prompt_blocks - len(tail))])


def compact_session(
    calls: List[Call],
    keep_blocks: int,
    compact_at: int,
    compact_to: int,
    fresh: FreshIds,
    stats,
) -> List[Call]:
    """The session as a model with a compact_at-token context serves it.

    Calls run as in the source until a call's prompt + output would pass
    compact_at. Then the agent compacts, as coding agents do near their
    window: the context restarts as compact_to tokens (its first keep_blocks
    blocks, the system prompt and tools, kept with their ids, then a fresh
    summary) plus the call's newly appended tokens (tokens past what it shares
    with the previous call), capped to fit. Later calls extend the compacted
    context by what they append in the source (shared - lost tail + new), with
    fresh ids, and compact again when it fills. A call that starts over in
    the source (shares less than half its predecessor's prompt) leaves the
    compacted context: if it fits it runs as in the source.

    Output block counts are kept, so ids stay (prompt + output) // 16 long.
    A call whose output alone does not fit next to compact_to tokens is
    dropped.
    """
    out: List[Call] = []
    keep_blocks = min(keep_blocks, compact_to // BLOCK_TOKENS)
    summary_blocks = compact_to // BLOCK_TOKENS - keep_blocks
    previous_real: Optional[np.ndarray] = None
    previous_prompt_blocks = 0
    simulated: Optional[np.ndarray] = None  # the previous emitted call's ids
    compacted = False
    for call in calls:
        real = call.ids
        prompt_blocks = call.prompt // BLOCK_TOKENS
        output_blocks = len(real) - prompt_blocks
        remainder = call.prompt % BLOCK_TOKENS
        shared = 0 if previous_real is None else common_prefix(previous_real, real[:prompt_blocks])
        restart = previous_real is None or shared < previous_prompt_blocks / 2
        new_blocks = prompt_blocks - shared
        # blocks of the previous call (prompt and output) this one does not keep
        lost = 0 if previous_real is None else len(previous_real) - shared
        previous_real, previous_prompt_blocks = real, prompt_blocks

        def fits(blocks: int) -> bool:
            return blocks * BLOCK_TOKENS + remainder + call.output <= compact_at

        if (not compacted or restart) and fits(prompt_blocks):
            compacted = False
            simulated = real
            out.append(call)
            continue
        if compacted and not restart:
            kept = max(len(simulated) - lost, 0)
            if fits(kept + new_blocks):
                prompt = np.concatenate([simulated[:kept], fresh.take(new_blocks)])
                stats["compacted_context"] += 1
                simulated = _emit(call, prompt, output_blocks, remainder, fresh, out)
                continue
        # compact: system prompt, summary, then the new tokens that fit
        head = real[: min(keep_blocks, prompt_blocks)]
        room = (compact_at - remainder - call.output) // BLOCK_TOKENS - len(head) - summary_blocks
        if room < 0:
            # the next call starts over
            stats["dropped_output_too_long"] += 1
            previous_real, compacted = None, False
            continue
        stats["compactions"] += 1
        stats["compaction_new_tokens_cut"] += max(new_blocks - room, 0) * BLOCK_TOKENS
        prompt = np.concatenate([head, fresh.take(summary_blocks + min(new_blocks, room))])
        compacted = True
        simulated = _emit(call, prompt, output_blocks, remainder, fresh, out)
    return out


def _emit(call: Call, prompt_ids, output_blocks, remainder, fresh, out) -> np.ndarray:
    call.prompt = len(prompt_ids) * BLOCK_TOKENS + remainder
    call.ids = np.concatenate([prompt_ids, fresh.take(output_blocks)])
    out.append(call)
    return call.ids


def shared_head_blocks(first_prompts: dict, limit_blocks: int) -> dict:
    """Per session, how many leading blocks of its first prompt another
    session's first prompt also has (its system prompt and tools, as far as
    the trace shows): with chained ids, block j is shared exactly when its id
    appears in another session's first prompt."""
    from collections import Counter

    counts: Counter = Counter()
    for ids in first_prompts.values():
        counts.update(ids[:limit_blocks].tolist())
    out = {}
    for session, ids in first_prompts.items():
        head = ids[:limit_blocks]
        shared = np.fromiter((counts[i] > 1 for i in head.tolist()), dtype=bool, count=len(head))
        out[session] = int(np.argmin(shared)) if not shared.all() else len(head)
    return out


def finalize_trace(
    raw_path: str,
    output: str,
    max_tokens: Optional[int] = 131072,
    compact_at: Optional[int] = 120_000,
    compact_to: int = 16_384,
    keep_blocks: Optional[dict] = None,
) -> "Counter":
    """A converter's raw session trace -> the built file.

    Every call gets output ids (add_output_ids). With compact_at, sessions
    are compacted to fit (compact_session; compact_at <= max_tokens), keeping
    per session keep_blocks[session_id] leading blocks (default: those its
    first prompt shares with another session's, shared_head_blocks). Without
    compact_at, calls over max_tokens are dropped instead.
    """
    from collections import Counter

    import pandas as pd

    from frontier.utils.block_hash_ids import decode_block_hash_ids

    stats: Counter = Counter()
    frame = pd.read_csv(raw_path, dtype={"block_hash_ids": str})
    frame = frame.sort_values(["session_id", "step"], kind="stable").reset_index(drop=True)
    with_arrivals = "arrived_at" in frame
    if compact_at is not None:
        if max_tokens is not None:
            compact_at = min(compact_at, max_tokens)
        if keep_blocks is None:
            firsts = frame.groupby("session_id", sort=False).head(1)
            keep_blocks = shared_head_blocks(
                {
                    s: decode_block_hash_ids(t)[: p // BLOCK_TOKENS]
                    for s, t, p in zip(firsts["session_id"], firsts["block_hash_ids"], firsts["num_prefill_tokens"])
                },
                compact_to // BLOCK_TOKENS,
            )
    fresh = FreshIds()
    placeholders = PlaceholderIds()
    writer = SessionTraceWriter(output, with_arrivals=with_arrivals)
    bounds = np.flatnonzero(np.r_[True, frame["session_id"].to_numpy()[1:] != frame["session_id"].to_numpy()[:-1], True])
    records = frame.to_dict("records")
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        calls = []
        for row in records[lo:hi]:
            ids = decode_block_hash_ids(row["block_hash_ids"])
            prompt = int(row["num_prefill_tokens"])
            covered = bool(row.get("ids_cover_output", 0) == 1)
            if not covered:
                ids = ids[: prompt // BLOCK_TOKENS]  # drop placeholders
            elif prompt + int(row["num_decode_tokens"]) < BLOCK_TOKENS:
                ids = ids[:0]
            calls.append(Call(row, prompt, int(row["num_decode_tokens"]), ids))
        add_output_ids(calls, fresh)
        stats["calls"] += len(calls)
        if compact_at is not None:
            session = records[lo]["session_id"]
            calls = compact_session(calls, keep_blocks.get(session, 0), compact_at, compact_to, fresh, stats)
        elif max_tokens is not None:
            kept = [c for c in calls if c.prompt + c.output <= max_tokens]
            stats["dropped_too_long"] += len(calls) - len(kept)
            calls = kept
        for call in calls:
            row = call.row
            writer.write(
                session_id=int(row["session_id"]),
                step=int(row["step"]),
                num_prefill_tokens=call.prompt,
                num_decode_tokens=call.output,
                block_hash_ids=encode_block_hash_ids(call.ids) if len(call.ids) else placeholders.next(),
                arrived_at=row.get("arrived_at") if with_arrivals else None,
                completed_at=(
                    None if not with_arrivals or pd.isna(row.get("completed_at")) else row["completed_at"]
                ),
                ids_cover_output=True,
            )
            stats["kept"] += 1
    writer.close()
    return stats
