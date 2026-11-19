"""FreeInference agentic trace -> session trace.

Source: huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace
(CC BY 4.0), release/<week>/traces/<day>.jsonl.zst, one agent session per
line. Each LLM request carries `block_ids`, its input as chained 16-token
blocks (o200k_base tokenizer; partial last block dropped), which is
Frontier's block size, so the ids are used as they are. Ids are local to a
week, so each week gets its own namespace.

Per request:
    arrived_at          start_ms (gateway), seconds from the first request
    num_prefill_tokens  sum of role_tokens (same tokenizer as block_ids),
                        made consistent with len(block_ids)
    num_decode_tokens   provider_tokens.completion (provider's tokenizer)
Subagent sessions nested under tool calls become sessions of their own.

The raw trace then goes through common.finalize_trace: outputs get the ids
of the next prompt's blocks they become, and sessions whose context outgrows
the model (the median request is ~140k tokens; 131072 for Llama-3.1-8B,
data/profiling/compute/a100/llama3.1-8b) are compacted, keeping the leading
blocks their first prompt shares with other sessions (system prompt, tools).
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
import shutil
import subprocess
from collections import Counter
from typing import IO, Iterator, List, Optional

import numpy as np

from frontier.trace_prep.common import (
    PlaceholderIds,
    SessionTraceWriter,
    check_local_ids,
    namespace_base,
    prompt_tokens_for_blocks,
)
from frontier.utils.block_hash_ids import encode_block_hash_ids

logger = logging.getLogger(__name__)

FIRST_WEEK = dt.date(2026, 5, 17)


def week_index(week: str) -> int:
    """Stable namespace for "YYYY-MM-DD_YYYY-MM-DD" (weeks since the first)."""
    start = dt.date.fromisoformat(week.split("_")[0])
    return (start - FIRST_WEEK).days // 7


def _open_text(path: str) -> IO[str]:
    if not path.endswith(".zst"):
        return open(path, "r")
    try:
        import zstandard

        reader = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
        return io.TextIOWrapper(reader, encoding="utf-8")
    except ImportError:
        if shutil.which("zstd") is None:
            raise RuntimeError("reading .zst needs `pip install zstandard` or the zstd CLI")
        process = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE)
        return io.TextIOWrapper(process.stdout, encoding="utf-8")


def iter_sessions(paths: List[str]) -> Iterator[dict]:
    for path in paths:
        with _open_text(path) as file:
            for line in file:
                if line.strip():
                    yield json.loads(line)


def flatten(session: dict) -> Iterator[dict]:
    """The session, then every subagent session nested under its tool calls."""
    yield session
    for request in session.get("requests") or []:
        for call in request.get("tool_calls") or []:
            subagent = call.get("subagent") if isinstance(call, dict) else None
            if isinstance(subagent, dict) and subagent.get("requests"):
                subagent.setdefault("week", session.get("week"))
                yield from flatten(subagent)


def convert(
    paths: List[str],
    output: str,
    max_tokens: Optional[int] = 131072,
    harnesses: Optional[List[str]] = None,
    compact_at: Optional[int] = 120_000,
    compact_to: int = 16_384,
) -> Counter:
    import os

    from frontier.trace_prep.common import finalize_trace

    stats: Counter = Counter()
    raw = f"{output}.raw.{os.getpid()}.csv"
    writer = SessionTraceWriter(raw)
    placeholders = PlaceholderIds()
    next_session_id = 0
    for top in iter_sessions(paths):
        if harnesses and top.get("harness") not in harnesses:
            stats["sessions_other_harness"] += 1
            continue
        for session in flatten(top):
            session_id = next_session_id
            next_session_id += 1
            stats["sessions"] += 1
            base = namespace_base(week_index(session["week"]))
            for step, request in enumerate(session["requests"]):
                stats["requests"] += 1
                ids = np.asarray(request.get("block_ids") or [], dtype=np.int64)
                completion = (request.get("provider_tokens") or {}).get("completion")
                role_tokens = request.get("role_tokens") or {}
                prompt = prompt_tokens_for_blocks(
                    len(ids), sum(role_tokens.values()) if role_tokens else None
                )
                if prompt <= 0 or not completion:
                    stats["dropped_empty"] += 1
                    continue
                check_local_ids(ids)
                writer.write(
                    arrived_at=request["start_ms"] / 1000.0,
                    completed_at=(
                        request["end_ms"] / 1000.0 if request.get("end_ms") else None
                    ),
                    session_id=session_id,
                    step=step,
                    num_prefill_tokens=prompt,
                    num_decode_tokens=int(completion),
                    block_hash_ids=(
                        encode_block_hash_ids(ids + base) if ids.size else placeholders.next()
                    ),
                )
    writer.close()
    try:
        stats.update(finalize_trace(raw, output, max_tokens, compact_at, compact_to))
    finally:
        os.remove(raw)
    return stats
