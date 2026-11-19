"""Compact text form of a request's block hash ids, for trace files.

A trace names each full KV block of a prompt by a chained hash id: two
prompts share their first k blocks exactly when their first k ids are equal.
Agentic prompts run to ~10k blocks of 16 tokens, and traces such as
FreeInference number new blocks consecutively, so a prompt's ids are a few
runs of consecutive integers. They are written as `|`-separated runs, each
`a-b` (a, a+1, ..., b) or a single `a`:

    "0-1706|912188|912190-912254"

Plain `|`- or `,`-separated ids are the special case of single-id runs, so
older trace files read the same.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def encode_block_hash_ids(ids: Sequence[int] | np.ndarray) -> str:
    """Runs of consecutive ids as "a-b|c|d-e"; empty string for no ids."""
    values = np.asarray(ids, dtype=np.int64)
    if values.size == 0:
        return ""
    breaks = np.flatnonzero(np.diff(values) != 1) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [values.size])) - 1
    return "|".join(
        str(values[s]) if s == e else f"{values[s]}-{values[e]}"
        for s, e in zip(starts.tolist(), ends.tolist())
    )


def decode_block_hash_ids(text: str) -> np.ndarray:
    """Inverse of encode_block_hash_ids; also reads "[1, 2]", "1|2" and "1,2"."""
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
    delimiter = "|" if "|" in stripped else ","
    parts = [part.strip() for part in stripped.split(delimiter) if part.strip()]
    chunks = []
    for part in parts:
        # a leading "-" is a sign, not a range
        dash = part.find("-", 1)
        if dash < 0:
            chunks.append(np.array([int(part)], dtype=np.int64))
            continue
        first, last = int(part[:dash]), int(part[dash + 1 :])
        if last < first:
            raise ValueError(f"Block hash id range {part!r} runs backwards.")
        chunks.append(np.arange(first, last + 1, dtype=np.int64))
    if not chunks:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(chunks)
