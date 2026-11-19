"""ShareGPT conversations -> multi-turn session trace with real block hashes.

Source: ShareGPT_V3_unfiltered_cleaned_split.json from the
anon8231489123/ShareGPT_Vicuna_unfiltered HF dataset. Needs `tokenizers`.

data/processed_traces/sharegpt_llama3_tokenizer.csv keeps only the first
user turn of each conversation, as in Fu et al. (arXiv 2408.15792), so it has
nothing for a prefix cache to reuse. Here every turn is a request, as a chat
client sends it: turn k's prompt is the Llama-3 chat template of the whole
conversation so far,

    <|begin_of_text|> (header(user) u_0 <|eot_id|> header(assistant) a_0 <|eot_id|>)...
                      header(user) u_k <|eot_id|> header(assistant)

and its output is the recorded reply a_k (turns with an empty reply end
the conversation). Turn k+1's prompt starts with turn
k's prompt token for token, so it can reuse it. Turn 0 is exactly a row of
the first-turn CSV (same tokenizer, template and filters).

Block ids are content hashes: each full 16-token block is hashed chained to
its prefix (blake2b), and each distinct hash gets a sequential id in order of
first appearance, so equal prefixes in different conversations share blocks,
and a prompt's ids are a few consecutive runs.

A conversation stops at its first malformed pair (not user then assistant,
or an empty side) or at the first turn whose prompt + output exceeds
max_tokens. There are no timestamps: frontier.trace_prep.sessions gives
turns arrival gaps.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from frontier.trace_prep.common import (
    BLOCK_TOKENS,
    BlockRuns,
    PlaceholderIds,
    SessionTraceWriter,
)

DEFAULT_TOKENIZER = "NousResearch/Meta-Llama-3-8B"  # ungated Llama-3 tokenizer
BEGIN_OF_TEXT = "<|begin_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
END_OF_TURN = "<|eot_id|>"

USER_ROLES = {"human", "user"}
ASSISTANT_ROLES = {"gpt", "chatgpt", "bard", "bing", "assistant", "model"}


def load_tokenizer(name: str = DEFAULT_TOKENIZER):
    import os

    from tokenizers import Tokenizer

    return Tokenizer.from_file(name) if os.path.exists(name) else Tokenizer.from_pretrained(name)


def conversation_pairs(turns: List[dict]) -> List[Tuple[str, str]]:
    """Leading (user, assistant) pairs, up to the first malformed one."""
    pairs = []
    for i in range(0, len(turns) - 1, 2):
        user, assistant = turns[i], turns[i + 1]
        if user.get("from") not in USER_ROLES or assistant.get("from") not in ASSISTANT_ROLES:
            break
        if not user.get("value") or not assistant.get("value"):
            break
        pairs.append((user["value"], assistant["value"]))
    return pairs


class _Template:
    def __init__(self, tokenizer) -> None:
        def special(token: str) -> int:
            token_id = tokenizer.token_to_id(token)
            assert token_id is not None, token
            return token_id

        def plain(text: str) -> List[int]:
            return tokenizer.encode(text, add_special_tokens=False).ids

        self.bos = [special(BEGIN_OF_TEXT)]
        self.eot = [special(END_OF_TURN)]
        newline = plain("\n\n")
        self.user_header = [special(START_HEADER)] + plain("user") + [special(END_HEADER)] + newline
        self.assistant_header = (
            [special(START_HEADER)] + plain("assistant") + [special(END_HEADER)] + newline
        )


class _BlockIds:
    """Sequential ids for chained 16-token block hashes, shared across
    conversations."""

    def __init__(self) -> None:
        self.ids: Dict[bytes, int] = {}

    def of(self, tokens: np.ndarray) -> BlockRuns:
        runs = BlockRuns()
        digest = b""
        for start in range(0, len(tokens) - BLOCK_TOKENS + 1, BLOCK_TOKENS):
            block = tokens[start : start + BLOCK_TOKENS].astype("<u4").tobytes()
            digest = hashlib.blake2b(digest + block, digest_size=12).digest()
            block_id = self.ids.setdefault(digest, len(self.ids))
            runs.extend(block_id, 1)
        return runs


def iter_conversations(path: str) -> Iterator[List[Tuple[str, str]]]:
    with open(path) as file:
        data = json.load(file)
    for item in data:
        pairs = conversation_pairs(item.get("conversations") or [])
        if pairs:
            yield pairs


def convert(
    path: str,
    output: str,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    max_tokens: Optional[int] = 4096,
    max_conversations: Optional[int] = None,
    batch_size: int = 2048,
) -> Counter:
    tokenizer = load_tokenizer(tokenizer_name)
    template = _Template(tokenizer)
    block_ids = _BlockIds()
    writer = SessionTraceWriter(output, with_arrivals=False)
    placeholders = PlaceholderIds()
    stats: Counter = Counter()

    def flush(batch: List[List[Tuple[str, str]]]) -> None:
        texts = [text for pairs in batch for pair in pairs for text in pair]
        encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
        cursor = 0
        for pairs in batch:
            session_id = stats["conversations"]
            stats["conversations"] += 1
            history: List[int] = list(template.bos)
            kept = 0
            for step in range(len(pairs)):
                user = encoded[cursor + 2 * step].ids
                reply = encoded[cursor + 2 * step + 1].ids
                prompt = history + template.user_header + user + template.eot + template.assistant_header
                if max_tokens is not None and len(prompt) + len(reply) > max_tokens:
                    stats["turns_dropped_too_long"] += len(pairs) - step
                    break
                # the reply's full blocks too: the next prompt holds them
                # verbatim, and vLLM caches generated blocks
                runs = block_ids.of(np.asarray(prompt + reply, dtype=np.int64))
                stats["turns_short"] += runs.num_blocks == 0
                writer.write(
                    session_id=session_id,
                    step=step,
                    num_prefill_tokens=len(prompt),
                    num_decode_tokens=len(reply),
                    block_hash_ids=runs.encode() if runs.num_blocks else placeholders.next(),
                    ids_cover_output=True,
                )
                kept += 1
                history = prompt + reply + template.eot
            stats["turns"] += kept
            stats["sessions_kept"] += kept > 0
            cursor += 2 * len(pairs)

    batch: List[List[Tuple[str, str]]] = []
    for pairs in iter_conversations(path):
        batch.append(pairs)
        if len(batch) == batch_size:
            flush(batch)
            batch = []
        if max_conversations and stats["conversations"] + len(batch) >= max_conversations:
            break
    if batch:
        flush(batch)
    writer.close()
    stats["distinct_blocks"] = len(block_ids.ids)
    return stats
