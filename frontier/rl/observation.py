"""
What the agent sees at a decision. Raw values only: scaling and log
transforms belong to the model. Features never read a request's output length
(num_decode_tokens or anything derived from it); a real server does not know
it. Oracle baselines get the Request objects separately (env.OracleAgent).
"""

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from frontier.entities import Request

if TYPE_CHECKING:
    from frontier.rl.scheduler import RLVLLMv1EngineReplicaScheduler

WAITING_FEATURES: Tuple[str, ...] = (
    "prompt_tokens",          # prompt of the current round (grows after a restart)
    "processed_tokens",       # already computed (chunked prefill progress)
    "cached_prompt_tokens",   # prefix-cache hit counted so far
    "wait_time",              # seconds since arrival
    "num_preemptions",        # times evicted so far
    "blocks_needed",          # KV blocks for the whole prompt
    "blocks_needed_over_free",
    "round_index",            # thinking / agent round, 0 for single-round requests
    "num_rounds",
    "current_priority",
)
RUNNING_FEATURES: Tuple[str, ...] = (
    "prompt_tokens",
    "processed_tokens",
    "emitted_tokens",         # output tokens so far
    "age",                    # seconds since arrival
    "is_prefill_complete",
    "current_priority",
)
GLOBAL_FEATURES: Tuple[str, ...] = (
    "free_block_fraction",
    "num_waiting",
    "num_running",
    "max_tokens_in_batch",
    "batch_size_cap",
)


@dataclass
class Observation:
    time: float
    num_priority_levels: int
    # one entry per waiting request, in the order of the action
    request_ids: np.ndarray
    arrival_times: np.ndarray
    session_ids: np.ndarray  # -1 when the request has no session
    waiting: np.ndarray  # [num_waiting, len(WAITING_FEATURES)]
    running: np.ndarray  # [num_running, len(RUNNING_FEATURES)]
    global_features: np.ndarray  # [len(GLOBAL_FEATURES)]

    @property
    def num_requests(self) -> int:
        return len(self.request_ids)

    def waiting_feature(self, name: str) -> np.ndarray:
        return self.waiting[:, WAITING_FEATURES.index(name)]

    def global_feature(self, name: str) -> float:
        return float(self.global_features[GLOBAL_FEATURES.index(name)])


def _round(request: Request) -> Tuple[int, int]:
    if request.is_thinking_mode_enabled:
        return request.current_thinking_round_index, request.thinking_depth
    return 0, 1


def build_observation(
    time: float,
    waiting: List[Request],
    scheduler: "RLVLLMv1EngineReplicaScheduler",
    num_priority_levels: int,
) -> Observation:
    block_size = scheduler.block_size
    free = max(1, scheduler.num_free_blocks)
    rows = []
    for r in waiting:
        blocks = ceil(r.num_prefill_tokens / block_size)
        round_index, num_rounds = _round(r)
        rows.append(
            (
                r.num_prefill_tokens,
                r.num_processed_tokens,
                r.num_prefill_tokens_cached,
                time - r.arrived_at,
                r.get_total_preemption_count(),
                blocks,
                blocks / free,
                round_index,
                num_rounds,
                r.priority,
            )
        )
    running_rows = [
        (
            r.num_prefill_tokens,
            r.num_processed_tokens,
            r.num_processed_decode_tokens,
            time - r.arrived_at,
            float(r.is_prefill_complete),
            r.priority,
        )
        for r in scheduler.running_requests
    ]
    global_features = (
        scheduler.num_free_blocks / max(1, scheduler.num_blocks),
        len(waiting),
        len(running_rows),
        scheduler.max_tokens_in_batch,
        scheduler.batch_size_cap,
    )
    return Observation(
        time=time,
        num_priority_levels=num_priority_levels,
        request_ids=np.array([r.id for r in waiting], dtype=np.int64),
        arrival_times=np.array([r.arrived_at for r in waiting], dtype=np.float64),
        session_ids=np.array(
            [r.session_id if r.session_id is not None else -1 for r in waiting],
            dtype=np.int64,
        ),
        waiting=np.array(rows, dtype=np.float64).reshape(len(rows), len(WAITING_FEATURES)),
        running=np.array(running_rows, dtype=np.float64).reshape(
            len(running_rows), len(RUNNING_FEATURES)
        ),
        global_features=np.array(global_features, dtype=np.float64),
    )
