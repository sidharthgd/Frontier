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
    "cached_prompt_tokens",   # prefix-cache hit: counted so far, or for a waiting
                              # request the hit it would get if admitted now
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


def cached_prefix_tokens(scheduler, request: Request) -> int:
    """Prompt tokens the scheduler's prefix cache holds for a waiting request
    (0 without prefix caching)."""
    return prefix_cache_state(scheduler, request)[0]


def prefix_cache_state(scheduler, request: Request) -> Tuple[int, int]:
    """(cached prompt tokens, idle cached blocks) of a waiting request
    (KVCacheManager.peek_prefix_cache; (0, 0) without prefix caching)."""
    return tuple(prefix_cache_view(scheduler, request)[:2])


def prefix_cache_view(scheduler, request: Request) -> Tuple[int, int, int]:
    """(cached prompt tokens, idle cached blocks, eviction distance in
    blocks, -1 for none) of a waiting request: the RL scheduler's
    cache_state, or the KV manager's peek for other schedulers."""
    if hasattr(scheduler, "cache_state"):
        return scheduler.cache_state(request)
    manager = getattr(scheduler, "_kv_cache_manager", None)
    if manager is None:
        return (0, 0, -1)
    return (*manager.peek_prefix_cache(request), -1)


# --- SJF job sizes, shared by both RL stacks' baselines ------------------------


def prefill_target_tokens(request: Request) -> int:
    """Tokens a waiting request must have computed before it decodes again:
    its prompt, or for a victim preempted after prefill its prompt plus the
    output it kept (vLLM v1 recomputes both; prefix_cache_view looks up the
    same span)."""
    if request.is_recomputing:
        return int(request.num_processed_tokens)
    return int(request.num_prefill_tokens)


def prefill_work_tokens(request: Request, cached_tokens: int) -> int:
    """The prefill a waiting request needs: prefill_target_tokens less what
    the prefix cache holds for it (cached_tokens, prefix_cache_view's first
    entry). A real server knows it (vLLM: request.num_tokens less its
    prefix-cache hit), so it is no oracle."""
    return max(prefill_target_tokens(request) - int(cached_tokens), 0)


def job_size_tokens(request: Request, cached_tokens: int, with_output: bool) -> int:
    """SJF job size: prefill_work_tokens, plus, with_output (an oracle), the
    output tokens still to generate. prefill-only SJF (no oracle) and total
    SJF in frontier/rl/agents.py and frontier/rl/sched (via
    sched/frontier_env.py's _OracleRequest) rank by these."""
    size = prefill_work_tokens(request, cached_tokens)
    if with_output:
        size += int(request.remaining_decode_tokens)
    return size


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
                max(r.num_prefill_tokens_cached, cached_prefix_tokens(scheduler, r)),
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
