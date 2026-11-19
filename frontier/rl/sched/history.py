"""
Request histories, sliding-window system stats and oracle job sizes, from
Vidur's RLVLLMReplicaScheduler and priority_policy.py (the stack's features and
SJF baselines read them; Frontier runs keep them in FrontierFeatureScheduler).
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

# Sliding windows (in scheduling iterations) for the system history the policy
# can observe, after Mutant's short/medium/long observation windows.
WINDOW_SIZES = {"short": 10, "medium": 100, "long": 1000}
WINDOW_METRICS = (
    "queue_length",
    "free_block_fraction",
    "batch_num_tokens",
    "batch_time",
)


@dataclass
class RequestHistory:
    """What happened to a request while it waited, relative to other requests."""

    # policy decisions made while the request was waiting
    num_decisions_seen: int = 0
    # iterations in which other waiting requests were admitted but it was not
    times_passed_over: int = 0
    # requests that arrived after it but were admitted while it waited
    num_overtaken_by: int = 0
    # position in the last effective ranking, 0 = front, 1 = back
    last_rank_percentile: Optional[float] = None
    # last priority the policy assigned; persists between decisions in
    # arrival_and_periodic mode
    priority: Optional[int] = None


class SlidingWindow:
    """Mean, min and max of the last `size` samples in O(1) amortized time."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._values: deque = deque()
        self._sum = 0.0
        self._count = 0
        # monotonic queues of (sample index, value)
        self._min_queue: deque = deque()
        self._max_queue: deque = deque()

    def add(self, value: float) -> None:
        index = self._count
        self._count += 1

        self._values.append(value)
        self._sum += value
        if len(self._values) > self._size:
            self._sum -= self._values.popleft()

        while self._min_queue and self._min_queue[-1][1] >= value:
            self._min_queue.pop()
        self._min_queue.append((index, value))
        while self._max_queue and self._max_queue[-1][1] <= value:
            self._max_queue.pop()
        self._max_queue.append((index, value))

        oldest = index - self._size + 1
        while self._min_queue[0][0] < oldest:
            self._min_queue.popleft()
        while self._max_queue[0][0] < oldest:
            self._max_queue.popleft()

    def stats(self) -> Tuple[float, float, float]:
        if not self._values:
            return 0.0, 0.0, 0.0
        return (
            self._sum / len(self._values),
            self._min_queue[0][1],
            self._max_queue[0][1],
        )



# prefill_tokens is the only key that does not read the output length
JOB_SIZE_KEYS = ("total_tokens", "decode_tokens", "iterations", "prefill_tokens")


def job_size(request, key: str, max_num_batched_tokens: int) -> int:
    """
    Remaining work of a waiting request, using its true output length (an
    oracle; the RL observation never exposes it), except for prefill_tokens.

      total_tokens   prefill + decode tokens
      decode_tokens  output tokens still to generate
      iterations     scheduler iterations it needs on its own: chunked prefill
                     iterations plus one per output token
      prefill_tokens prompt tokens still to compute (no oracle: a server knows
                     them), i.e. uncached-prompt SJF with a prefix cache

    A preempted request restarts with its generated tokens folded into the
    prompt, so these stay the remaining work after a restart.

    Prompt tokens a prefix cache already holds (num_cached_prefill_tokens, set
    by frontier/rl/sched/frontier_env.py for Frontier's cache; 0 for Vidur requests)
    are not work: a 120k prompt with 119k cached is a 1k-token prefill.
    """
    prefill = request.num_prefill_tokens - getattr(request, "num_cached_prefill_tokens", 0)
    if key == "total_tokens":
        return prefill + request.num_decode_tokens
    if key == "decode_tokens":
        return request.num_decode_tokens
    if key == "prefill_tokens":
        return max(prefill, 0)
    if key == "iterations":
        prefill_iterations = -(-prefill // max_num_batched_tokens)
        return prefill_iterations + request.num_decode_tokens
    if key == "prefill_tokens":
        return request.num_prefill_tokens
    raise ValueError(f"Unknown job size key {key}. Valid keys: {JOB_SIZE_KEYS}")
