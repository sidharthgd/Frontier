"""
The RL hook into Frontier's vLLM v1 scheduler.

RLVLLMv1EngineReplicaScheduler is VLLMv1EngineReplicaScheduler with one extra
step: just before the waiting queue is sorted for admission, a priority policy
assigns every waiting request an integer priority (lower runs first). The
queue is then sorted the way vLLM's --scheduling-policy priority sorts it,
(priority, arrival, id), and vLLM's own admission code takes requests from the
front until the token budget, batch cap or KV memory runs out.

It is installed under the vllm_v1 registry key (see rl_scheduler_installed), so
every isinstance check and config path in Frontier treats it as the stock
scheduler. With no policy attached it behaves exactly like the parent.

As in vLLM's priority mode, priorities also decide preemption: the running
request with the largest (priority, arrival) is evicted first. Priorities
stick to a request until the policy changes them.
"""

from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

from frontier.entities import Batch, Request
from frontier.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ReplicaSchedulerType


class PriorityPolicy:
    """
    What the scheduler calls. get_priorities returns one priority per request
    in `waiting` (same order), or None to keep the current priorities.
    on_admission is called after every get_priorities call with the requests
    that were admitted and the first request in ranked order that was not
    (None if all were). Admission takes a prefix of the ranking, so that
    request is the boundary between admitted and not.
    """

    def get_priorities(
        self, waiting: List[Request], scheduler: "RLVLLMv1EngineReplicaScheduler"
    ) -> Optional[Sequence[int]]:
        return None

    def on_admission(
        self, admitted: List[Request], boundary: Optional[Request]
    ) -> None:
        pass

    def on_batch_end(self, batch: Batch) -> None:
        pass


class RLVLLMv1EngineReplicaScheduler(VLLMv1EngineReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._priority_policy: Optional[PriorityPolicy] = None
        self._ranked_waiting: List[Request] = []

    def set_priority_policy(self, policy: Optional[PriorityPolicy]) -> None:
        if policy is not None:
            assert self._scheduling_policy == "priority", (
                "the RL scheduler needs "
                "--vllm_v1_scheduler_config_scheduling_policy priority"
            )
        self._priority_policy = policy

    # -- state the policy and observation read ------------------------------

    @property
    def num_blocks(self) -> int:
        return int(self._config.num_blocks)

    @property
    def num_free_blocks(self) -> int:
        return int(self._config.num_blocks - self._num_allocated_blocks)

    @property
    def block_size(self) -> int:
        return int(self._config.block_size)

    @property
    def max_tokens_in_batch(self) -> int:
        return int(self._max_num_scheduled_tokens)

    @property
    def batch_size_cap(self) -> int:
        return int(self._max_num_running_reqs)

    @property
    def running_requests(self) -> List[Request]:
        return list(self._running_requests)

    @property
    def prefix_caching_enabled(self) -> bool:
        return bool(self._is_prefix_caching_enabled())

    # -- the hook -----------------------------------------------------------

    def _get_sorted_waiting_queue(self) -> List[Request]:
        if self._priority_policy is not None:
            waiting = self._preempted_requests + self._request_queue
            priorities = self._priority_policy.get_priorities(waiting, self)
            if priorities is not None:
                assert len(priorities) == len(waiting)
                for request, priority in zip(waiting, priorities):
                    request.set_priority(int(priority))
        ranked = super()._get_sorted_waiting_queue()
        self._ranked_waiting = list(ranked)
        return ranked

    def _schedule_waiting_requests(
        self, token_budget: int
    ) -> Tuple[int, List[Request], List[int]]:
        self._ranked_waiting = []
        token_budget, scheduled, num_tokens = super()._schedule_waiting_requests(
            token_budget
        )
        if self._priority_policy is not None and self._ranked_waiting:
            admitted = set(id(r) for r in scheduled)
            boundary = next(
                (r for r in self._ranked_waiting if id(r) not in admitted), None
            )
            self._priority_policy.on_admission(scheduled, boundary)
        return token_budget, scheduled, num_tokens

    def on_batch_end(self, batch: Batch) -> None:
        super().on_batch_end(batch)
        if self._priority_policy is not None:
            self._priority_policy.on_batch_end(batch)


@contextmanager
def rl_scheduler_installed() -> Iterator[None]:
    """Build schedulers inside this block to get the RL scheduler for vllm_v1."""
    key = ReplicaSchedulerType.VLLM_V1
    original = ReplicaSchedulerRegistry.get_class(key)
    ReplicaSchedulerRegistry._registry[key] = RLVLLMv1EngineReplicaScheduler
    try:
        yield
    finally:
        ReplicaSchedulerRegistry._registry[key] = original


def replica_schedulers(simulator) -> List[RLVLLMv1EngineReplicaScheduler]:
    """Every RL replica scheduler in a built Simulator."""
    found = []
    for cluster_scheduler in simulator._global_scheduler._cluster_schedulers.values():
        for scheduler in cluster_scheduler._replica_schedulers.values():
            if isinstance(scheduler, RLVLLMv1EngineReplicaScheduler):
                found.append(scheduler)
    return found
