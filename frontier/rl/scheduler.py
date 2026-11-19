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
from typing import Iterator, List, Optional, Sequence, Tuple

from frontier.entities import Batch, Request
from frontier.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ReplicaSchedulerType


class _FreeOrder:
    """Sequence numbers of the blocks in the free queue, with a Fenwick tree
    of which are still queued (from agent-sessions-local)."""

    def __init__(self) -> None:
        self._size = 1 << 16
        self._tree = [0] * (self._size + 1)
        self._queued = bytearray(self._size)
        self._next = 0

    def _grow(self) -> None:
        self._size *= 2
        self._queued.extend(bytearray(self._size - len(self._queued)))
        self._tree = [0] * (self._size + 1)
        for i in range(self._size):
            if self._queued[i]:
                self._update(i, 1)

    def _update(self, seq: int, delta: int) -> None:
        i = seq + 1
        while i <= self._size:
            self._tree[i] += delta
            i += i & -i

    def add(self, block) -> None:
        if self._next >= self._size:
            self._grow()
        block._free_seq = self._next
        self._queued[self._next] = 1
        self._update(self._next, 1)
        self._next += 1

    def discard(self, block) -> None:
        seq = getattr(block, "_free_seq", None)
        if seq is not None and self._queued[seq]:
            self._queued[seq] = 0
            self._update(seq, -1)

    def ahead(self, seq: int) -> int:
        """Blocks queued with a smaller sequence number: those evicted before this one."""
        total, i = 0, seq
        while i > 0:
            total += self._tree[i]
            i -= i & -i
        return total


# How the RL environment (frontier/rl/frontier_env.py) uses the hook: every non-empty waiting queue gets
# priorities, even a single request, because a priority outlives admission:
# priority-mode preemption evicts the running request with the largest
# (priority, arrival), and a request never ranked keeps the default 0, which
# would shield it from preemption. A ranking only becomes an RL decision
# (transition) when at least MIN_WAITING_TO_RECORD requests wait: a lone
# request has nothing to be reordered against. A ranking that admitted
# nothing is undone (no effect on admission, so none on preemption either).
MIN_WAITING_TO_RECORD = 2


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

    # Whether the workload's block ids are chained (frontier/trace_prep's
    # are): cache lookups for the policy can then binary search
    # (KVCacheManager.peek_prefix_cache).
    chained_block_hashes = False

    def _install_free_order(self) -> _FreeOrder:
        """
        Track the free queue's order exactly. Each block gets a sequence number
        when it enters the queue (the queue is FIFO, so the order is the
        sequence), and a Fenwick tree counts the sequence numbers still queued.
        The number of blocks ahead of a block, i.e. how many allocations until
        it is evicted, is then one prefix sum: O(log blocks) per queue
        operation and per query.
        """
        queue = self._kv_cache_manager.block_pool.free_block_queue
        order = getattr(queue, "_rl_order", None)
        if order is not None:
            return order
        order = _FreeOrder()
        block = queue.free_list_head
        while block is not None:
            order.add(block)
            block = block.next_free_block
        append, remove = queue.append, queue.remove

        def tracked_append(block):
            append(block)
            order.add(block)

        def tracked_remove(block):
            order.discard(block)
            remove(block)

        # popleft goes through remove
        queue.append, queue.remove = tracked_append, tracked_remove
        queue._rl_order = order
        return order

    def cache_state(self, request: Request) -> Tuple[int, int, int]:
        """
        (cached prompt tokens, idle cached blocks, eviction distance) of a
        waiting request, read without touching hit statistics or eviction
        order (KVCacheManager.peek_prefix_cache for the first two). The
        eviction distance is how many blocks will be allocated before its hit
        loses a block: the queue position of its idle hit block that is
        evicted first (-1: no idle hit blocks, or no prefix caching).

        With chained ids that block is the hit's last: requests hold their
        blocks from block 0 and free them tail first, so a later block of a
        chain is never queued after an earlier one. Otherwise every idle hit
        block is checked.
        """
        manager = getattr(self, "_kv_cache_manager", None)
        if manager is None or not self.prefix_caching_enabled:
            return 0, 0, -1
        tokens, idle = manager.peek_prefix_cache(request, chained=self.chained_block_hashes)
        if not idle:
            return tokens, 0, -1
        order = self._install_free_order()
        pool = manager.block_pool
        hashes = request.lookup_block_hash_array(self.block_size)[: tokens // self.block_size]
        if self.chained_block_hashes:
            first_out = pool.get_cached_block(int(hashes[-1]))._free_seq
        else:
            first_out = self._first_evicted_scan(hashes)
        return tokens, idle, order.ahead(first_out)

    def _first_evicted_scan(self, hashes) -> int:
        """Smallest free-queue sequence number among the idle blocks."""
        pool = self._kv_cache_manager.block_pool
        seqs = [
            block._free_seq
            for block in (pool.get_cached_block(int(h)) for h in hashes)
            if block.ref_cnt == 0
        ]
        return min(seqs)

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
