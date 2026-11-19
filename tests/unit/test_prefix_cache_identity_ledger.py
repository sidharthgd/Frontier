"""Regression tests for Prefix cache physical-block identity evidence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.entities.request import Request
from frontier.kv_cache.base_kv_cache_manager import KVCacheManager
from frontier.kv_cache.kv_cache_block_pool import BlockPool
from frontier.scheduler.replica_scheduler import (
    vllm_v1_prefix_cache as prefix_cache_module,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


def _request(*, hashes: list[int], num_prefill_tokens: int = 4) -> Request:
    return Request(
        arrived_at=0.0,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=1,
        block_hash_ids=hashes,
    )


def _manager(num_gpu_blocks: int = 2) -> KVCacheManager:
    return KVCacheManager(
        block_size=2,
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=True,
        caching_hash_algo="builtin",
        num_preallocate_tokens=0,
    )


def test_allocation_result_records_binding_creator_and_epoch() -> None:
    manager = _manager()
    creator = _request(hashes=[11, 22])

    result = manager.allocate_slots(creator, 4)

    assert result is not None
    assert [block.block_id for block in result.new_blocks] == [0, 1]
    assert [
        (
            binding.block_hash,
            binding.block_id,
            binding.creator_request_id,
            binding.binding_epoch,
        )
        for binding in result.new_bindings
    ] == [
        (11, 0, creator.id, 1),
        (22, 1, creator.id, 1),
    ]
    assert result.evicted_bindings == ()


@pytest.mark.parametrize("num_tokens", (0, -1, 1.5, True))
def test_kv_cache_manager_rejects_invalid_allocation_tokens(num_tokens) -> None:
    manager = _manager()
    request = _request(hashes=[11, 22])

    with pytest.raises(ValueError, match="num_tokens must be a positive integer"):
        manager.allocate_slots(request, num_tokens)


@pytest.mark.parametrize("num_blocks", (-1, 1.5, True))
def test_block_pool_rejects_invalid_allocation_count(num_blocks) -> None:
    pool = BlockPool(num_gpu_blocks=2, enable_caching=True)

    with pytest.raises(
        ValueError,
        match="num_blocks must be a non-negative integer",
    ):
        pool.get_new_blocks(num_blocks)


def test_committed_full_hit_admission_records_reuse_eviction_and_rebinding(
    monkeypatch,
) -> None:
    manager = _manager()
    creator = _request(hashes=[11, 22])
    first_result = manager.allocate_slots(creator, 4)
    assert first_result is not None
    manager.free(creator)

    consumer = _request(hashes=[11, 22])
    scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    scheduler._kv_cache_manager = manager
    scheduler._config = SimpleNamespace(block_size=2, num_blocks=2)
    scheduler._cluster_type = ClusterType.MONOLITHIC
    scheduler._replica_id = 3
    scheduler._replica_local_id = None
    scheduler._active_schedule_iteration_id = 7
    scheduler._current_schedule_time = 1.25
    scheduler._prefix_cache_identity_event_seq = 0
    scheduler._allocation_map = {}
    scheduler._num_allocated_blocks = 0

    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        prefix_cache_module,
        "_log_frontier_vllm_v1_schedule_decision",
        lambda event: events.append(dict(event)),
    )

    admission = scheduler._prepare_prefix_cache_admission(consumer)

    assert [block.block_hash for block in admission.raw_hit_blocks] == [11, 22]
    assert [block.block_hash for block in admission.effective_hit_blocks] == [11]
    assert admission.raw_cached_tokens == 4
    assert admission.effective_cached_tokens == 2
    assert admission.num_new_tokens == 2
    assert admission.full_hit_backoff_applied is True

    allocation = scheduler._allocate_request(
        consumer,
        admission.num_new_tokens,
        new_computed_blocks=list(admission.effective_hit_blocks),
        prefix_cache_admission=admission,
    )

    assert allocation is not None
    assert [event["event"] for event in events] == [
        "prefix_cache_admission",
        "prefix_cache_allocation",
    ]
    admission_event, allocation_event = events
    assert admission_event["replica_local_id"] is None
    assert allocation_event["replica_local_id"] is None
    assert admission_event["identity_event_seq"] == 0
    assert allocation_event["identity_event_seq"] == 1
    assert admission_event["query_hashes"] == [11, 22]
    assert admission_event["raw_hit_blocks"] == [
        {
            "query_index": 0,
            "block_hash": 11,
            "block_id": 0,
            "creator_request_id": str(creator.id),
            "binding_epoch": 1,
        },
        {
            "query_index": 1,
            "block_hash": 22,
            "block_id": 1,
            "creator_request_id": str(creator.id),
            "binding_epoch": 1,
        },
    ]
    assert admission_event["admitted_hit_blocks"] == [
        admission_event["raw_hit_blocks"][0]
    ]
    assert admission_event["full_hit_backoff_applied"] is True
    assert allocation_event["reused_blocks"] == [
        {
            "block_hash": 11,
            "block_id": 0,
            "creator_request_id": str(creator.id),
            "binding_epoch": 1,
        }
    ]
    assert allocation_event["new_block_ids"] == [1]
    assert allocation_event["evicted_bindings"] == [
        {
            "block_hash": 22,
            "block_id": 1,
            "creator_request_id": str(creator.id),
            "binding_epoch": 1,
        }
    ]
    assert allocation_event["new_bindings"] == [
        {
            "block_hash": 22,
            "block_id": 1,
            "creator_request_id": str(consumer.id),
            "binding_epoch": 2,
        }
    ]


def _admission_scheduler(manager: KVCacheManager) -> VLLMv1EngineReplicaScheduler:
    scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    scheduler._kv_cache_manager = manager
    scheduler._config = SimpleNamespace(block_size=2, num_blocks=manager.num_gpu_blocks)
    return scheduler


# A trace row that trace replay shortened keeps the hashes of its longer
# original prompt. They name tokens the request does not have.
@pytest.mark.parametrize(
    "num_prefill_tokens, prompt_hashes",
    [(3, [11]), (5, [11, 22])],
    ids=["prompt_inside_the_second_block", "prompt_inside_the_third_block"],
)
def test_a_prompt_looks_up_only_the_hashes_of_its_full_blocks(
    num_prefill_tokens, prompt_hashes
) -> None:
    manager = _manager(num_gpu_blocks=4)
    creator = _request(hashes=[11, 22, 33], num_prefill_tokens=6)
    assert manager.allocate_slots(creator, 6) is not None
    manager.free(creator)
    consumer = _request(hashes=[11, 22, 33], num_prefill_tokens=num_prefill_tokens)

    admission = _admission_scheduler(manager)._prepare_prefix_cache_admission(consumer)

    assert [block.block_hash for block in admission.raw_hit_blocks] == prompt_hashes
    assert [block.block_hash for block in admission.effective_hit_blocks] == prompt_hashes
    assert admission.num_new_tokens == num_prefill_tokens - 2 * len(prompt_hashes)
    assert admission.full_hit_backoff_applied is False
    assert manager.prefix_cache_stats.queries == len(prompt_hashes)


def test_blocks_past_the_prompt_keep_no_trace_hash() -> None:
    # The creator's decode tokens fill the blocks its extra hashes name, so a
    # longer prompt with the same hashes hits only the creator's prompt block.
    manager = _manager(num_gpu_blocks=4)
    creator = _request(hashes=[11, 22, 33], num_prefill_tokens=3)

    result = manager.allocate_slots(creator, 6)

    assert result is not None
    assert [binding.block_hash for binding in result.new_bindings] == [11]
    manager.free(creator)
    consumer = _request(hashes=[11, 22, 33], num_prefill_tokens=6)

    admission = _admission_scheduler(manager)._prepare_prefix_cache_admission(consumer)

    assert [block.block_hash for block in admission.raw_hit_blocks] == [11]
    assert admission.effective_cached_tokens == 2
    assert admission.num_new_tokens == 4


def test_hit_binding_fast_check_agrees_with_the_checked_loop() -> None:
    from frontier.kv_cache.kv_cache_block import KVCacheBlock
    from frontier.scheduler.replica_scheduler.vllm_v1_prefix_cache import (
        _checked_hit_bindings,
        _consistent_hit_bindings,
    )

    def bound(block_id, block_hash):
        block = KVCacheBlock(block_id)
        block.bind(block_hash, creator_request_id=1)
        return block

    blocks = [bound(i, h) for i, h in enumerate([11, 22, 33])]
    for queries in ([11, 22, 33], [11, 22, 33, 44]):
        fast = _consistent_hit_bindings(blocks, queries)
        assert fast == _checked_hit_bindings(blocks, queries)
        assert [b.block_hash for b in fast] == [11, 22, 33]
    assert _consistent_hit_bindings([], []) == []

    cases = [
        (blocks, [11, 99, 33], "disagrees"),  # wrong hash
        (blocks, [11, 22], None),  # more hits than queries: IndexError
        (blocks[:1] + [KVCacheBlock(7)] + blocks[1:], [11, 22, 33, 44], "no binding"),
    ]
    for hit, queries, message in cases:
        assert _consistent_hit_bindings(hit, queries) is None
        with pytest.raises(IndexError if message is None else ValueError, match=message):
            _checked_hit_bindings(hit, queries)


def _ledger_scheduler(manager):
    scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    scheduler._kv_cache_manager = manager
    scheduler._config = SimpleNamespace(block_size=2, num_blocks=2)
    scheduler._cluster_type = ClusterType.MONOLITHIC
    scheduler._replica_id = 3
    scheduler._replica_local_id = None
    scheduler._active_schedule_iteration_id = 7
    scheduler._current_schedule_time = 1.25
    scheduler._prefix_cache_identity_event_seq = 0
    scheduler._allocation_map = {}
    scheduler._num_allocated_blocks = 0
    return scheduler


@pytest.mark.parametrize("logged", [False, True])
def test_unlogged_identity_events_keep_sequence_and_checks(monkeypatch, logged) -> None:
    import dataclasses

    assert prefix_cache_module._decision_log_discards_events()
    if logged:
        monkeypatch.setattr(
            prefix_cache_module, "_log_frontier_vllm_v1_schedule_decision", lambda event: None
        )
        assert not prefix_cache_module._decision_log_discards_events()
    manager = _manager()
    creator = _request(hashes=[11, 22])
    manager.allocate_slots(creator, 4)
    manager.free(creator)
    consumer = _request(hashes=[11, 22])
    scheduler = _ledger_scheduler(manager)
    admission = scheduler._prepare_prefix_cache_admission(consumer)
    allocation = scheduler._allocate_request(
        consumer,
        admission.num_new_tokens,
        new_computed_blocks=list(admission.effective_hit_blocks),
        prefix_cache_admission=admission,
    )
    assert allocation is not None
    assert scheduler._prefix_cache_identity_event_seq == 2

    # a hit binding that disagrees with its query raises the same error either way
    bad = dataclasses.replace(
        admission,
        raw_hit_bindings=(dataclasses.replace(admission.raw_hit_bindings[0], block_hash=99),),
    )
    with pytest.raises(ValueError, match="query_index=0, query_hash=11, binding_hash=99"):
        scheduler._emit_prefix_cache_identity_events(
            request=consumer, num_new_tokens=2, allocation=allocation, admission=bad
        )
    assert scheduler._prefix_cache_identity_event_seq == 3
    too_many = dataclasses.replace(
        admission, effective_hit_bindings=admission.raw_hit_bindings * 2
    )
    with pytest.raises(ValueError, match="exceeds the ordered query hash count"):
        scheduler._emit_prefix_cache_identity_events(
            request=consumer, num_new_tokens=2, allocation=allocation, admission=too_many
        )
