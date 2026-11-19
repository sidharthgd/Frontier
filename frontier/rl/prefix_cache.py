"""
A waiting request's prefix-cache hit and the prefill it still needs: what the
features (features.py), the SJF baselines (history.job_size via
frontier_env._OracleRequest) and the reward context read from Frontier.
"""

from typing import Tuple

from frontier.entities import Request


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


def prefill_target_tokens(request: Request) -> int:
    """Tokens a waiting request must have computed before it decodes again:
    its prompt, or for a victim preempted after prefill its prompt plus the
    output it kept (vLLM v1 recomputes both; prefix_cache_view looks up the
    same span)."""
    if request.is_recomputing:
        return int(request.num_processed_tokens)
    return int(request.num_prefill_tokens)
