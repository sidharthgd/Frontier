"""
What the learned policy reads: an Observation's raw features (observation.py)
normalized for a network. Token counts are in thousands (KV blocks in free-
block units) and go through log1p, as do times in seconds and counts; ranks
are percentiles in [0, 1]. Nothing reads a request's output length.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from frontier.rl.observation import Observation

WAITING_INPUTS: Tuple[str, ...] = (
    "log_prompt_k",
    "log_uncached_k",        # prompt tokens not in the prefix cache now
    "cached_fraction",
    "log_processed_k",
    "log_wait_s",
    "log_preemptions",
    "log_uncached_blocks_over_free",
    "log_call_index",        # the call's index in its agent session
    "current_priority",      # the previous decision's level, scaled to [0, 1]
    "arrival_percentile",
    "uncached_percentile",   # sjf_uncached's order, as a feature
    "prompt_percentile",
    "has_evict_distance",    # its cached prefix has blocks no running request holds
    "log_evict_distance_k",  # blocks allocated before it loses one (thousands)
    "evict_over_queued",     # that distance over the blocks the queue needs, capped at 4, scaled to [0, 1]
    "time_to_deadline_frac", # (deadline - wait) / deadline, clipped to [-2, 1]; 1 with no deadline
    "overdue",               # 1 once the wait has reached the deadline (the scheduler then puts it first)
    "log_uncached_steps",    # uncached tokens in units of one step's token budget
)
RUNNING_INPUTS: Tuple[str, ...] = (
    "log_prompt_k",
    "log_processed_k",
    "log_emitted",
    "log_age_s",
    "is_prefill_complete",
    "current_priority",
)
GLOBAL_INPUTS: Tuple[str, ...] = (
    "free_block_fraction",
    "log_num_waiting",
    "log_num_running",
    "log_waiting_uncached_k",  # prefill work queued
    "log_running_context_k",   # KV held by running requests
    "log_deadline_s",          # 0 with no deadline
    "oldest_wait_over_deadline",  # longest wait / deadline, capped at 3; 0 with no deadline
    "overdue_fraction",        # share of waiting calls already past the deadline
)


@dataclass
class PolicyInput:
    num_priority_levels: int
    request_ids: np.ndarray
    arrival_times: np.ndarray
    waiting_features: np.ndarray  # [num_waiting, len(WAITING_INPUTS)]
    running_features: np.ndarray  # [num_running, len(RUNNING_INPUTS)]
    global_features: np.ndarray  # [len(GLOBAL_INPUTS)]

    @property
    def num_requests(self) -> int:
        return len(self.request_ids)

    @property
    def num_running(self) -> int:
        return len(self.running_features)


def _percentile(values: np.ndarray, obs: Observation) -> np.ndarray:
    n = len(values)
    order = np.lexsort((obs.request_ids, obs.arrival_times, values))
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return ranks / max(1, n - 1)


def _priority(values: np.ndarray, num_levels: int) -> np.ndarray:
    values = np.nan_to_num(values.astype(float), nan=0.0)
    return np.clip(values, 0, num_levels - 1) / (num_levels - 1)


def policy_input(obs: Observation) -> PolicyInput:
    w = obs.waiting_feature
    L = obs.num_priority_levels
    prompt = w("prompt_tokens")
    cached = np.minimum(w("cached_prompt_tokens"), prompt)
    uncached = prompt - cached
    free_blocks = max(1.0, obs.global_feature("num_free_blocks"))
    block_size = obs.global_feature("block_size")
    evict = w("evict_distance_blocks")
    has_evict = evict >= 0
    queued_blocks = max(1.0, float(uncached.sum()) / block_size)
    deadline = obs.global_feature("deadline_s")
    has_deadline = deadline > 0
    wait = w("wait_time")
    if has_deadline:
        to_deadline = np.clip((deadline - wait) / deadline, -2.0, 1.0)
        overdue = (wait >= deadline).astype(float)
    else:
        to_deadline = np.ones(obs.num_requests)
        overdue = np.zeros(obs.num_requests)
    step_tokens = max(1.0, obs.global_feature("max_tokens_in_batch"))
    waiting = np.stack(
        [
            np.log1p(prompt / 1000),
            np.log1p(uncached / 1000),
            cached / np.maximum(prompt, 1),
            np.log1p(w("processed_tokens") / 1000),
            np.log1p(w("wait_time")),
            np.log1p(w("num_preemptions")),
            np.log1p(uncached / block_size / free_blocks),
            np.log1p(w("round_index")),
            _priority(w("current_priority"), L),
            _percentile(obs.arrival_times, obs),
            _percentile(uncached, obs),
            _percentile(prompt, obs),
            has_evict.astype(float),
            np.log1p(np.where(has_evict, evict, 0.0) / 1000),
            np.where(has_evict, np.minimum(evict / queued_blocks, 4.0) / 4.0, 0.0),
            to_deadline,
            overdue,
            np.log1p(uncached / step_tokens),
        ],
        axis=1,
    ) if obs.num_requests else np.zeros((0, len(WAITING_INPUTS)))

    r = obs.running
    running = np.stack(
        [
            np.log1p(r[:, 0] / 1000),
            np.log1p(r[:, 1] / 1000),
            np.log1p(r[:, 2]),
            np.log1p(r[:, 3]),
            r[:, 4],
            _priority(r[:, 5], L),
        ],
        axis=1,
    ) if len(r) else np.zeros((0, len(RUNNING_INPUTS)))

    global_features = np.array(
        [
            obs.global_feature("free_block_fraction"),
            np.log1p(obs.global_feature("num_waiting")),
            np.log1p(obs.global_feature("num_running")),
            np.log1p(uncached.sum() / 1000),
            np.log1p((r[:, 0].sum() if len(r) else 0.0) / 1000),
            np.log1p(deadline) if has_deadline else 0.0,
            min(float(wait.max()) / deadline, 3.0) if has_deadline and len(wait) else 0.0,
            float(overdue.mean()) if has_deadline and len(wait) else 0.0,
        ]
    )
    return PolicyInput(
        num_priority_levels=L,
        request_ids=obs.request_ids,
        arrival_times=obs.arrival_times,
        waiting_features=waiting.astype(np.float32),
        running_features=running.astype(np.float32),
        global_features=global_features.astype(np.float32),
    )

