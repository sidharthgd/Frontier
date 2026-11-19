"""
Baseline agents, by name (as in frontier/rl/sched/agents.py):

    fcfs                        every request priority 0: (arrival, id) order
    random                      uniform priorities
    sjf_prefill[_age<r>]        least prefill work first; aging subtracts r tokens
                                per second waited. Needs no oracle.
    sjf_total[_age<r>]          oracle: prefill work + output still to generate

Prefill work is observation.prefill_work_tokens, the size the training
stack's SJF baselines (frontier/rl/sched: sjf_prefill_tokens,
sjf_total_tokens) use too: the prompt (for a victim preempted after prefill,
the prompt plus the output it kept, which vLLM recomputes) less what the
prefix cache holds at ranking time. A 120k call whose first 119k tokens are
cached is a 1k-token job.
"""

import re
from typing import Optional

import numpy as np

from frontier.rl.env import BaseAgent, OracleAgent
from frontier.rl.observation import Observation, cached_prefix_tokens, job_size_tokens


class FCFSAgent(BaseAgent):
    def act(self, obs: Observation):
        return np.zeros(obs.num_requests, dtype=np.int64), {}


class RandomAgent(BaseAgent):
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: Observation):
        return self._rng.integers(0, obs.num_priority_levels, obs.num_requests), {}


def _ranks(sizes: np.ndarray, obs: Observation) -> np.ndarray:
    # same tie-break as the scheduler: (size, arrival, id)
    order = np.lexsort((obs.request_ids, obs.arrival_times, sizes))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order))
    return ranks


class _SJFAgent(OracleAgent):
    """SJF by job_size_tokens with aging. It reads the waiting requests (a
    victim's recompute length, the cache lookup), hence an OracleAgent, but
    only with_output reads the true output length."""

    unbounded_priorities = True
    with_output = False

    def __init__(self, aging: float = 0.0) -> None:
        self._aging = aging

    def act_oracle(self, obs, waiting, scheduler):
        sizes = np.array(
            [
                job_size_tokens(r, cached_prefix_tokens(scheduler, r), self.with_output)
                for r in waiting
            ],
            dtype=float,
        )
        sizes -= self._aging * (obs.time - obs.arrival_times)
        return _ranks(sizes, obs), {}


class PromptSJFAgent(_SJFAgent):
    """Least prefill work first (no oracle)."""


class OracleSJFAgent(_SJFAgent):
    """Least prefill work plus remaining output first (oracle)."""

    with_output = True


_SJF = re.compile(r"^sjf_(?P<key>prefill|total)(?:_age(?P<age>[0-9.]+))?$")


def baseline_agent(name: str, seed: Optional[int] = None) -> BaseAgent:
    if name == "fcfs":
        return FCFSAgent()
    if name == "random":
        return RandomAgent(seed)
    match = _SJF.match(name)
    if match:
        aging = float(match["age"] or 0.0)
        return PromptSJFAgent(aging) if match["key"] == "prefill" else OracleSJFAgent(aging)
    raise ValueError(f"unknown baseline {name}. {__doc__}")
