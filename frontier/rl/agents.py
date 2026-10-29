"""
Baseline agents, by name (as in vidur/rl/agents.py):

    fcfs                        every request priority 0: (arrival, id) order
    random                      uniform priorities
    sjf_prefill[_age<r>]        shortest prompt first; aging subtracts r tokens per
                                second waited. Needs no oracle.
    sjf_total[_age<r>]          oracle: prompt + true output length
"""

import re
from typing import Optional

import numpy as np

from frontier.rl.env import BaseAgent, OracleAgent
from frontier.rl.observation import Observation


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


class PromptSJFAgent(BaseAgent):
    unbounded_priorities = True

    def __init__(self, aging: float = 0.0) -> None:
        self._aging = aging

    def act(self, obs: Observation):
        sizes = obs.waiting_feature("prompt_tokens") - obs.waiting_feature(
            "processed_tokens"
        )
        sizes = sizes - self._aging * obs.waiting_feature("wait_time")
        return _ranks(sizes, obs), {}


class OracleSJFAgent(OracleAgent):
    unbounded_priorities = True

    def __init__(self, aging: float = 0.0) -> None:
        self._aging = aging

    def act_oracle(self, obs, waiting, scheduler):
        sizes = np.array(
            [r.num_prefill_tokens + r.num_decode_tokens for r in waiting], dtype=float
        )
        sizes -= self._aging * (obs.time - obs.arrival_times)
        return _ranks(sizes, obs), {}


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
