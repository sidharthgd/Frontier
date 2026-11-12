"""
Baseline agents, by name (as in vidur/rl/agents.py):

    fcfs                        every request priority 0: (arrival, id) order
    random                      uniform priorities
    sjf_prefill[_age<r>]        shortest prompt first; aging subtracts r tokens per
                                second waited. Needs no oracle.
    sjf_total[_age<r>]          oracle: prompt + true output length
    sjf_uncached[_age<r>]       shortest uncached prompt first: the prompt tokens
                                not in the prefix cache now, i.e. the prefill a
                                call actually costs. Needs no oracle.
    session_fewest[_age<r>]     fewest earlier calls in its session first, an
                                attained-service rule like Autellix's
    oracle_cost[_age<r>]        oracle: uncached prompt + true output length
    sjf_uncached_evict<g>_<k>   sjf_uncached, but a call whose cached prefix will lose
                                a block within k allocations (eviction distance < k
                                blocks) is sized g * its cached tokens smaller
    sjf_uncached_deadline<d>    sjf_uncached, but calls waiting longer than d seconds
                                go first (oldest first)
    sjf_uncached_cold<q>_ramp<t>  sjf_uncached, but a call in the top q percent of recent calls by
                                uncached tokens ("cold") has its size shrink linearly with its
                                wait, reaching zero after t seconds. Others are unchanged.
    sjf_uncached_qcold<q>_ramp<t>  the same, but "cold" means the top q percent of the calls
                                waiting right now (no history window)
    sjf_uncached_hrrn<t>        uncached tokens / (1 + wait / t): a call's size shrinks in
                                proportion to how long it has waited (highest response
                                ratio next, as in HRRN scheduling)
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


class SizeAgent(BaseAgent):
    """Ranks by a size computed from the observation, minus aging * wait."""

    unbounded_priorities = True

    def __init__(self, size_fn, aging: float = 0.0) -> None:
        self._size_fn = size_fn
        self._aging = aging

    def act(self, obs: Observation):
        sizes = self._size_fn(obs) - self._aging * obs.waiting_feature("wait_time")
        return _ranks(sizes, obs), {}


def _uncached(obs: Observation) -> np.ndarray:
    return np.maximum(
        obs.waiting_feature("prompt_tokens") - obs.waiting_feature("cached_prompt_tokens"), 0
    )


class OracleCostAgent(OracleAgent):
    unbounded_priorities = True

    def __init__(self, aging: float = 0.0) -> None:
        self._aging = aging

    def act_oracle(self, obs, waiting, scheduler):
        sizes = _uncached(obs) + np.array([r.num_decode_tokens for r in waiting], dtype=float)
        sizes -= self._aging * obs.waiting_feature("wait_time")
        return _ranks(sizes, obs), {}


def _evict_size(gamma: float, within_blocks: int):
    def size(obs: Observation) -> np.ndarray:
        distance = obs.waiting_feature("evict_distance_blocks")
        at_risk = (distance >= 0) & (distance < within_blocks)
        return _uncached(obs) - gamma * obs.waiting_feature("cached_prompt_tokens") * at_risk

    return size


def _deadline_size(seconds: float):
    def size(obs: Observation) -> np.ndarray:
        overdue = obs.waiting_feature("wait_time") > seconds
        return np.where(overdue, -1e12, _uncached(obs))

    return size


def _hrrn_size(scale_s: float):
    def size(obs: Observation) -> np.ndarray:
        return _uncached(obs) / (1.0 + obs.waiting_feature("wait_time") / scale_s)

    return size


class ColdRampAgent(BaseAgent):
    """
    sjf_uncached, with targeted aging for the most expensive calls. A call is
    cold if its uncached tokens are in the top `top_percent` of the last
    `window` calls seen (a relative threshold, so no token counts to tune); a
    cold call's size is scaled by max(0, 1 - wait / ramp_s).
    """

    unbounded_priorities = True

    def __init__(self, top_percent: float, ramp_s: float, window: int = 200, in_queue: bool = False) -> None:
        from collections import deque

        self._in_queue = in_queue
        self._quantile = 1.0 - top_percent / 100.0
        self._ramp_s = ramp_s
        self._recent = deque(maxlen=window)
        self._seen = set()

    def act(self, obs: Observation):
        uncached = _uncached(obs)
        if self._in_queue:
            threshold = float(np.quantile(uncached, self._quantile))
        else:
            for request_id, value in zip(obs.request_ids, uncached):
                if request_id not in self._seen:
                    self._seen.add(request_id)
                    self._recent.append(float(value))
            threshold = float(np.quantile(self._recent, self._quantile))
        cold = (uncached >= threshold) & (uncached > 0)
        ramp = np.clip(1.0 - obs.waiting_feature("wait_time") / self._ramp_s, 0.0, 1.0)
        return _ranks(np.where(cold, uncached * ramp, uncached), obs), {}


_SJF = re.compile(r"^sjf_(?P<key>prefill|total)(?:_age(?P<age>[0-9.]+))?$")
_RULE = re.compile(r"^(?P<rule>sjf_uncached|session_fewest|oracle_cost)(?:_age(?P<age>[0-9.]+))?$")
_EVICT = re.compile(r"^sjf_uncached_evict(?P<g>[0-9.]+)_(?P<k>[0-9]+)$")
_COLD = re.compile(r"^sjf_uncached_(?P<kind>q?cold)(?P<q>[0-9.]+)_ramp(?P<t>[0-9.]+)$")
_HRRN = re.compile(r"^sjf_uncached_hrrn(?P<t>[0-9.]+)$")
_DEADLINE = re.compile(r"^sjf_uncached_deadline(?P<d>[0-9.]+)$")


def baseline_agent(name: str, seed: Optional[int] = None) -> BaseAgent:
    if name == "fcfs":
        return FCFSAgent()
    if name == "random":
        return RandomAgent(seed)
    match = _SJF.match(name)
    if match:
        aging = float(match["age"] or 0.0)
        return PromptSJFAgent(aging) if match["key"] == "prefill" else OracleSJFAgent(aging)
    match = _RULE.match(name)
    if match:
        aging = float(match["age"] or 0.0)
        if match["rule"] == "sjf_uncached":
            return SizeAgent(_uncached, aging)
        if match["rule"] == "session_fewest":
            return SizeAgent(lambda obs: obs.waiting_feature("round_index"), aging)
        return OracleCostAgent(aging)
    match = _EVICT.match(name)
    if match:
        return SizeAgent(_evict_size(float(match["g"]), int(match["k"])))
    match = _DEADLINE.match(name)
    if match:
        return SizeAgent(_deadline_size(float(match["d"])))
    match = _COLD.match(name)
    if match:
        return ColdRampAgent(float(match["q"]), float(match["t"]), in_queue=match["kind"] == "qcold")
    match = _HRRN.match(name)
    if match:
        return SizeAgent(_hrrn_size(float(match["t"])))
    raise ValueError(f"unknown baseline {name}. {__doc__}")
