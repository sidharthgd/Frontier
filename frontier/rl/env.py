"""
What the RL stack's environments share: the Transition and Episode records,
episode summaries, and SchedulingEnv's reference runs (baselines on the same
workload), reward thresholds and normalization. FrontierEnv
(frontier_env.py) is the environment; it simulates episodes in Frontier.

Ported from the Vidur repo (branch frontier-backend, vidur/rl/env.py), whose
own simulator backend stayed behind.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from frontier.entities import Request
from frontier.rl.agents import BaseAgent, FCFSAgent, OracleAgent, baseline_agent
from frontier.rl.observation import Observation, ObservationConfig
from frontier.rl.reward import (
    MEAN_COMPONENT_REFERENCES,
    THRESHOLD_METRICS,
    RewardConfig,
    RewardContext,
    episode_cost,
    episode_cost_components,
    episode_metric_samples,
    window_thresholds,
    TTFT_REF_THRESHOLD,
    reference_ttft_thresholds,
    zero_components,
)
from frontier.rl.workload import (  # noqa: F401 (re-exported)
    SESSION_TRACE_FILES,
    TRACE_FILES,
    SimulationConfig,
    make_request_generator_config,
    make_simulation_config,
)
from frontier.rl.workload import set_workload_seed as _set_workload_seed


@dataclass
class Transition:
    obs: Observation
    action: np.ndarray
    # whatever the agent returned alongside the action (e.g. log_prob, value)
    info: Dict[str, Any]
    time: float
    num_admitted: int
    # index of the simulator event during which the decision was made (to replay
    # or fork the simulation at this decision)
    event_index: int = -1
    # [num_requests] whether each waiting request was admitted into the batch
    admitted: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # index (in obs order) of the first request in ranked order that was not
    # admitted; None if all were
    boundary_index: Optional[int] = None
    # inclusive per-request bounds on the priority that would have produced the
    # same admission (see action.py); the training log-prob is
    # Σ_i log P(action_low_i <= a_i <= action_high_i) over obs.rank_mask
    action_low: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    action_high: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    # filled in at the next decision (or episode end): what accumulated between
    # this decision and the next one
    reward: float = 0.0
    reward_components: Dict[str, float] = field(default_factory=zero_components)
    duration: float = 0.0


@dataclass
class Episode:
    transitions: List[Transition]
    # accumulated before the first decision; no action can be credited with it
    initial_reward_components: Dict[str, float]
    requests: List[Request]
    # original output length per request id (Request.restart() rewrites
    # num_decode_tokens to the number of tokens left)
    num_decode_tokens: Dict[int, int]
    # every generated token: time, request id, gap since the request's previous
    # token (nan for its first token)
    token_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    token_request_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    token_gaps: np.ndarray = field(default_factory=lambda: np.zeros(0))
    summary: Dict[str, float] = field(default_factory=dict)
    # per-request thresholds the rewards used (reference_window mode)
    request_thresholds: Dict[str, Dict[int, float]] = field(default_factory=dict)
    # every reward was multiplied by this (1 / baseline cost or gap when normalizing)
    reward_scale: float = 1.0
    # unscaled episode cost (Σ weight * component, including what accrued before
    # the first decision) and the same for the normalization baselines
    cost: float = float("nan")
    reference_costs: Dict[str, float] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        return sum(transition.reward for transition in self.transitions)

    @property
    def normalized_score(self) -> Optional[float]:
        """
        (cost_a - cost) / (cost_a - cost_b) for normalize_by_baselines=(a, b),
        with cost_b = 0 for (a,): 0 = as good as a, 1 = as good as b (or zero
        cost). None when not normalizing.
        """
        if not self.reference_costs:
            return None
        cost_a, cost_b = (list(self.reference_costs.values()) + [0.0])[:2]
        if cost_a == cost_b:
            return float("nan")
        return (cost_a - self.cost) / (cost_a - cost_b)


def _act(
    agent: BaseAgent,
    obs: Observation,
    waiting_requests: List[Request],
    scheduler,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if isinstance(agent, OracleAgent):
        priorities, info = agent.act_oracle(obs, waiting_requests, scheduler)
    else:
        priorities, info = agent.act(obs)
    return np.asarray(priorities, dtype=np.int64), info



def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else float("nan")


def _summarize(episode: Episode, end_time: float) -> Dict[str, float]:
    requests = episode.requests
    num_decode_tokens = episode.num_decode_tokens
    completed = [request for request in requests if request.completed]
    samples = episode_metric_samples(episode)
    e2e, ttft, tbt = samples["e2e"], samples["ttft"], samples["tbt"]
    e2e_normalized = np.array(
        [(r.completed_at - r.arrived_at) / num_decode_tokens[r.id] for r in completed]
    )

    def mean(values: np.ndarray) -> float:
        return float(np.mean(values)) if len(values) else float("nan")

    return {
        "num_requests": len(requests),
        "num_completed": len(completed),
        "num_decisions": len(episode.transitions),
        "total_reward": float(episode.total_reward),
        "end_time": float(end_time),
        "mean_e2e": mean(e2e),
        "p50_e2e": _quantile(e2e, 0.5),
        "p95_e2e": _quantile(e2e, 0.95),
        "p99_e2e": _quantile(e2e, 0.99),
        "max_e2e": float(np.max(e2e)) if len(e2e) else float("nan"),
        "mean_e2e_normalized": mean(e2e_normalized),
        "p95_e2e_normalized": _quantile(e2e_normalized, 0.95),
        "p99_e2e_normalized": _quantile(e2e_normalized, 0.99),
        "mean_ttft": mean(ttft),
        "p95_ttft": _quantile(ttft, 0.95),
        "p99_ttft": _quantile(ttft, 0.99),
        "mean_tbt": mean(tbt),
        "p95_tbt": _quantile(tbt, 0.95),
        "p99_tbt": _quantile(tbt, 0.99),
        "output_tokens_per_s": (
            sum(num_decode_tokens[r.id] for r in completed) / end_time
            if end_time > 0
            else float("nan")
        ),
        "num_restarts": int(sum(r.num_restarts for r in requests)),
    }



class SchedulingEnv:
    """
    Reference runs, reward context and threshold calibration for an
    environment whose run_episode(agent, workload_seed, request_generator_config,
    reward_config) simulates one episode (FrontierEnv). Subclasses set
    _config, _reward_config, _observation_config and _reference_episodes.

    The workload (arrival times, request lengths) is fully determined by
    workload_seed, so several rollouts on the same seed differ only through the
    agent's own sampling.
    """

    @property
    def config(self) -> SimulationConfig:
        return self._config

    @property
    def reward_config(self) -> RewardConfig:
        return self._reward_config

    @property
    def observation_config(self) -> ObservationConfig:
        return self._observation_config

    def reference_episode(
        self,
        baseline: str,
        workload_seed: Optional[int] = None,
        request_generator_config: Optional[Any] = None,
    ) -> Episode:
        """
        A baseline's episode on this workload (see agents.baseline_agent),
        cached. Baselines do not depend on the reward, so the cached run can be
        rescored under any reward config with episode_cost().
        """
        key = (baseline, workload_seed, repr(request_generator_config))
        if key not in self._reference_episodes:
            # a plain reward config, so building it needs no reference runs
            self._reference_episodes[key] = self.run_episode(
                baseline_agent(baseline, seed=workload_seed),
                workload_seed,
                request_generator_config,
                reward_config=RewardConfig(),
            )
        return self._reference_episodes[key]

    def reward_context(
        self,
        reward_config: RewardConfig,
        workload_seed: Optional[int] = None,
        request_generator_config: Optional[Any] = None,
    ) -> RewardContext:
        """Per-request thresholds and reward scale for this workload."""
        context = RewardContext()
        if reward_config.window_thresholds:
            reference = self.reference_episode(
                reward_config.threshold_reference,
                workload_seed,
                request_generator_config,
            )
            context.request_thresholds = window_thresholds(reference, reward_config)
        if reward_config.weights.get("ttft_ref_sq"):
            reference = self.reference_episode(
                reward_config.threshold_reference,
                workload_seed,
                request_generator_config,
            )
            context.request_thresholds[TTFT_REF_THRESHOLD] = reference_ttft_thresholds(
                reference, reward_config
            )

        if reward_config.normalize_by_baselines is not None:
            for baseline in reward_config.normalize_by_baselines:
                reference = self.reference_episode(
                    baseline, workload_seed, request_generator_config
                )
                context.reference_costs[baseline] = episode_cost(
                    reference, reward_config, context.request_thresholds
                )
            cost_a, cost_b = (list(context.reference_costs.values()) + [0.0])[:2]
            gap = max(
                abs(cost_a - cost_b),
                reward_config.normalization_min_fraction * abs(cost_a),
                1e-9,
            )
            context.scale = 1.0 / gap
        return context

    def mean_gap_closed(
        self,
        episode: Episode,
        workload_seed: Optional[int] = None,
        request_generator_config: Optional[Any] = None,
        mean_term: str = "e2e_normalized",
        reference: Optional[str] = None,
    ) -> float:
        """
        Evaluation metric, not a reward: the fraction of the FCFS-to-SJF gain on
        one mean-latency component that the episode achieved,
            (M_fcfs - M) / (M_fcfs - M_sjf)
        0 = FCFS, 1 = oracle SJF, > 1 = better than SJF (possible: SJF is not
        optimal under continuous batching). The episode must be from the given
        workload. reference defaults to MEAN_COMPONENT_REFERENCES[mean_term].
        Tails are reported separately, against FCFS.
        """
        reference = reference or MEAN_COMPONENT_REFERENCES[mean_term]
        config = RewardConfig(weights={mean_term: 1.0})

        def mean_cost(e: Episode) -> float:
            return episode_cost_components(e, config, {})[mean_term]

        cost_fcfs, cost_sjf = (
            mean_cost(
                self.reference_episode(name, workload_seed, request_generator_config)
            )
            for name in ("fcfs", reference)
        )
        if cost_fcfs == cost_sjf:
            return float("nan")
        return (cost_fcfs - mean_cost(episode)) / (cost_fcfs - cost_sjf)

    def calibrate_thresholds(self, reward_config: RewardConfig) -> RewardConfig:
        """
        Fill the thresholds left as None with calibration_quantile of their
        metric over FCFS episodes on calibration_seeds. Fixed thresholds keep the
        reward stationary while the policy learns. In reference_window mode only
        tbt_slo is calibrated here; the rest are set per workload.
        """
        samples: Dict[str, List[np.ndarray]] = {"ttft": [], "e2e": [], "tbt": []}
        for seed in reward_config.calibration_seeds:
            episode = self.run_episode(
                FCFSAgent(), workload_seed=seed, reward_config=RewardConfig()
            )
            for metric, values in episode_metric_samples(episode).items():
                samples[metric].append(values)

        q = reward_config.calibration_quantile
        updates = {
            name: _quantile(np.concatenate(samples[metric]), q)
            for name, metric in THRESHOLD_METRICS.items()
            if getattr(reward_config, name) is None
            and name not in reward_config.window_thresholds
        }
        return dataclasses.replace(reward_config, **updates)
