"""
RL environment on top of Frontier (ported from our Vidur fork, vidur/rl/env.py).

    env = FrontierSchedulingEnv(WorkloadSpec(name="toy", qps=10, num_requests=200))
    episode = env.run_episode(agent, seed=0)

run_episode runs one simulation to the end. Whenever the vLLM v1 scheduler is
about to admit waiting requests, it builds an Observation, asks the agent for
one priority per waiting request (agent.act(obs) -> (priorities, info)), and
lets vLLM admit from the front of the (priority, arrival, id) order.

A decision is kept as a Transition only if it admitted something. Otherwise
the ranking had no effect: the previous priorities are restored (priorities
also drive preemption, so leaving them would be an effect) and the decision
is dropped. Each kept transition gets the reward components that accrue until
the next kept decision (vidur/rl/reward.py semantics), computed from request
timings after the episode; weights are a RewardConfig, chosen by the trainer.

Nothing here decides features transforms, reward weights or the model. Feature
values are raw; normalization is the model's job.
"""

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from frontier.entities import Batch, Request
from frontier.rl.observation import Observation, build_observation
from frontier.rl.reward import RewardConfig, transition_components
from frontier.simulator import Simulator
from frontier.rl.scheduler import (
    PriorityPolicy,
    RLVLLMv1EngineReplicaScheduler,
    replica_schedulers,
    rl_scheduler_installed,
)


@dataclass
class WorkloadSpec:
    """
    One workload. Lengths come from `length_trace` (a CSV with
    num_prefill_tokens, num_decode_tokens) or are fixed; arrivals are Poisson,
    Gamma (cv > 1 is bursty) or, with `replay_trace`, replayed from a CSV
    with arrived_at (and optionally session_id, block_hash_ids,
    thinking_round_plans_json, tool_call_latency) that also gives lengths.
    """

    name: str
    num_requests: int = 300
    qps: float = 10.0
    arrivals: str = "poisson"  # poisson | gamma | replay
    cv: float = 1.0
    length_trace: Optional[str] = None
    prefill_tokens: int = 1000
    decode_tokens: int = 200
    replay_trace: Optional[str] = None
    replay_time_scale: float = 1.0
    max_tokens: int = 4096
    # hardware and engine
    model: str = "llama2_7b_dense_example"
    device: str = "h200"
    network_device: str = "h200_dgx"
    tensor_parallel: int = 1
    max_tokens_in_batch: int = 2048
    batch_size_cap: int = 256
    num_blocks: Optional[int] = None
    enable_prefix_caching: bool = False
    # fixed time per batch instead of the profiled predictor (plumbing tests)
    dummy_exec_ms: Optional[float] = None
    # any other frontier.main flags, e.g. ["--enable_thinking_mode", ...]
    extra_args: List[str] = field(default_factory=list)


def simulation_args(
    spec: WorkloadSpec, seed: int, output_dir: str, log_level: str = "error"
) -> List[str]:
    args = [
        "--log_level", log_level,
        "--seed", str(seed),
        "--sys_arch", "co-location",
        "--simulation_mode", "online",
        "--cc_backend_config_type", "analytical",
        "--replica_config_model_name", spec.model,
        "--replica_config_device", spec.device,
        "--replica_config_network_device", spec.network_device,
        "--replica_config_attn_tensor_parallel_size", str(spec.tensor_parallel),
        "--replica_scheduler_config_type", "vllm_v1",
        "--vllm_v1_scheduler_config_scheduling_policy", "priority",
        "--vllm_v1_scheduler_config_enable_chunked_prefill",
        "--vllm_v1_scheduler_config_max_tokens_in_batch", str(spec.max_tokens_in_batch),
        "--vllm_v1_scheduler_config_batch_size_cap", str(spec.batch_size_cap),
        "--decode_cuda_graph_mode", "none",
        "--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size",
        str(max(spec.max_tokens_in_batch, 1024)),
        "--no-random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling",
        "--metrics_config_output_dir", output_dir,
        "--no-metrics_config_write_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
    ]
    if spec.dummy_exec_ms is not None:
        args += [
            "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
            "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms",
            str(spec.dummy_exec_ms),
        ]
    else:
        args += ["--no-random_forrest_execution_time_predictor_config_enable_dummy_mode"]
    if spec.num_blocks is not None:
        args += [
            "--vllm_v1_scheduler_config_num_blocks_mode", "explicit",
            "--vllm_v1_scheduler_config_num_blocks", str(spec.num_blocks),
        ]
    if spec.enable_prefix_caching:
        args += ["--vllm_v1_scheduler_config_enable_prefix_caching"]

    if spec.arrivals == "replay":
        assert spec.replay_trace, "arrivals=replay needs replay_trace"
        args += [
            "--request_generator_config_type", "trace_replay",
            "--trace_request_generator_config_trace_file", spec.replay_trace,
            "--trace_request_generator_config_time_scale_factor", str(spec.replay_time_scale),
            "--trace_request_generator_config_max_tokens", str(spec.max_tokens),
        ]
    else:
        args += [
            "--request_generator_config_type", "synthetic",
            "--synthetic_request_generator_config_num_requests", str(spec.num_requests),
        ]
        if spec.length_trace:
            args += [
                "--length_generator_config_type", "trace",
                "--trace_request_length_generator_config_trace_file", spec.length_trace,
                "--trace_request_length_generator_config_max_tokens", str(spec.max_tokens),
            ]
        else:
            args += [
                "--length_generator_config_type", "fixed",
                "--fixed_request_length_generator_config_prefill_tokens", str(spec.prefill_tokens),
                "--fixed_request_length_generator_config_decode_tokens", str(spec.decode_tokens),
            ]
        if spec.arrivals == "poisson":
            args += [
                "--interval_generator_config_type", "poisson",
                "--poisson_request_interval_generator_config_qps", str(spec.qps),
            ]
        elif spec.arrivals == "gamma":
            args += [
                "--interval_generator_config_type", "gamma",
                "--gamma_request_interval_generator_config_qps", str(spec.qps),
                "--gamma_request_interval_generator_config_cv", str(spec.cv),
            ]
        else:
            raise ValueError(f"unknown arrivals {spec.arrivals}")
    return args + list(spec.extra_args)


def make_simulation_config(
    spec: WorkloadSpec, seed: int, output_dir: str, log_level: str = "error"
):
    """A SimulationConfig built through Frontier's own CLI parser."""
    from frontier.config import SimulationConfig

    argv = sys.argv
    sys.argv = ["frontier.main"] + simulation_args(spec, seed, output_dir, log_level)
    try:
        return SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = argv


@contextmanager
def _working_dir(path: str) -> Iterator[None]:
    # Frontier resolves data/ and cache/ relative to the working directory
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


FRONTIER_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# agents and episodes
# ---------------------------------------------------------------------------


class BaseAgent:
    """act(obs) -> (one priority per obs.request_ids entry, info dict)."""

    # oracle baselines may return any int (e.g. a job size); learned policies
    # are held to [0, num_priority_levels)
    unbounded_priorities: bool = False

    def act(self, obs: Observation) -> Tuple[Sequence[int], Dict[str, Any]]:
        raise NotImplementedError


class OracleAgent(BaseAgent):
    """Also sees the true Request objects (output lengths included)."""

    def act_oracle(
        self,
        obs: Observation,
        waiting: List[Request],
        scheduler: RLVLLMv1EngineReplicaScheduler,
    ) -> Tuple[Sequence[int], Dict[str, Any]]:
        raise NotImplementedError


@dataclass
class Transition:
    time: float
    obs: Observation
    priorities: np.ndarray
    info: Dict[str, Any]
    admitted_ids: List[int]
    boundary_id: Optional[int]
    reward_components: Dict[str, float] = field(default_factory=dict)
    reward: float = 0.0


@dataclass
class RequestRecord:
    id: int
    arrived_at: float
    num_prefill_tokens: int
    num_decode_tokens: int
    session_id: Optional[int]
    first_token_at: Optional[float] = None
    completed_at: Optional[float] = None
    token_times: List[float] = field(default_factory=list)
    num_preemptions: int = 0
    num_cached_prefill_tokens: int = 0


@dataclass
class Episode:
    workload: str
    seed: int
    transitions: List[Transition]
    requests: List[RequestRecord]
    end_time: float
    # decisions that admitted nothing and were dropped
    num_dropped_decisions: int
    summary: Dict[str, float] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        return float(sum(t.reward for t in self.transitions))


class AgentPriorityPolicy(PriorityPolicy):
    """Bridges the scheduler hook and an agent, recording transitions."""

    def __init__(self, agent: BaseAgent, num_priority_levels: int, clock) -> None:
        self._agent = agent
        self._num_levels = num_priority_levels
        self._clock = clock
        self._pending: Optional[Tuple[Transition, List[Request], List[int]]] = None
        self.transitions: List[Transition] = []
        self.num_dropped = 0
        self.records: Dict[int, RequestRecord] = {}

    def _record(self, request: Request) -> RequestRecord:
        record = self.records.get(request.id)
        if record is None:
            record = RequestRecord(
                id=request.id,
                arrived_at=float(request.arrived_at),
                num_prefill_tokens=int(request.num_prefill_tokens),
                num_decode_tokens=int(request.num_decode_tokens),
                session_id=request.session_id,
            )
            self.records[request.id] = record
        return record

    def get_priorities(self, waiting, scheduler):
        for request in waiting:
            self._record(request)
        if not waiting:
            return None
        time = float(self._clock())
        obs = build_observation(time, waiting, scheduler, self._num_levels)
        if isinstance(self._agent, OracleAgent):
            priorities, info = self._agent.act_oracle(obs, waiting, scheduler)
        else:
            priorities, info = self._agent.act(obs)
        priorities = np.asarray(priorities, dtype=np.int64)
        assert priorities.shape == (len(waiting),)
        if not self._agent.unbounded_priorities:
            assert priorities.min() >= 0 and priorities.max() < self._num_levels, (
                f"priorities must be in [0, {self._num_levels})"
            )
        previous = [r.priority for r in waiting]
        transition = Transition(time, obs, priorities, info, [], None)
        self._pending = (transition, list(waiting), previous)
        return priorities

    def on_admission(self, admitted, boundary):
        if self._pending is None:
            return
        transition, waiting, previous = self._pending
        self._pending = None
        if not admitted:
            # no effect on admission: undo, so it has no effect on preemption
            for request, priority in zip(waiting, previous):
                request.set_priority(priority)
            self.num_dropped += 1
            return
        transition.admitted_ids = [r.id for r in admitted]
        transition.boundary_id = boundary.id if boundary is not None else None
        self.transitions.append(transition)

    def on_batch_end(self, batch: Batch) -> None:
        time = float(batch.completed_at)
        for request in batch.requests:
            record = self._record(request)
            if request.is_prefill_complete:
                record.token_times.append(time)
                if record.first_token_at is None:
                    record.first_token_at = time
            if request.completed and record.completed_at is None:
                record.completed_at = float(request.completed_at)
                record.num_preemptions = int(request.get_total_preemption_count())
                record.num_cached_prefill_tokens = int(request.num_prefill_tokens_cached)


class FrontierSchedulingEnv:
    def __init__(
        self,
        spec: WorkloadSpec,
        reward_config: Optional[RewardConfig] = None,
        num_priority_levels: int = 16,
        output_dir: str = "/tmp/frontier_rl",
        log_level: str = "error",
    ) -> None:
        self.spec = spec
        self.reward_config = reward_config or RewardConfig()
        self.num_priority_levels = num_priority_levels
        self._output_dir = output_dir
        # Frontier logs every scheduling step at INFO, about a third of an
        # episode's time; the config applies --log_level when it is built
        self._log_level = log_level

    def run_stock(self, seed: int = 0):
        """Stock Frontier (no RL scheduler, no policy) on the same workload; returns the Simulator."""
        from frontier.utils.random import set_seeds

        with _working_dir(FRONTIER_ROOT):
            config = make_simulation_config(self.spec, seed, self._output_dir, self._log_level)
            set_seeds(seed)
            simulator = _RLSimulator(config)
            simulator.run()
        return simulator

    def make_simulator(self, seed: int):
        from frontier.utils.random import set_seeds

        config = make_simulation_config(self.spec, seed, self._output_dir, self._log_level)
        set_seeds(seed)
        with rl_scheduler_installed():
            simulator = _RLSimulator(config)
        return simulator

    def run_episode(self, agent: BaseAgent, seed: int = 0) -> Episode:
        with _working_dir(FRONTIER_ROOT):
            simulator = self.make_simulator(seed)
            policy = AgentPriorityPolicy(
                agent, self.num_priority_levels, clock=lambda: simulator._time
            )
            schedulers = replica_schedulers(simulator)
            assert len(schedulers) == 1, "the env drives a single replica"
            schedulers[0].set_priority_policy(policy)
            simulator.run()
        self.last_simulator = simulator

        requests = sorted(policy.records.values(), key=lambda r: r.id)
        episode = Episode(
            workload=self.spec.name,
            seed=seed,
            transitions=policy.transitions,
            requests=requests,
            end_time=float(simulator._time),
            num_dropped_decisions=policy.num_dropped,
        )
        components = transition_components(episode, self.reward_config)
        for transition, values in zip(episode.transitions, components):
            transition.reward_components = values
            transition.reward = self.reward_config.combine(values)
        episode.summary = summarize(episode)
        return episode


class _RLSimulator(Simulator):
    """Frontier's simulator without the end-of-run metric dump."""

    def _write_output(self) -> None:
        pass


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else float("nan")


def summarize(episode: Episode) -> Dict[str, float]:
    done = [r for r in episode.requests if r.completed_at is not None]
    e2e = np.array([r.completed_at - r.arrived_at for r in done])
    ttft = np.array(
        [r.first_token_at - r.arrived_at for r in done if r.first_token_at is not None]
    )
    gaps = np.concatenate(
        [np.diff(r.token_times) for r in done if len(r.token_times) > 1] or [np.array([])]
    )
    out = {
        "num_requests": len(episode.requests),
        "num_completed": len(done),
        "num_decisions": len(episode.transitions),
        "num_dropped_decisions": episode.num_dropped_decisions,
        "end_time": episode.end_time,
        "total_reward": episode.total_reward,
        "num_preemptions": int(sum(r.num_preemptions for r in done)),
        "cached_prefill_fraction": (
            sum(r.num_cached_prefill_tokens for r in done)
            / max(1, sum(r.num_prefill_tokens for r in done))
        ),
    }
    for name, values in (("e2e", e2e), ("ttft", ttft), ("tbt", gaps)):
        out[f"mean_{name}"] = float(values.mean()) if len(values) else float("nan")
        for q in (0.5, 0.95, 0.99):
            out[f"p{int(q * 100)}_{name}"] = _quantile(values, q)
        out[f"max_{name}"] = float(values.max()) if len(values) else float("nan")
    return out
