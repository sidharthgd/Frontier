"""
RL environment on Frontier (frontier/rl). Uses dummy execution times, so it
needs no profiling data or trained predictors.

    python -m pytest tests/rl
"""

import numpy as np
import pytest

from frontier.rl.agents import baseline_agent
from frontier.rl.env import BaseAgent, FrontierSchedulingEnv, OracleAgent, WorkloadSpec
from frontier.rl.observation import build_observation
from frontier.rl.reward import RewardConfig

# tight KV memory so admission blocks and running requests get preempted
SPEC = WorkloadSpec(
    name="tiny", num_requests=120, qps=30, prefill_tokens=600, decode_tokens=300,
    dummy_exec_ms=0.05, num_blocks=400,
)
SEEDS = (0, 1)


@pytest.fixture(scope="module")
def env():
    return FrontierSchedulingEnv(SPEC, RewardConfig(weights={"e2e": 1.0}))


def _timings(requests):
    # ids come from a global counter, so they differ between simulations
    return sorted(
        (r.arrived_at, r.prefill_completed_at, r.completed_at, r.get_total_preemption_count())
        for r in requests
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_equal_priorities_match_stock_frontier(env, seed):
    stock = env.run_stock(seed)
    env.run_episode(baseline_agent("fcfs"), seed)
    rl = env.last_simulator
    assert _timings(rl._all_requests) == _timings(stock._all_requests)
    # the workload actually exercises preemption
    assert sum(r.get_total_preemption_count() for r in stock._all_requests) > 0


def test_seeds_give_different_workloads():
    spec = WorkloadSpec(
        name="seeded", num_requests=30, qps=10,
        length_trace="data/processed_traces/splitwise_conv.csv", dummy_exec_ms=0.05,
    )
    env = FrontierSchedulingEnv(spec)
    a = env.run_episode(baseline_agent("fcfs"), 0).requests
    b = env.run_episode(baseline_agent("fcfs"), 1).requests
    a2 = env.run_episode(baseline_agent("fcfs"), 0).requests
    lengths = lambda rs: [(r.arrived_at, r.num_prefill_tokens, r.num_decode_tokens) for r in rs]
    assert lengths(a) == lengths(a2)
    assert lengths(a) != lengths(b)


def test_ranking_changes_the_schedule(env):
    fcfs = env.run_episode(baseline_agent("fcfs"), 0).summary
    sjf = env.run_episode(baseline_agent("sjf_prefill"), 0).summary
    rnd = env.run_episode(baseline_agent("random", seed=3), 0).summary
    assert fcfs["mean_e2e"] != rnd["mean_e2e"] or fcfs["mean_e2e"] != sjf["mean_e2e"]


def test_only_effective_decisions_are_kept(env):
    episode = env.run_episode(baseline_agent("random", seed=0), 0)
    assert episode.transitions
    assert all(t.admitted_ids for t in episode.transitions)
    assert episode.num_dropped_decisions > 0
    times = [t.time for t in episode.transitions]
    assert times == sorted(times)


def test_reward_components_sum_to_request_totals(env):
    episode = env.run_episode(baseline_agent("random", seed=1), 0)
    start = episode.transitions[0].time
    expected = sum(
        max(0.0, r.completed_at - max(r.arrived_at, start)) for r in episode.requests
    )
    got = sum(t.reward_components["e2e"] for t in episode.transitions)
    assert got == pytest.approx(expected, rel=1e-9)
    assert episode.total_reward == pytest.approx(-got, rel=1e-9)


class _OutputLengthGuard:
    def __init__(self, request):
        object.__setattr__(self, "_r", request)

    def __getattr__(self, name):
        assert "decode_tokens" not in name and "total_tokens" not in name, name
        return getattr(object.__getattribute__(self, "_r"), name)


def test_observation_never_reads_output_length(env):
    class Checking(OracleAgent):
        checked = 0

        def act_oracle(self, obs, waiting, scheduler):
            guarded = [_OutputLengthGuard(r) for r in waiting]
            build_observation(obs.time, guarded, scheduler, obs.num_priority_levels)
            Checking.checked += 1
            return np.zeros(len(waiting), dtype=np.int64), {}

    env.run_episode(Checking(), 0)
    assert Checking.checked > 0


def test_learned_agents_are_held_to_the_priority_levels(env):
    class TooHigh(BaseAgent):
        def act(self, obs):
            return np.full(obs.num_requests, obs.num_priority_levels), {}

    with pytest.raises(AssertionError):
        env.run_episode(TooHigh(), 0)
