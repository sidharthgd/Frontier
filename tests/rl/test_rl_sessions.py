"""
Closed-loop agent sessions in the RL env (frontier/rl/agent_trace.py), with
dummy execution times and a small synthetic session file.

    python -m pytest tests/rl/test_rl_sessions.py
"""

import numpy as np
import pandas as pd
import pytest

from frontier.rl.agent_trace import SessionWorkload
from frontier.rl.agents import baseline_agent
from frontier.rl.env import FrontierSchedulingEnv, a100_llama31_8b


@pytest.fixture(scope="module")
def calls_csv(tmp_path_factory):
    rng = np.random.default_rng(0)
    rows, chain = [], 0
    for session in range(6):
        chain += 1
        prompt = int(rng.integers(2000, 6000))
        for call in range(8):
            output = int(rng.integers(20, 200))
            rows.append((session, call, "claude", chain, prompt, output,
                         float(rng.uniform(0, 2)), prompt))
            prompt += output + int(rng.integers(100, 600))
    path = tmp_path_factory.mktemp("sessions") / "calls.csv"
    pd.DataFrame(rows, columns=[
        "session", "call", "provider", "chain", "prompt_tokens", "output_tokens",
        "gap_after_s", "real_input_tokens",
    ]).to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def env(calls_csv):
    workload = SessionWorkload(calls_csv, num_sessions=4, calls_per_session=6, min_calls=6)
    return FrontierSchedulingEnv(a100_llama31_8b(
        name="sessions", sessions=workload, enable_prefix_caching=True,
        dummy_exec_ms=1.0, num_blocks=3000,
    ))


def test_calls_arrive_after_their_predecessor_finishes(env):
    episode = env.run_episode(baseline_agent("fcfs"), 0)
    assert episode.summary["num_completed"] == 24
    by_session = {}
    for r in episode.requests:
        by_session.setdefault(r.session_id, []).append(r)
    assert len(by_session) == 4
    for calls in by_session.values():
        calls.sort(key=lambda r: r.arrived_at)
        for previous, current in zip(calls, calls[1:]):
            assert current.arrived_at >= previous.completed_at


def test_sessions_hit_the_prefix_cache(env):
    summary = env.run_episode(baseline_agent("fcfs"), 0).summary
    assert summary["cached_prefill_fraction"] > 0.5
    assert summary["mean_session_serving"] > 0


@pytest.mark.parametrize("seed", [0, 1])
def test_fcfs_matches_stock_frontier_on_sessions(env, seed):
    def timings(requests):
        return sorted((r.arrived_at, r.prefill_completed_at, r.completed_at) for r in requests)

    stock = env.run_stock(seed)
    env.run_episode(baseline_agent("fcfs"), seed)
    assert timings(env.last_simulator._all_requests) == timings(stock._all_requests)


def test_cache_aware_rule_sees_cached_prefixes(env):
    seen = []

    class Recorder:
        unbounded_priorities = True

        def act(self, obs):
            seen.append(obs.waiting_feature("cached_prompt_tokens").max())
            return np.zeros(obs.num_requests, dtype=np.int64), {}

    from frontier.rl.env import BaseAgent
    agent = type("RecorderAgent", (BaseAgent,), dict(Recorder.__dict__))()
    env.run_episode(agent, 0)
    assert max(seen) > 0


def test_eviction_distance_counts_blocks_before_loss(env):
    from frontier.rl.env import AgentPriorityPolicy
    from frontier.rl.scheduler import replica_schedulers

    seen = []

    class Probe(type(baseline_agent("fcfs"))):
        def act(self, obs):
            seen.append(obs.waiting_feature("evict_distance_blocks").copy())
            return super().act(obs)

    env.run_episode(Probe(), 0)
    distances = np.concatenate(seen)
    cached = distances[distances >= 0]
    assert len(cached), "later calls of a session should find evictable cached blocks"
    assert cached.max() < 3000  # num_blocks of this fixture
    assert (distances >= -1).all()


def test_evict_and_deadline_rules_run(env):
    for name in ("sjf_uncached_evict0.5_300", "sjf_uncached_deadline5"):
        summary = env.run_episode(baseline_agent(name), 0).summary
        assert summary["num_completed"] == 24


def test_shared_prefix_is_one_chain_across_sessions(calls_csv):
    plain = SessionWorkload(calls_csv, num_sessions=4, calls_per_session=6, min_calls=6)
    shared = SessionWorkload(calls_csv, num_sessions=4, calls_per_session=6, min_calls=6,
                             shared_prefix_tokens=(("claude", 1600),))
    _, _, plain_requests = plain.build(0)
    _, _, requests = shared.build(0)
    heads = {tuple(r.block_hash_ids[:100]) for r in requests}
    assert len(heads) == 1  # every session starts with the same 100 blocks
    assert len({tuple(r.block_hash_ids[:100]) for r in plain_requests}) > 1
    for a, b in zip(plain_requests, requests):
        assert len(a.block_hash_ids) == len(b.block_hash_ids)
        # beyond the shared prefix a context still continues its own chain
        assert a.block_hash_ids[100:] == b.block_hash_ids[100:]
