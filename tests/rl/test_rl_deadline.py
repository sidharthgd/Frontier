"""
The per-episode TTFT deadline (calls past it go first), its policy inputs, the
SLO-aware checkpoint score, the fixed seed pool and resuming a run.

    python -m pytest tests/rl/test_rl_deadline.py
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from frontier.rl.agent_trace import SessionWorkload
from frontier.rl.agents import baseline_agent
from frontier.rl.env import AgentPriorityPolicy, BaseAgent, FrontierSchedulingEnv, a100_llama31_8b
from frontier.rl.policy_input import GLOBAL_INPUTS, WAITING_INPUTS, policy_input
from frontier.rl.train import TRAIN_SEED_BASE, TrainConfig, Trainer, score


@pytest.fixture(scope="module")
def env(tmp_path_factory):
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
    workload = SessionWorkload(str(path), num_sessions=6, calls_per_session=6, min_calls=6)
    return FrontierSchedulingEnv(a100_llama31_8b(
        name="sessions", sessions=workload, enable_prefix_caching=True,
        dummy_exec_ms=5.0, num_blocks=1200,
    ))


def _request(id, arrived_at, first_token_at=None):
    request = SimpleNamespace(
        id=id, arrived_at=arrived_at, num_prefill_tokens=10, num_decode_tokens=5,
        session_id=0, priority=7,
    )
    request.set_priority = lambda p: setattr(request, "priority", p)
    return request, first_token_at


def test_apply_deadline_puts_only_overdue_calls_waiting_for_a_first_token_first():
    now = [100.0]
    policy = AgentPriorityPolicy(baseline_agent("fcfs"), 16, clock=lambda: now[0])
    policy.deadline_s = 30.0
    old, _ = _request(0, 60.0)                  # waited 40 s: overdue
    exactly, _ = _request(1, 70.0)              # waited exactly 30 s: overdue
    recent, _ = _request(2, 90.0)               # waited 10 s: not
    started, _ = _request(3, 10.0)              # overdue, but it already has its first token
    policy._record(started).first_token_at = 50.0
    policy.apply_deadline([old, exactly, recent, started])
    assert [r.priority for r in (old, exactly, recent, started)] == [-1, -1, 7, 7]
    policy.deadline_s = None
    recent2, _ = _request(4, 0.0)
    policy.apply_deadline([recent2])
    assert recent2.priority == 7


class Newest(BaseAgent):
    """Anti-FCFS: the newest arrival first."""

    def act(self, obs):
        rank = np.argsort(np.argsort(-obs.arrival_times))
        return np.minimum(rank, obs.num_priority_levels - 1).astype(np.int64), {}


def _timings(simulator):
    return sorted((r.arrived_at, r.prefill_completed_at, r.completed_at)
                  for r in simulator._all_requests)


def test_a_zero_deadline_makes_any_policy_fcfs(env):
    env.run_episode(baseline_agent("fcfs"), 0)
    fcfs = _timings(env.last_simulator)
    env.run_episode(Newest(), 0)
    assert _timings(env.last_simulator) != fcfs  # the agent alone does reorder
    env.run_episode(Newest(), 0, deadline_s=0.0)
    assert _timings(env.last_simulator) == fcfs


def test_deadline_inputs_follow_the_deadline(env):
    seen = []

    class Probe(BaseAgent):
        def act(self, obs):
            seen.append((obs, policy_input(obs)))
            return np.zeros(obs.num_requests, dtype=np.int64), {}

    env.run_episode(Probe(), 1)
    assert all(pi.waiting_features.shape[1] == len(WAITING_INPUTS) for _, pi in seen)
    for obs, pi in seen:
        assert pi.global_features.shape == (len(GLOBAL_INPUTS),)
        assert obs.global_feature("deadline_s") == -1
        assert (pi.waiting_features[:, WAITING_INPUTS.index("time_to_deadline_frac")] == 1).all()
        assert (pi.waiting_features[:, WAITING_INPUTS.index("overdue")] == 0).all()
        assert pi.global_features[GLOBAL_INPUTS.index("log_deadline_s")] == 0

    seen.clear()
    env.run_episode(Probe(), 1, deadline_s=2.0)
    assert seen
    saw_overdue = False
    for obs, pi in seen:
        assert obs.global_feature("deadline_s") == 2.0
        wait = obs.waiting_feature("wait_time")
        frac = pi.waiting_features[:, WAITING_INPUTS.index("time_to_deadline_frac")]
        overdue = pi.waiting_features[:, WAITING_INPUTS.index("overdue")]
        assert np.allclose(frac, np.clip((2.0 - wait) / 2.0, -2, 1), atol=1e-6)
        assert (overdue == (wait >= 2.0)).all()
        g = pi.global_features
        assert g[GLOBAL_INPUTS.index("log_deadline_s")] == pytest.approx(np.log1p(2.0), rel=1e-6)
        assert g[GLOBAL_INPUTS.index("overdue_fraction")] == pytest.approx(overdue.mean(), abs=1e-6)
        assert g[GLOBAL_INPUTS.index("oldest_wait_over_deadline")] == pytest.approx(
            min(wait.max() / 2.0, 3.0), rel=1e-5)
        saw_overdue |= bool(overdue.any())
        assert np.isfinite(pi.waiting_features).all() and np.isfinite(g).all()
    assert saw_overdue


def test_score_ignores_tails_within_the_margin_and_charges_the_rest():
    config = TrainConfig(output_dir="unused", score_tail_margin=10.0, score_tail_weight=0.5)

    def metrics(mean, tails):
        out = {"mean_session_serving": mean}
        for load, (ttft, e2e) in tails.items():
            out[f"s{load}/p99_ttft"], out[f"s{load}/p99_e2e"] = ttft, e2e
        return out

    matches_fcfs = metrics(-15.0, {12: (5.0, 0.0), 24: (10.0, -3.0)})
    assert score(matches_fcfs, config) == pytest.approx(-15.0)
    # sjf_uncached-like: great mean, tails far past FCFS
    starving = metrics(-40.0, {12: (58.0, 0.0), 24: (364.0, 52.0)})
    excess = np.mean([48.0, 0.0, 354.0, 42.0])
    assert score(starving, config) == pytest.approx(-40.0 + 0.5 * excess)
    assert score(starving, config) > score(matches_fcfs, config)


def test_seed_pool_cycles_through_fixed_seeds_and_resume_continues_it(tmp_path):
    config = TrainConfig(output_dir=str(tmp_path), num_workers=1, seed_pool=5)
    first = Trainer(config)
    try:
        assert first.seeds(3) == [TRAIN_SEED_BASE, TRAIN_SEED_BASE + 1, TRAIN_SEED_BASE + 2]
        first.best = -12.5
        optimizer = torch.optim.Adam(first.model.policy_parameters(), lr=1e-3)
        first.save_state(7, optimizer)
        expected = first.seeds(4)
        draw = first.rng.integers(1_000_000)
    finally:
        first.pool.close()
    second = Trainer(TrainConfig(output_dir=str(tmp_path), num_workers=1, seed_pool=5, resume=True))
    try:
        state = second.load_state()
        assert state["iteration"] == 7 and second.best == -12.5
        assert second.seeds(4) == expected == [TRAIN_SEED_BASE + 3, TRAIN_SEED_BASE + 4,
                                                TRAIN_SEED_BASE, TRAIN_SEED_BASE + 1]
        assert second.rng.integers(1_000_000) == draw
        for a, b in zip(first.model.state_dict().values(), second.model.state_dict().values()):
            assert torch.equal(a, b)
    finally:
        second.pool.close()
