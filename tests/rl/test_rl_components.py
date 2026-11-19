"""
Checks for the RL stack on Frontier: request history windows, observation
features, reward components and admission action bounds, on short FrontierEnv
episodes. (Vidur's own scheduler checks stayed in the Vidur repo with its
simulator backend.)

Run from the repo root:
    python -m pytest tests/rl/test_rl_components.py -v
"""

import dataclasses
import math

import numpy as np
import pytest

from frontier.rl.agents import FCFSAgent, RandomAgent
from frontier.rl.reward import RewardConfig, per_request_reward_components
from frontier.rl.frontier_env import _RequestView
from frontier.rl.history import SlidingWindow
from frontier.rl.observation import ObservationConfig
from frontier.rl.reward import PER_REQUEST_COMPONENTS, episode_cost_components
from frontier.rl.rollout import WorkloadConfig

WORKLOAD_SEED = 0

THRESHOLDS = dict(
    tbt_slo=0.03,
    ttft_threshold=0.5,
    e2e_threshold=5.0,
    starvation_threshold=0.3,
)

# small KV cache so the queue builds up and requests get preempted
WORKLOAD = WorkloadConfig("tiny", qps=20, num_blocks=4000, num_requests=120)


def _env(tmp_path_factory, reward_config):
    return WORKLOAD.make_env(
        reward_config, ObservationConfig(), 16, str(tmp_path_factory.mktemp("sim"))
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return _env(tmp_path_factory, RewardConfig(**THRESHOLDS))


@pytest.fixture(scope="module")
def random_episode(env):
    return env.run_episode(RandomAgent(seed=0), workload_seed=WORKLOAD_SEED)


def test_sliding_window_matches_brute_force():
    rng = np.random.default_rng(0)
    values = rng.normal(size=500)
    window = SlidingWindow(37)
    for i, value in enumerate(values):
        window.add(value)
        recent = values[max(0, i - 36) : i + 1]
        mean, low, high = window.stats()
        assert math.isclose(mean, recent.mean(), abs_tol=1e-9)
        assert low == recent.min() and high == recent.max()


# ---------------------------------------------------------------------------
# observation features
# ---------------------------------------------------------------------------

FORBIDDEN_REQUEST_ATTRIBUTES = {
    "num_decode_tokens",
    "_num_decode_tokens",
    "total_tokens",
    "pd_ratio",
    "size",
    "completed_at",
    "e2e_time",
    "e2e_time_normalized",
    "execution_time_normalized",
    "model_execution_time_normalized",
}


class OutputLengthGuard:
    """Proxy that fails on any attribute derived from the output length."""

    def __init__(self, request):
        object.__setattr__(self, "_request", request)

    def __getattr__(self, name):
        assert name not in FORBIDDEN_REQUEST_ATTRIBUTES, f"feature read {name}"
        return getattr(object.__getattribute__(self, "_request"), name)


def test_features_never_read_output_length():
    # features see Frontier requests only through _RequestView
    fields = {f.name for f in dataclasses.fields(_RequestView)}
    assert not fields & FORBIDDEN_REQUEST_ATTRIBUTES


def test_observation_shapes(env, random_episode):
    config = env.observation_config
    assert random_episode.transitions
    for t in random_episode.transitions:
        obs = t.obs
        assert obs.waiting_features.shape == (
            obs.num_requests,
            config.num_waiting_features,
        )
        assert obs.running_features.shape == (
            obs.num_running,
            config.num_running_features,
        )
        assert obs.global_features.shape == (config.num_global_features,)
        assert obs.rank_mask.all()
        assert math.isclose(obs.time, t.time)
        assert np.isfinite(obs.waiting_features).all()
        assert np.isfinite(obs.global_features).all()


# ---------------------------------------------------------------------------
# admission action bounds
# ---------------------------------------------------------------------------


def test_action_bounds_reproduce_admission(random_episode):
    rng = np.random.default_rng(0)
    transitions = random_episode.transitions
    assert any(t.boundary_index is not None for t in transitions)

    for t in transitions:
        low, high = t.action_low, t.action_high
        assert ((low <= t.action) & (t.action <= high)).all()

        if t.boundary_index is None:
            assert t.admitted.all()
            continue
        assert not t.admitted[t.boundary_index]
        if np.array_equal(low, t.action) and np.array_equal(high, t.action):
            # admission skipped a request: only the exact levels reproduce it
            continue

        # any action inside the bounds puts exactly the admitted requests ahead
        # of the boundary request in the (priority, arrival) order
        for _ in range(5):
            action = rng.integers(low, high + 1)
            keys = t.obs.ranking_keys(action)
            b = t.boundary_index
            ahead = np.array([keys[i] < keys[b] for i in range(len(action))])
            assert np.array_equal(ahead, t.admitted)


# ---------------------------------------------------------------------------
# rewards
# ---------------------------------------------------------------------------


def test_hinge_components_match_request_metrics(env, random_episode):
    requests = random_episode.requests
    assert all(r.completed for r in requests)
    totals = episode_cost_components(random_episode, env.reward_config)

    ttft = np.array([r.prefill_completed_at - r.arrived_at for r in requests])
    e2e = np.array([r.completed_at - r.arrived_at for r in requests])
    gaps = random_episode.token_gaps[~np.isnan(random_episode.token_gaps)]
    expected = {
        "ttft_hinge": np.maximum(0, ttft - THRESHOLDS["ttft_threshold"]).sum(),
        "e2e_hinge": np.maximum(0, e2e - THRESHOLDS["e2e_threshold"]).sum(),
        "starvation": (
            np.maximum(0, ttft - THRESHOLDS["starvation_threshold"]) ** 2
        ).sum(),
        "tbt_violation": np.maximum(0, gaps - THRESHOLDS["tbt_slo"]).sum(),
    }
    for name, value in expected.items():
        assert value > 0, name
        assert math.isclose(totals[name], value, rel_tol=1e-9), name


def test_token_log_matches_requests(random_episode):
    for r in random_episode.requests:
        mask = random_episode.token_request_ids == r.id
        times = random_episode.token_times[mask]
        assert times[0] == r.prefill_completed_at
        assert times[-1] == r.completed_at


def test_per_request_rewards_sum_to_transition_rewards(env, random_episode):
    components = per_request_reward_components(random_episode, env.reward_config)
    num_transitions = len(random_episode.transitions)
    for name in PER_REQUEST_COMPONENTS:
        assert components[name].shape == (
            num_transitions,
            len(random_episode.requests),
        )
        per_transition = components[name].sum(axis=1)
        recorded = np.array(
            [t.reward_components[name] for t in random_episode.transitions]
        )
        np.testing.assert_allclose(per_transition, recorded, rtol=1e-7, atol=1e-9)


def test_calibrated_thresholds(tmp_path_factory):
    env = _env(
        tmp_path_factory,
        RewardConfig(
            ttft_threshold=None,
            tbt_slo=None,
            calibration_quantile=0.9,
            calibration_seeds=(1000,),
        ),
    )
    reward_config = env.reward_config
    assert not reward_config.needs_calibration
    # every threshold left as None was filled in
    for name in ("tbt_slo", "ttft_threshold", "e2e_threshold", "starvation_threshold"):
        assert getattr(reward_config, name) > 0, name

    episode = env.run_episode(FCFSAgent(), workload_seed=1000)
    ttft = np.array([r.prefill_completed_at - r.arrived_at for r in episode.requests])
    assert math.isclose((ttft > reward_config.ttft_threshold).mean(), 0.1, abs_tol=0.02)


def test_ttft_ref_sq_charges_waits_beyond_each_calls_fcfs_ttft(tmp_path_factory):
    reward_config = RewardConfig(
        weights={"e2e": 1.0, "ttft_ref_sq": 1.0}, ttft_ref_k=0.5, ttft_ref_floor=0.05
    )
    env = _env(tmp_path_factory, reward_config)
    episode = env.run_episode(RandomAgent(seed=0), workload_seed=WORKLOAD_SEED)
    reference = {
        r.id: r.prefill_completed_at - r.arrived_at
        for r in env.reference_episode("fcfs", WORKLOAD_SEED).requests
    }
    ttft = np.array([r.prefill_completed_at - r.arrived_at for r in episode.requests])
    tau = np.array([max(0.05, 0.5 * reference[r.id]) for r in episode.requests])
    expected = (np.maximum(0, ttft - tau) ** 2).sum()
    assert expected > 0
    assert math.isclose(
        episode_cost_components(episode, reward_config)["ttft_ref_sq"], expected, rel_tol=1e-9
    )
    per_request = per_request_reward_components(episode, reward_config)["ttft_ref_sq"]
    np.testing.assert_allclose(
        per_request.sum(axis=1),
        [t.reward_components["ttft_ref_sq"] for t in episode.transitions],
        rtol=1e-7, atol=1e-9,
    )
