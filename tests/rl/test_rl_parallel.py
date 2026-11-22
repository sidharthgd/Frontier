"""
Checks that the worker-pool options change only where and when episodes run,
not their results: split_groups, share_predictors (frontier/rl/preload.py),
the on-disk reference cache and reference prefetching (frontier/rl/rollout.py).

Run from the repo root (so ./data and ./cache resolve):
    python -m pytest tests/rl/test_rl_parallel.py -v
"""

import json
import os

import numpy as np
import pytest

from frontier.rl.model import config_for
from frontier.rl.observation import ObservationConfig
from frontier.rl.reward import RewardConfig, episode_cost
from frontier.rl.rollout import RolloutCollector, RolloutTask, WorkloadConfig
from frontier.rl.train import Trainer, load_config

WORKLOAD = WorkloadConfig("tiny", qps=20, num_blocks=4000, num_requests=60)
REWARD = RewardConfig(
    weights={"e2e_normalized": 1.0, "ttft_hinge": 0.01},
    threshold_mode="reference_window",
    calibration_quantile=0.95,
    reference_window_requests=100000,
    normalize_by_baselines=("fcfs",),
)
OBSERVATION = ObservationConfig()
MODEL = config_for(OBSERVATION, 16)
TINY = "workloads=[{name: tiny, qps: 20, num_blocks: 4000, num_requests: 60}]"


def _collector(tmp_path, **kwargs):
    return RolloutCollector(
        [WORKLOAD], REWARD, OBSERVATION, MODEL, str(tmp_path / "sim"), **kwargs
    )


def _summary(trajectory):
    return (
        trajectory.task,
        trajectory.cost,
        trajectory.fcfs_cost,
        [t.action.tolist() for t in trajectory.transitions],
        trajectory.request_rewards.tolist(),
    )


def test_reference_cache_round_trip(tmp_path):
    cache = tmp_path / "references"
    first = _collector(tmp_path, reference_cache_dir=str(cache)).env("tiny")
    episode = first.reference_episode("fcfs", 3)
    [directory] = os.listdir(cache)
    assert directory.startswith("tiny_")
    assert os.listdir(cache / directory) == ["fcfs__3.pkl"]

    # a fresh env (another worker or run) loads it instead of simulating
    second = _collector(tmp_path, reference_cache_dir=str(cache)).env("tiny")
    second.run_episode = None  # simulating would fail
    loaded = second.reference_episode("fcfs", 3)
    assert all(t.obs is None for t in loaded.transitions)
    assert loaded.summary == episode.summary
    context = first.reward_context(REWARD, 3)
    assert episode_cost(loaded, REWARD, context.request_thresholds) == episode_cost(
        episode, REWARD, context.request_thresholds
    )

    # other workload settings get their own directory
    other = WorkloadConfig("tiny", qps=21, num_blocks=4000, num_requests=60)
    collector = RolloutCollector(
        [other], REWARD, OBSERVATION, MODEL, str(tmp_path / "sim"),
        reference_cache_dir=str(cache),
    )
    collector.env("tiny").reference_episode("fcfs", 3)
    assert len(os.listdir(cache)) == 2


def test_pool_options_match_in_process(tmp_path):
    from frontier.rl.model import SchedulerModel
    import torch

    torch.manual_seed(0)
    state = SchedulerModel(MODEL).state_dict()
    tasks = [RolloutTask("tiny", seed, k, 100 * seed + k) for seed in (1, 2) for k in range(2)]
    later = [RolloutTask("tiny", 5, k, 500 + k) for k in range(2)]

    serial = _collector(tmp_path / "serial")
    expected = [_summary(t) for t in serial.collect(state, tasks + later)]

    cache = tmp_path / "references"
    pooled = _collector(
        tmp_path / "pooled",
        num_workers=3,
        split_groups=True,
        share_predictors=True,
        reference_cache_dir=str(cache),
        prefetch_workers=1,
    )
    try:
        first = pooled.collect(state, tasks, prefetch=later)
        second = pooled.collect(state, later)
    finally:
        pooled.close()
    assert [_summary(t) for t in first + second] == expected
    [directory] = os.listdir(cache)
    # (gap_metric also caches the oracle SJF reference)
    assert {"fcfs__1.pkl", "fcfs__2.pkl", "fcfs__5.pkl"} <= set(
        os.listdir(cache / directory)
    )


def test_trainer_with_prefetch_smoke(tmp_path):
    output = tmp_path / "ppo"
    config = load_config(
        "tests/rl/imitation_smoke.yaml",
        [
            TINY,
            "num_workers=2",
            "prefetch_workers=1",
            "split_groups=true",
            "share_predictors=true",
            f"reference_cache_dir={tmp_path / 'references'}",
            "iterations=2",
            "seeds_per_iteration=1",
            "group_size=2",
            "eval_every=2",
            "eval_seeds=[1000000]",
            "eval_decodings=[argmax]",
            "select_decoding=argmax",
            "ppo.epochs=1",
            f"output_dir={output}",
        ],
    )
    trainer = Trainer(config)
    try:
        trainer.train()
    finally:
        trainer.close()
    lines = [json.loads(x) for x in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(lines) == 2
    assert np.isfinite(lines[-1]["eval/argmax/improvement_over_fcfs"])
    assert (output / "best.pt").exists()
    # the eval FCFS reference was cached by the first iteration's prefetch
    [directory] = os.listdir(tmp_path / "references")
    assert "fcfs__1000000.pkl" in os.listdir(tmp_path / "references" / directory)


def test_resume_continues_the_run(tmp_path):
    import torch

    output = tmp_path / "ppo"
    common = [
        TINY, "num_workers=0", "seeds_per_iteration=1", "group_size=2",
        "eval_every=2", "eval_seeds=[1000000]", "eval_decodings=[argmax]",
        "select_decoding=argmax", "ppo.epochs=1", f"output_dir={output}",
    ]
    first = Trainer(load_config("tests/rl/imitation_smoke.yaml", common + ["iterations=2"]))
    try:
        first.train()
    finally:
        first.close()
    saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)

    resumed = Trainer(load_config(
        "tests/rl/imitation_smoke.yaml",
        common + ["iterations=3", f"resume_checkpoint={output / 'latest.pt'}"],
    ))
    assert resumed.iteration == 2
    assert resumed.best_eval == saved["best_eval"]
    assert resumed.optimizer.state_dict()["state"]  # Adam moments restored
    try:
        resumed.train()
    finally:
        resumed.close()
    lines = [json.loads(x) for x in (output / "metrics.jsonl").read_text().splitlines()]
    assert [line["iteration"] for line in lines] == [1, 2, 3]


def test_reference_memory_is_bounded_with_a_disk_cache(tmp_path):
    from frontier.rl.env import REFERENCES_IN_MEMORY

    env = _collector(tmp_path, reference_cache_dir=str(tmp_path / "references")).env("tiny")
    for seed in range(REFERENCES_IN_MEMORY + 2):
        env.reference_episode("fcfs", seed)
    assert len(env._reference_episodes) == REFERENCES_IN_MEMORY
    # an evicted one comes back from disk, identical
    env.run_episode = None
    assert np.isfinite(env.reference_episode("fcfs", 0).cost)


def test_sjf_wait_cap_serves_overdue_requests_first_in_arrival_order():
    from types import SimpleNamespace

    from frontier.rl.agents import baseline_agent

    agent = baseline_agent("sjf_prefill_tokens_cap20")
    assert agent._cap == 20.0 and agent._aging == 0.0
    assert baseline_agent("sjf_prefill_tokens_age200_cap40")._cap == 40.0
    assert baseline_agent("sjf_prefill_tokens_age200_cap40")._aging == 200.0
    # waited 30, 25, 5, 1 s; sizes 900, 100, 50, 10 tokens
    obs = SimpleNamespace(
        time=100.0,
        arrival_times=np.array([70.0, 75.0, 95.0, 99.0]),
        request_ids=np.arange(4),
    )
    requests = [SimpleNamespace(num_prefill_tokens=n) for n in (900, 100, 50, 10)]
    import frontier.rl.agents as agents

    original = agents.job_size
    agents.job_size = lambda r, key, limit: r.num_prefill_tokens
    try:
        ranks = agent.ranks(obs, requests, SimpleNamespace(max_num_batched_tokens=2048))
        plain = baseline_agent("sjf_prefill_tokens").ranks(
            obs, requests, SimpleNamespace(max_num_batched_tokens=2048)
        )
    finally:
        agents.job_size = original
    # overdue (0, 1) first by arrival, then 3 (10 tokens) before 2 (50 tokens)
    assert ranks.tolist() == [0, 1, 3, 2]
    assert plain.tolist() == [3, 2, 1, 0]


def test_sjf_relative_aging_scales_with_size():
    from types import SimpleNamespace

    import frontier.rl.agents as agents
    from frontier.rl.agents import baseline_agent

    agent = baseline_agent("sjf_prefill_tokens_rel30")
    assert agent._relative_aging == 30.0 and agent._cap is None
    # a 90k prompt that waited 300 s (effective 90k / 11 = 8.2k) against a new 10k
    obs = SimpleNamespace(time=300.0, arrival_times=np.array([0.0, 300.0]), request_ids=np.arange(2))
    requests = [SimpleNamespace(num_prefill_tokens=n) for n in (90_000, 10_000)]
    original = agents.job_size
    agents.job_size = lambda r, key, limit: r.num_prefill_tokens
    try:
        relative = agent.ranks(obs, requests, SimpleNamespace(max_num_batched_tokens=2048))
        absolute = baseline_agent("sjf_prefill_tokens_age200").ranks(
            obs, requests, SimpleNamespace(max_num_batched_tokens=2048))
    finally:
        agents.job_size = original
    assert relative.tolist() == [0, 1]  # the old long prompt goes first
    assert absolute.tolist() == [1, 0]  # 90k - 200 * 300 = 30k > 10k


def test_group_advantages_skip_rollouts_without_decisions():
    from frontier.rl.ppo import group_advantages

    full = (np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([-1.0, -1.0]), np.array([-2.0, -1.0]))
    other = (np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([-0.5, -0.5]), np.array([-1.0, -0.5]))
    empty = (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
    a, b, c = group_advantages([full, other, empty])
    assert np.allclose(a, [-1.0, -0.5]) and np.allclose(b, [1.0, 0.5]) and len(c) == 0
    # a lone non-empty rollout gets a zero baseline
    [solo, none] = group_advantages([full, empty])
    assert np.allclose(solo, full[3]) and len(none) == 0
