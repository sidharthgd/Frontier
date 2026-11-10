"""
The learned policy and its trainer (frontier/rl/train.py) on small sessions
with dummy execution times.

    python -m pytest tests/rl/test_rl_train.py
"""

import numpy as np
import pandas as pd
import pytest
import torch

from frontier.rl.agent_trace import SessionWorkload
from frontier.rl.agents import baseline_agent
from frontier.rl.env import FrontierSchedulingEnv, a100_llama31_8b
from frontier.rl.model import SchedulerModel
from frontier.rl.policy_input import GLOBAL_INPUTS, RUNNING_INPUTS, WAITING_INPUTS
from frontier.rl.train import (
    PolicyAgent,
    Step,
    Task,
    TrainConfig,
    Trajectory,
    _batch,
    group_advantages,
    imitation_loss,
    pair_agreement,
    pl_log_prob,
)


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


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return SchedulerModel(TrainConfig(output_dir="unused").model_config()).eval()


def test_policy_agent_runs_and_records_its_inputs(env, model):
    agent = PolicyAgent(model, "sample", 0, baseline_agent("sjf_uncached"), beta=0.0)
    episode = env.run_episode(agent, 0)
    assert episode.summary["num_completed"] == 36
    assert episode.transitions
    for t in episode.transitions:
        pi = t.info["pi"]
        assert pi.waiting_features.shape == (t.obs.num_requests, len(WAITING_INPUTS))
        assert pi.running_features.shape[1] == len(RUNNING_INPUTS)
        assert pi.global_features.shape == (len(GLOBAL_INPUTS),)
        assert np.isfinite(pi.waiting_features).all()
        assert sorted(t.info["teacher_rank"]) == list(range(t.obs.num_requests))


def test_teacher_with_beta_one_matches_the_teacher(env, model):
    teacher = env.run_episode(baseline_agent("sjf_uncached"), 1).summary
    mixed = env.run_episode(PolicyAgent(model, "rank", 0, baseline_agent("sjf_uncached"), beta=1.0), 1).summary
    # bucketing ranks into 16 levels keeps the order for queues of up to 16
    assert mixed["mean_e2e"] == pytest.approx(teacher["mean_e2e"], rel=1e-9)


def test_imitation_learns_the_teacher_order(env, model):
    agent = PolicyAgent(model, "rank", 0, baseline_agent("sjf_uncached"), beta=1.0)
    steps = []
    for seed in range(3):
        for t in env.run_episode(agent, seed).transitions:
            if t.obs.num_requests > 1:
                steps.append(Step(t.time, t.info["pi"], t.info["modeled"],
                                  np.asarray(t.info["levels"]), True, t.reward, t.info["teacher_rank"]))
    student = SchedulerModel(model.config)
    optimizer = torch.optim.Adam(student.policy_parameters(), lr=3e-3)
    for _ in range(150):
        batch, _, teacher = _batch(student, steps)
        mask = batch.waiting_mask & (teacher >= 0)
        loss = imitation_loss(student(batch, with_value=False).dist, teacher, mask).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        dist = student(batch, with_value=False).dist
    assert pair_agreement(dist.scores, teacher, mask) > 0.9


def _trajectory(seed: int, times, rewards, end: float) -> Trajectory:
    steps = [Step(t, None, np.arange(1), np.zeros(1, int), True, r) for t, r in zip(times, rewards)]
    return Trajectory(Task(4, seed, "policy"), steps, end, {})


def test_group_advantages_favor_the_cheaper_rollout():
    good = _trajectory(0, [0.0, 1.0, 2.0], [-1.0, -1.0, -1.0], 3.0)
    bad = _trajectory(0, [0.0, 1.0, 2.0], [-2.0, -2.0, -2.0], 3.0)
    other_seed = _trajectory(1, [0.0, 1.5], [-5.0, -5.0], 3.0)
    adv = group_advantages([good, bad, other_seed, other_seed], tau_s=100.0)
    assert (adv[0] > adv[1]).all()
    # a seed's group never mixes with another seed's
    assert np.allclose(adv[2], adv[3])


def test_plackett_luce_prefix_probabilities_sum_to_one():
    import itertools

    torch.manual_seed(0)
    scores = torch.randn(1, 4)
    total = 0.0
    for prefix in itertools.permutations(range(4), 2):
        rest = [i for i in range(4) if i not in prefix]
        order = torch.tensor([list(prefix) + rest])
        total += pl_log_prob(scores, order, torch.tensor([4]), torch.tensor([2])).exp().item()
    assert total == pytest.approx(1.0, abs=1e-5)


def test_gumbel_top_k_samples_the_plackett_luce_distribution(model):
    scores = torch.tensor([1.0, 0.0, -1.0])
    generator = torch.Generator().manual_seed(0)
    first = torch.zeros(3)
    for _ in range(20000):
        u = torch.rand(3, generator=generator).clamp(1e-9, 1 - 1e-9)
        first[torch.argmax(scores - torch.log(-torch.log(u)))] += 1
    assert torch.allclose(first / first.sum(), torch.softmax(scores, 0), atol=0.02)


def test_pl_agent_records_an_ordering_matching_its_priorities(env, model):
    agent = PolicyAgent(model, "pl", 0, None, 0.0, 2.0)
    episode = env.run_episode(agent, 0)
    assert episode.summary["num_completed"] == 36
    for t in episode.transitions:
        order = t.info["order"]
        assert sorted(order) == list(range(t.obs.num_requests))
        priorities = np.asarray(t.priorities)
        # the most urgent request in the sampled order has the best (lowest) level
        assert priorities[order[0]] == priorities.min()
