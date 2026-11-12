"""
The exact set-probability credit for the Plackett-Luce learner (train.pl_set_log_prob):
it equals the sum over orderings, its gradient is right, and a policy trained
with it learns which calls to admit in a toy problem.

    python -m pytest tests/rl/test_pl_set.py
"""

import itertools

import numpy as np
import pytest
import torch

from frontier.rl.train import pl_log_prob, pl_set_log_prob


def brute_force_set_prob(scores: np.ndarray, subset) -> float:
    w = np.exp(scores)
    total = 0.0
    for order in itertools.permutations(subset):
        remaining, p = w.sum(), 1.0
        for i in order:
            p *= w[i] / remaining
            remaining -= w[i]
        total += p
    return total


@pytest.mark.parametrize("n,m", [(4, 1), (6, 3), (6, 5), (7, 2)])
def test_set_probability_matches_the_sum_over_orderings(n, m):
    rng = np.random.default_rng(n * 10 + m)
    scores = rng.normal(scale=1.5, size=n)
    for subset in itertools.combinations(range(n), m):
        in_set = torch.zeros(1, n, dtype=torch.bool)
        in_set[0, list(subset)] = True
        got = pl_set_log_prob(torch.tensor(scores[None]), in_set, torch.ones(1, n, dtype=torch.bool)).exp().item()
        assert got == pytest.approx(brute_force_set_prob(scores, subset), rel=2e-3, abs=1e-7)


def test_set_probabilities_sum_to_one_and_padding_is_ignored():
    scores = torch.tensor([[0.3, -1.0, 2.0, 0.5, 9.9, -9.9]], dtype=torch.float64)
    valid = torch.tensor([[True, True, True, True, False, False]])
    total = 0.0
    for subset in itertools.combinations(range(4), 2):
        in_set = torch.zeros(1, 6, dtype=torch.bool)
        in_set[0, list(subset)] = True
        total += pl_set_log_prob(scores, in_set, valid).exp().item()
    assert total == pytest.approx(1.0, rel=2e-3)


def test_gradient_matches_finite_differences():
    torch.manual_seed(0)
    scores = torch.randn(1, 6, dtype=torch.float64, requires_grad=True)
    in_set = torch.tensor([[True, False, True, False, False, True]])
    valid = torch.ones(1, 6, dtype=torch.bool)
    pl_set_log_prob(scores, in_set, valid).sum().backward()
    eps = 1e-5
    for i in range(6):
        up, down = scores.detach().clone(), scores.detach().clone()
        up[0, i] += eps
        down[0, i] -= eps
        numeric = (pl_set_log_prob(up, in_set, valid) - pl_set_log_prob(down, in_set, valid)).item() / (2 * eps)
        assert scores.grad[0, i].item() == pytest.approx(numeric, rel=2e-3, abs=1e-5)


def _train_toy(likelihood: str, seed: int, steps: int = 400):
    """Choose the cheapest 2 of 8 calls; reward = minus the cost of the admitted pair."""
    torch.manual_seed(seed)
    n, m = 8, 2
    cost = torch.tensor([5.0, 1.0, 4.0, 2.0, 6.0, 3.0, 7.0, 8.0])
    scores = torch.zeros(1, n, requires_grad=True)
    optimizer = torch.optim.Adam([scores], lr=0.05)
    valid = torch.ones(1, n, dtype=torch.bool)
    grad_norms = []
    for _ in range(steps):
        group = 16
        gumbel = -torch.log(-torch.log(torch.rand(group, n).clamp(1e-9, 1 - 1e-9)))
        order = torch.argsort(-(scores.detach() + gumbel), dim=1)
        in_set = torch.zeros(group, n, dtype=torch.bool)
        in_set.scatter_(1, order[:, :m], True)
        reward = -(cost.unsqueeze(0) * in_set).sum(1)
        advantage = reward - reward.mean()  # group baseline
        batch_scores = scores.expand(group, n)
        if likelihood == "set":
            log_prob = pl_set_log_prob(batch_scores, in_set, valid.expand(group, n))
        else:
            log_prob = pl_log_prob(batch_scores, order, torch.full((group,), n), torch.full((group,), m))
        loss = -(advantage * log_prob).mean()
        optimizer.zero_grad()
        loss.backward()
        grad_norms.append(scores.grad.norm().item())
        optimizer.step()
    best = set(torch.argsort(cost)[:m].tolist())
    chosen = set(torch.argsort(-scores.detach()[0])[:m].tolist())
    return chosen == best, float(np.mean(grad_norms))


def test_toy_policy_learns_the_cheapest_calls_with_the_set_likelihood():
    results = [_train_toy("set", seed) for seed in range(4)]
    assert all(ok for ok, _ in results), results
