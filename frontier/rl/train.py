"""
Train a priority policy on agent sessions (frontier/rl/agent_trace.py) with
the A100 + Llama-3.1-8B setup, prefix caching on:

    python -m frontier.rl.train --output-dir rl_runs/train/agents_v1

Two phases, as in our Vidur fork (vidur/rl/imitation.py, vidur/rl/ppo.py):

1. Imitation (DAgger) of a teacher, by default sjf_uncached, the best rule on
   these sessions. The first rollouts are the teacher's; later ones mix
   teacher and student (teacher with probability beta, decaying), and every
   decision is labeled with the teacher's ranking. Loss: the teacher's ranking
   cut into the priority levels (level NLL) plus a pairwise ranking loss on
   the scores.
2. PPO fine-tuning with group advantages: group_size rollouts per workload
   seed; a decision at time t is compared with the other rollouts' mean
   discounted return-to-go from t (time-aligned leave-one-out). The clipped
   surrogate is per request (each request's level is one action sharing its
   decision's advantage), on decisions that left something waiting. An
   optional imitation term anchors the policy to the teacher.

Reward (frontier/rl/reward.py): -(e2e + ttft_hinge_weight * ttft_hinge), i.e.
the calls' total time in the system (= the sessions' serving time) plus the
time calls spend waiting for a first token beyond ttft_threshold seconds.
Discounting is continuous in time: exp(-duration / tau_s) per decision.

Eval: rank decoding on held-out workload seeds (train seeds start at
TRAIN_SEED_BASE), against FCFS and the teacher on the same seeds. Metrics go
to <output_dir>/metrics.jsonl, weights to best.pt / latest.pt.
"""

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from frontier.rl.model import ModelConfig, SchedulerModel, collate, full_actions, modeled_indices
from frontier.rl.policy_input import (
    GLOBAL_INPUTS,
    RUNNING_INPUTS,
    WAITING_INPUTS,
    PolicyInput,
    policy_input,
)

TRAIN_SEED_BASE = 1000
CALLS = "rl_runs/agent_traces/tracelab_calls.csv"


@dataclass
class TrainConfig:
    output_dir: str
    # workloads: concurrent sessions, each a window of calls_per_session calls
    calls_csv: str = CALLS
    loads: Tuple[int, ...] = (12, 16, 24)
    calls_per_session: int = 20
    provider: Optional[str] = None  # "claude" | "codex" | None
    # contexts of a provider share its system prompt in the prefix cache
    shared_prefix: bool = True
    eval_seeds: Tuple[int, ...] = (0, 1, 2, 3)
    num_workers: int = 8
    seed: int = 0
    # reward
    ttft_threshold: float = 30.0
    ttft_hinge_weight: float = 2.0
    tau_s: float = 120.0
    # model
    num_priority_levels: int = 16
    encoder: str = "transformer"
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    init_middle_level_prob: float = 0.25
    init_checkpoint: Optional[str] = None
    # imitation
    teacher: str = "sjf_uncached"
    imitation_iterations: int = 4
    imitation_initial_seeds: int = 8  # per load, teacher only
    imitation_seeds: int = 4  # per load per iteration, mixed
    beta_start: float = 0.5
    beta_decay: float = 0.5
    imitation_epochs: int = 3
    imitation_lr: float = 3e-4
    # ppo
    ppo_iterations: int = 40
    seeds_per_iteration: int = 2  # per load
    group_size: int = 4
    learning_rate: float = 2e-4
    clip: float = 0.2
    epochs: int = 4
    minibatch_size: int = 256
    entropy_coef: float = 0.001
    imitation_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.03
    eval_every: int = 5
    # "ppo": per-request levels sampled independently, clipped surrogate on the
    # sum of their log-probabilities. "pl": a Plackett-Luce ranking policy, the
    # model's scores are the logits of an ordering sampled by Gumbel-top-k
    # (score / pl_temperature + Gumbel noise); the surrogate uses the
    # log-probability of the admitted prefix of that ordering, so only choices
    # that decided who ran count.
    method: str = "ppo"
    pl_temperature: float = 4.0

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            num_waiting_features=len(WAITING_INPUTS),
            num_running_features=len(RUNNING_INPUTS),
            num_global_features=len(GLOBAL_INPUTS),
            num_priority_levels=self.num_priority_levels,
            encoder=self.encoder,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            init_middle_level_prob=self.init_middle_level_prob,
        )


# --- agents --------------------------------------------------------------------


def _ranks(keys: np.ndarray, pi: PolicyInput) -> np.ndarray:
    order = np.lexsort((pi.request_ids, pi.arrival_times, keys))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order))
    return ranks


def bucket_ranks(ranks: np.ndarray, num_levels: int) -> np.ndarray:
    return ranks * num_levels // max(1, len(ranks))


class PolicyAgent:
    """
    Acts with a SchedulerModel. decoding: "sample" (training), or "rank": sort
    by expected level, ties by arrival, and cut the ranking into the levels.
    With a teacher, every decision is labeled with the teacher's ranking, and
    the teacher acts with probability beta (DAgger).
    """

    unbounded_priorities = False

    def __init__(self, model, decoding="sample", seed=0, teacher=None, beta=0.0, temperature=1.0):
        self._temperature = temperature
        self._model = model
        self._decoding = decoding
        self._generator = torch.Generator().manual_seed(seed)
        self._rng = np.random.default_rng(seed)
        self._teacher = teacher
        self._beta = beta

    @torch.no_grad()
    def act(self, obs):
        pi = policy_input(obs)
        info = {"pi": pi}
        if self._teacher is not None:
            info["teacher_rank"] = _ranks(np.asarray(self._teacher.act(obs)[0], float), pi)
            if self._rng.random() < self._beta:
                levels = bucket_ranks(info["teacher_rank"], pi.num_priority_levels)
                info["modeled"] = np.arange(pi.num_requests)
                info["levels"] = levels
                return levels, info
        modeled = modeled_indices(pi, self._model.config.max_queue_len)
        dist = self._model(collate([pi], self._model.config, [modeled]), with_value=False).dist
        if self._decoding == "pl":
            scores = dist.scores[0, : len(modeled)] / self._temperature
            u = torch.rand(scores.shape, generator=self._generator).clamp(1e-9, 1 - 1e-9)
            order = torch.argsort(-(scores - torch.log(-torch.log(u)))).numpy()
            ranks = np.empty(len(order), dtype=np.int64)
            ranks[order] = np.arange(len(order))
            levels = bucket_ranks(ranks, dist.num_levels)
            info["order"] = order  # positions within `modeled`, first = most urgent
        elif self._decoding == "sample":
            levels = dist.sample(self._generator)[0].numpy()
        else:
            expected = (dist.level_log_probs()[0].exp() * torch.arange(dist.num_levels)).sum(-1)
            sub = PolicyInput(
                pi.num_priority_levels, pi.request_ids[modeled], pi.arrival_times[modeled],
                pi.waiting_features[modeled], pi.running_features, pi.global_features,
            )
            levels = bucket_ranks(_ranks(expected[: len(modeled)].numpy(), sub), dist.num_levels)
        action = full_actions(pi, modeled, levels[: len(modeled)])
        info["modeled"] = modeled
        info["levels"] = action
        return action, info


# --- rollouts ------------------------------------------------------------------


@dataclass
class Task:
    load: int
    workload_seed: int
    kind: str  # "policy" | a baseline name
    agent_seed: int = 0
    decoding: str = "sample"
    teacher: bool = False
    beta: float = 0.0


@dataclass
class Step:
    time: float
    pi: PolicyInput
    modeled: np.ndarray
    levels: np.ndarray
    relevant: bool  # something was left waiting: the ranking mattered
    reward: float
    teacher_rank: Optional[np.ndarray] = None
    order: Optional[np.ndarray] = None  # Plackett-Luce ordering (positions in `modeled`)
    num_admitted: int = 0


@dataclass
class Trajectory:
    task: Task
    steps: List[Step]
    end_time: float
    summary: Dict[str, float]


_WORKER: Dict[str, object] = {}


def _env(config: TrainConfig, load: int):
    envs = _WORKER.setdefault("envs", {})
    if load not in envs:
        from frontier.rl.agent_trace import DEFAULT_SHARED_PREFIX, SessionWorkload
        from frontier.rl.env import FrontierSchedulingEnv, a100_llama31_8b
        from frontier.rl.reward import RewardConfig

        workload = SessionWorkload(
            config.calls_csv, num_sessions=load, calls_per_session=config.calls_per_session,
            session_rate=load / 60.0, provider=config.provider,
            shared_prefix_tokens=DEFAULT_SHARED_PREFIX if config.shared_prefix else (),
        )
        envs[load] = FrontierSchedulingEnv(
            a100_llama31_8b(
                name=f"sessions{load}", sessions=workload,
                enable_prefix_caching=True, prediction_rounding_bits=5,
            ),
            reward_config=RewardConfig(
                weights={"e2e": 1.0, "ttft_hinge": config.ttft_hinge_weight},
                ttft_threshold=config.ttft_threshold,
            ),
            num_priority_levels=config.num_priority_levels,
            output_dir=f"/tmp/frontier_rl_train_{os.getpid()}",
        )
    return envs[load]


def _init_worker(config: TrainConfig) -> None:
    torch.set_num_threads(1)
    _WORKER["config"] = config
    _WORKER["model"] = SchedulerModel(config.model_config()).eval()


def run_task(payload) -> Trajectory:
    state_dict, task = payload
    from frontier.rl.agents import baseline_agent

    config: TrainConfig = _WORKER["config"]
    env = _env(config, task.load)
    if task.kind == "policy":
        model = _WORKER["model"]
        model.load_state_dict(state_dict)
        teacher = baseline_agent(config.teacher) if task.teacher else None
        agent = PolicyAgent(model, task.decoding, task.agent_seed, teacher, task.beta,
                            config.pl_temperature)
    else:
        agent = baseline_agent(task.kind, task.agent_seed)
    episode = env.run_episode(agent, task.workload_seed)
    steps = []
    if task.kind == "policy":
        for t in episode.transitions:
            steps.append(Step(
                time=t.time, pi=t.info["pi"], modeled=t.info["modeled"],
                levels=np.asarray(t.info["levels"]), relevant=t.boundary_id is not None,
                reward=t.reward, teacher_rank=t.info.get("teacher_rank"),
                order=t.info.get("order"), num_admitted=len(t.admitted_ids),
            ))
    return Trajectory(task, steps, episode.end_time, episode.summary)


class Pool:
    def __init__(self, config: TrainConfig) -> None:
        self._pool = mp.get_context("spawn").Pool(config.num_workers, _init_worker, (config,))

    def run(self, state_dict, tasks: Sequence[Task]) -> List[Trajectory]:
        state_dict = {k: v.detach().cpu() for k, v in state_dict.items()} if state_dict else None
        return self._pool.map(run_task, [(state_dict, t) for t in tasks], chunksize=1)

    def close(self) -> None:
        self._pool.close()
        self._pool.join()


# --- losses --------------------------------------------------------------------


def _batch(model: SchedulerModel, steps: Sequence[Step]):
    batch = collate([s.pi for s in steps], model.config, [s.modeled for s in steps], 8)
    width = batch.waiting.shape[1]
    levels = np.zeros((len(steps), width), np.int64)
    teacher = np.full((len(steps), width), -1, np.int64)
    for b, s in enumerate(steps):
        k = len(s.modeled)
        levels[b, :k] = s.levels[s.modeled]
        if s.teacher_rank is not None:
            # the teacher's order among the modeled requests
            teacher[b, :k] = np.argsort(np.argsort(s.teacher_rank[s.modeled]))
    return batch, torch.from_numpy(levels), torch.from_numpy(teacher)


def imitation_loss(dist, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample level NLL of the bucketed teacher ranking + pairwise rank loss."""
    count = mask.sum(1).clamp(min=1)
    target = (teacher.clamp(min=0) * dist.num_levels) // count.unsqueeze(1)
    nll = -dist.level_log_probs().gather(-1, target.unsqueeze(-1)).squeeze(-1)
    level_loss = (nll * mask).sum(1) / count
    before = (teacher.unsqueeze(2) < teacher.unsqueeze(1)) & mask.unsqueeze(2) & mask.unsqueeze(1)
    pair = F.softplus(dist.scores.unsqueeze(2) - dist.scores.unsqueeze(1))
    rank_loss = (pair * before).sum((1, 2)) / before.sum((1, 2)).clamp(min=1)
    return level_loss + rank_loss


def _minibatches(n: int, size: int, rng: np.random.Generator) -> List[np.ndarray]:
    order = rng.permutation(n)
    return [order[i : i + size] for i in range(0, n, size)]


def train_imitation(model, optimizer, dataset: List[Step], config: TrainConfig, rng) -> Dict[str, float]:
    model.train()
    losses, agreement = [], []
    for _ in range(config.imitation_epochs):
        agreement = []
        for idx in _minibatches(len(dataset), config.minibatch_size, rng):
            steps = [dataset[i] for i in idx]
            batch, _, teacher = _batch(model, steps)
            mask = batch.waiting_mask & (teacher >= 0)
            dist = model(batch, with_value=False).dist
            loss = imitation_loss(dist, teacher, mask).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy_parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            agreement.append(pair_agreement(dist.scores.detach(), teacher, mask))
    model.eval()
    return {"imitation_loss": float(np.mean(losses[-len(agreement):])),
            "pair_agreement": float(np.nanmean(agreement))}


def pair_agreement(scores: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> float:
    """Share of teacher-ordered request pairs the scores order the same way."""
    before = (teacher.unsqueeze(2) < teacher.unsqueeze(1)) & mask.unsqueeze(2) & mask.unsqueeze(1)
    agree = (scores.unsqueeze(2) < scores.unsqueeze(1)) & before
    total = before.sum().item()
    return agree.sum().item() / total if total else float("nan")


def _returns(traj: Trajectory, tau_s: float):
    times = np.array([s.time for s in traj.steps])
    durations = np.diff(np.append(times, traj.end_time))
    rewards = np.array([s.reward for s in traj.steps])
    gammas = np.exp(-durations / tau_s)
    returns = np.zeros(len(rewards))
    running = 0.0
    for k in range(len(rewards) - 1, -1, -1):
        running = rewards[k] + gammas[k] * running
        returns[k] = running
    return times, durations, rewards, returns


def _return_at(times, durations, rewards, returns, t: float) -> float:
    k = int(np.searchsorted(times, t, side="right")) - 1
    if k < 0:
        return float(returns[0])
    fraction = min(1.0, (t - times[k]) / durations[k]) if durations[k] > 0 else 1.0
    return float(returns[k] - rewards[k] * fraction)


def group_advantages(trajectories: Sequence[Trajectory], tau_s: float) -> List[np.ndarray]:
    groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, t in enumerate(trajectories):
        groups[(t.task.load, t.task.workload_seed)].append(i)
    per = [_returns(t, tau_s) for t in trajectories]
    advantages: List[Optional[np.ndarray]] = [None] * len(trajectories)
    for members in groups.values():
        for i in members:
            others = [per[j] for j in members if j != i and len(per[j][0])]
            times, _, _, returns = per[i]
            baseline = np.array([np.mean([_return_at(*o, t) for o in others]) for t in times])
            advantages[i] = returns - baseline
    # unit scale per load, over the decisions that mattered
    for load in {t.task.load for t in trajectories}:
        members = [i for i, t in enumerate(trajectories) if t.task.load == load]
        values = np.concatenate([
            advantages[i][[s.relevant for s in trajectories[i].steps]] for i in members
        ] or [np.zeros(0)])
        if len(values) > 1:
            mean, std = values.mean(), values.std() + 1e-8
            for i in members:
                advantages[i] = (advantages[i] - mean) / std
    return advantages


def ppo_update(model, optimizer, trajectories, config: TrainConfig, rng) -> Dict[str, float]:
    advantages = group_advantages(trajectories, config.tau_s)
    samples = [
        (s, float(a))
        for t, adv in zip(trajectories, advantages)
        for s, a in zip(t.steps, adv)
        if s.relevant
    ]
    if not samples:
        return {}

    def forward(chunk):
        steps = [s for s, _ in chunk]
        batch, levels, teacher = _batch(model, steps)
        dist = model(batch, with_value=False).dist
        log_prob = dist.level_log_probs().gather(-1, levels.unsqueeze(-1)).squeeze(-1)
        return batch, dist, log_prob, teacher

    old = []
    model.eval()
    with torch.no_grad():
        for idx in _minibatches(len(samples), 512, np.random.default_rng(0)):
            chunk = [samples[i] for i in idx]
            _, _, log_prob, _ = forward(chunk)
            old.append((idx, log_prob))
    old_log_prob = [None] * len(samples)
    for idx, lp in old:
        for row, i in enumerate(idx):
            old_log_prob[i] = lp[row]

    model.train()
    stats = defaultdict(list)
    for epoch in range(config.epochs):
        kls = []
        for idx in _minibatches(len(samples), config.minibatch_size, rng):
            chunk = [samples[i] for i in idx]
            batch, dist, log_prob, teacher = forward(chunk)
            mask = batch.waiting_mask
            width = mask.shape[1]
            old_lp = torch.stack([F.pad(old_log_prob[i], (0, width - len(old_log_prob[i]))) for i in idx])
            adv = torch.tensor([a for _, a in chunk], dtype=torch.float32).unsqueeze(1)
            log_ratio = (log_prob - old_lp).clamp(-20, 20)
            ratio = log_ratio.exp()
            surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - config.clip, 1 + config.clip) * adv)
            # mean over each decision's requests, then over decisions
            count = mask.sum(1).clamp(min=1)
            policy_loss = -((surrogate * mask).sum(1) / count).mean()
            entropy = ((dist.entropy() * mask).sum(1) / count).mean()
            loss = policy_loss - config.entropy_coef * entropy
            imitation = torch.zeros(())
            if config.imitation_coef > 0:
                labeled = mask & (teacher >= 0)
                imitation = imitation_loss(dist, teacher, labeled).mean()
                loss = loss + config.imitation_coef * imitation
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.policy_parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                kl = ((((ratio - 1) - log_ratio) * mask).sum(1) / count).mean().item()
                clipped = ((((ratio - 1).abs() > config.clip) & mask).sum() / mask.sum()).item()
            kls.append(kl)
            for k, v in (("policy_loss", policy_loss.item()), ("entropy", entropy.item()),
                         ("approx_kl", kl), ("clip_fraction", clipped),
                         ("grad_norm", float(grad_norm)), ("imitation_loss", imitation.item())):
                stats[k].append(v)
        stats["epochs"] = [epoch + 1]
        if config.target_kl is not None and np.mean(kls) > config.target_kl:
            break
    model.eval()
    out = {k: float(np.mean(v)) for k, v in stats.items()}
    out["num_samples"] = len(samples)
    return out


def pl_log_prob(scores: torch.Tensor, order: torch.Tensor, num: torch.Tensor, admitted: torch.Tensor) -> torch.Tensor:
    """
    [B] log-probability that a Plackett-Luce ranking with logits `scores` [B, N]
    starts with the first `admitted` entries of `order` [B, N] (positions into
    scores), for `num` real entries per row: sum over k < admitted of
    s[o_k] - logsumexp(s[o_k:]).
    """
    n = scores.shape[1]
    position = torch.arange(n, device=scores.device).unsqueeze(0)
    valid = position < num.unsqueeze(1)
    ordered = scores.gather(1, order).masked_fill(~valid, -1e9)
    suffix = torch.flip(torch.logcumsumexp(torch.flip(ordered, [1]), dim=1), [1])
    term = torch.where(valid & (position < admitted.unsqueeze(1)), ordered - suffix, torch.zeros_like(ordered))
    return term.sum(1)


def ppo_update_pl(model, optimizer, trajectories, config: TrainConfig, rng) -> Dict[str, float]:
    advantages = group_advantages(trajectories, config.tau_s)
    samples = [
        (s, float(a))
        for t, adv in zip(trajectories, advantages)
        for s, a in zip(t.steps, adv)
        if s.relevant and s.order is not None
    ]
    if not samples:
        return {}

    def forward(chunk):
        steps = [s for s, _ in chunk]
        batch, _, _ = _batch(model, steps)
        scores = model(batch, with_value=False).dist.scores / config.pl_temperature
        width = scores.shape[1]
        order = torch.zeros(len(steps), width, dtype=torch.long)
        for b, s in enumerate(steps):
            order[b, : len(s.order)] = torch.from_numpy(s.order)
        num = torch.tensor([len(s.order) for s in steps])
        admitted = torch.tensor([s.num_admitted for s in steps])
        return pl_log_prob(scores, order, num, admitted)

    old = torch.zeros(len(samples))
    model.eval()
    with torch.no_grad():
        for idx in _minibatches(len(samples), 512, np.random.default_rng(0)):
            old[torch.from_numpy(idx)] = forward([samples[i] for i in idx])

    model.train()
    stats = defaultdict(list)
    for epoch in range(config.epochs):
        kls = []
        for idx in _minibatches(len(samples), config.minibatch_size, rng):
            chunk = [samples[i] for i in idx]
            log_prob = forward(chunk)
            adv = torch.tensor([a for _, a in chunk], dtype=torch.float32)
            log_ratio = (log_prob - old[torch.from_numpy(idx)]).clamp(-20, 20)
            ratio = log_ratio.exp()
            surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - config.clip, 1 + config.clip) * adv)
            loss = -surrogate.mean()
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.policy_parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                kl = ((ratio - 1) - log_ratio).mean().item()
                clipped = ((ratio - 1).abs() > config.clip).float().mean().item()
            kls.append(kl)
            for k, v in (("policy_loss", loss.item()), ("approx_kl", kl),
                         ("clip_fraction", clipped), ("grad_norm", float(grad_norm))):
                stats[k].append(v)
        stats["epochs"] = [epoch + 1]
        if config.target_kl is not None and np.mean(kls) > config.target_kl:
            break
    model.eval()
    out = {k: float(np.mean(v)) for k, v in stats.items()}
    out["num_samples"] = len(samples)
    return out


# --- evaluation ----------------------------------------------------------------

EVAL_METRICS = ("mean_session_serving", "p99_session_serving", "mean_ttft", "p99_ttft", "cached_prefill_fraction")


def _vs_fcfs(results: Dict[Tuple[str, int, int], Dict[str, float]], kind: str, loads, seeds) -> Dict[str, float]:
    out = {}
    for load in loads:
        for m in EVAL_METRICS:
            if m == "cached_prefill_fraction":
                out[f"s{load}/{m}"] = float(np.mean([results[(kind, load, s)][m] for s in seeds]))
                continue
            change = [results[(kind, load, s)][m] / results[("fcfs", load, s)][m] - 1 for s in seeds]
            out[f"s{load}/{m}"] = 100 * float(np.mean(change))
    out["mean_session_serving"] = float(np.mean([out[f"s{l}/mean_session_serving"] for l in loads]))
    out["p99_ttft"] = float(np.mean([out[f"s{l}/p99_ttft"] for l in loads]))
    return out


def evaluate(pool, model, config: TrainConfig, baselines) -> Dict[str, float]:
    tasks = [Task(l, s, "policy", decoding="rank") for l in config.loads for s in config.eval_seeds]
    results = dict(baselines)
    for t in pool.run(model.state_dict(), tasks):
        results[("policy", t.task.load, t.task.workload_seed)] = t.summary
    return _vs_fcfs(results, "policy", config.loads, config.eval_seeds)


def score(metrics: Dict[str, float], config: TrainConfig) -> float:
    """Lower is better: % session serving change plus a share of the TTFT tail's."""
    return metrics["mean_session_serving"] + 0.1 * metrics["p99_ttft"]


# --- main ----------------------------------------------------------------------


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        with open(os.path.join(config.output_dir, "config.json"), "w") as f:
            json.dump(dataclasses.asdict(config), f, indent=2)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.model = SchedulerModel(config.model_config()).eval()
        if config.init_checkpoint:
            self.model.load_state_dict(torch.load(config.init_checkpoint)["model"])
        self.pool = Pool(config)
        self.next_seed = TRAIN_SEED_BASE
        self.best = float("inf")
        self.metrics_file = open(os.path.join(config.output_dir, "metrics.jsonl"), "a")
        self.start = time.time()

    def log(self, record: Dict) -> None:
        record = {"elapsed_s": round(time.time() - self.start, 1), **record}
        self.metrics_file.write(json.dumps(record) + "\n")
        self.metrics_file.flush()
        print(json.dumps(record), flush=True)

    def seeds(self, n: int) -> List[int]:
        seeds = list(range(self.next_seed, self.next_seed + n))
        self.next_seed += n
        return seeds

    def save(self, name: str) -> None:
        torch.save(
            {"model": self.model.state_dict(), "model_config": dataclasses.asdict(self.model.config),
             "train_config": dataclasses.asdict(self.config)},
            os.path.join(self.config.output_dir, name),
        )

    def eval_and_save(self, phase: str, iteration: int) -> None:
        metrics = evaluate(self.pool, self.model, self.config, self.baselines)
        self.save("latest.pt")
        s = score(metrics, self.config)
        if s < self.best:
            self.best = s
            self.save("best.pt")
        self.log({"phase": phase, "iteration": iteration, "eval": metrics, "score": s, "best": self.best})

    def run(self) -> None:
        c = self.config
        # FCFS and the teacher on the eval seeds, once
        cache = os.path.join(c.output_dir, "baselines.json")
        if os.path.exists(cache):
            with open(cache) as f:
                self.baselines = {tuple(json.loads(k)): v for k, v in json.load(f).items()}
        else:
            tasks = [Task(l, s, k) for k in ("fcfs", c.teacher) for l in c.loads for s in c.eval_seeds]
            self.baselines = {(t.task.kind, t.task.load, t.task.workload_seed): t.summary
                              for t in self.pool.run(None, tasks)}
            with open(cache, "w") as f:
                json.dump({json.dumps(list(k)): v for k, v in self.baselines.items()}, f)
        self.log({"phase": "baselines", "teacher": _vs_fcfs(self.baselines, c.teacher, c.loads, c.eval_seeds)})

        if c.imitation_iterations > 0 and not c.init_checkpoint:
            optimizer = torch.optim.Adam(self.model.policy_parameters(), lr=c.imitation_lr)
            dataset: List[Step] = []
            beta = 1.0
            for it in range(c.imitation_iterations + 1):
                n = c.imitation_initial_seeds if it == 0 else c.imitation_seeds
                tasks = [Task(l, s, "policy", agent_seed=s, decoding="rank", teacher=True, beta=beta)
                         for l in c.loads for s in self.seeds(n)]
                for t in self.pool.run(self.model.state_dict(), tasks):
                    dataset.extend(s for s in t.steps if s.relevant)
                stats = train_imitation(self.model, optimizer, dataset, c, self.rng)
                self.log({"phase": "imitation", "iteration": it, "beta": beta,
                          "dataset": len(dataset), **stats})
                beta = c.beta_start if it == 0 else beta * c.beta_decay
            self.eval_and_save("imitation", c.imitation_iterations)
        else:
            self.eval_and_save("init", 0)

        optimizer = torch.optim.Adam(self.model.policy_parameters(), lr=c.learning_rate)
        for it in range(1, c.ppo_iterations + 1):
            tasks = [Task(l, s, "policy", agent_seed=1_000_000 * g + s, teacher=c.imitation_coef > 0,
                          decoding="pl" if c.method == "pl" else "sample")
                     for l in c.loads for s in self.seeds(c.seeds_per_iteration)
                     for g in range(c.group_size)]
            trajectories = self.pool.run(self.model.state_dict(), tasks)
            update = ppo_update_pl if c.method == "pl" else ppo_update
            stats = update(self.model, optimizer, trajectories, c, self.rng)
            serving = defaultdict(list)
            for t in trajectories:
                serving[t.task.load].append(t.summary["mean_session_serving"])
            self.log({"phase": "ppo", "iteration": it, **stats,
                      "train_mean_session_serving": {l: float(np.mean(v)) for l, v in serving.items()}})
            if it % c.eval_every == 0 or it == c.ppo_iterations:
                self.eval_and_save("ppo", it)
        self.pool.close()


def _parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="TrainConfig overrides, values as JSON (e.g. loads=[16,24])")
    args = parser.parse_args()
    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        value = _parse_value(value)
        overrides[key] = tuple(value) if isinstance(value, list) else value
    Trainer(TrainConfig(output_dir=args.output_dir, **overrides)).run()


if __name__ == "__main__":
    main()
