"""
How long does RL training cost on this env? Times one-process episodes with
the real (profiled) predictor:

    python -m frontier.rl.benchmark --num-requests 300 --qps 10 --episodes 3

and reports setup time (Simulator construction, predictor loading), event-loop
time, decisions per episode, and per-decision cost with a stand-in policy
network the size of the Vidur PPO model (d_model 64, 2 transformer layers, a
forward pass per decision). Then extrapolates to a training run of
iterations x seeds_per_iteration x group_size episodes.
"""

import argparse
import time

import numpy as np

from frontier.rl.agents import baseline_agent
from frontier.rl.env import BaseAgent, FrontierSchedulingEnv, WorkloadSpec, _working_dir, FRONTIER_ROOT


class StandInPolicy(BaseAgent):
    """A transformer of the Vidur policy's size; output ignored (FCFS order)."""

    def __init__(self, num_features: int, d_model: int = 64, num_layers: int = 2):
        import torch

        torch.set_num_threads(1)
        self._torch = torch
        layer = torch.nn.TransformerEncoderLayer(d_model, 4, 4 * d_model, batch_first=True)
        self._net = torch.nn.Sequential(
            torch.nn.Linear(num_features, d_model),
            torch.nn.TransformerEncoder(layer, num_layers),
            torch.nn.Linear(d_model, 1),
        ).eval()
        self.forward_s = 0.0

    def act(self, obs):
        start = time.perf_counter()
        x = self._torch.as_tensor(np.log1p(np.abs(obs.waiting[:128])), dtype=self._torch.float32)
        with self._torch.no_grad():
            self._net(x[None])
        self.forward_s += time.perf_counter() - start
        return np.zeros(obs.num_requests, dtype=np.int64), {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--num-requests", type=int, default=300)
    parser.add_argument("--qps", type=float, default=10.0)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    spec = WorkloadSpec(
        name="bench", num_requests=args.num_requests, qps=args.qps,
        length_trace="data/processed_traces/splitwise_conv.csv", num_blocks=args.num_blocks,
    )
    env = FrontierSchedulingEnv(spec)

    with _working_dir(FRONTIER_ROOT):
        start = time.perf_counter()
        env.make_simulator(0)
        setup_s = time.perf_counter() - start

    rows = []
    for name in ["fcfs", "policy"]:
        for seed in range(args.episodes):
            agent = baseline_agent("fcfs") if name == "fcfs" else StandInPolicy(10)
            start = time.perf_counter()
            episode = env.run_episode(agent, seed)
            total_s = time.perf_counter() - start
            forward_s = getattr(agent, "forward_s", 0.0)
            decisions = episode.summary["num_decisions"] + episode.summary["num_dropped_decisions"]
            rows.append((name, seed, total_s, forward_s, decisions,
                         episode.summary["num_decisions"], episode.end_time))
            print(f"{name:6s} seed {seed}: {total_s:6.2f}s wall ({forward_s:5.2f}s in the "
                  f"network), {decisions} decisions ({episode.summary['num_decisions']} kept), "
                  f"{episode.end_time:6.1f}s simulated, mean e2e "
                  f"{episode.summary['mean_e2e']:.2f}s", flush=True)

    policy = [r for r in rows if r[0] == "policy"]
    episode_s = float(np.mean([r[2] for r in policy]))
    print(f"\nsetup (Simulator + predictor load): {setup_s:.2f}s per episode")
    print(f"policy episode: {episode_s:.2f}s, of which network "
          f"{np.mean([r[3] for r in policy]):.2f}s over {np.mean([r[4] for r in policy]):.0f} decisions")
    # the Vidur PPO run: 200 iterations x 8 seeds x 4 rollouts, plus an FCFS
    # reference per (seed, workload) and evaluation every 10 iterations
    for iterations, seeds, group in ((100, 8, 4), (200, 8, 4)):
        episodes = iterations * seeds * (group + 1)
        hours = episodes * episode_s / args.workers / 3600
        print(f"{iterations} iterations x {seeds} seeds x {group} rollouts (+FCFS reference) = "
              f"{episodes} episodes: ~{hours:.1f} h on {args.workers} workers (rollouts only)")


if __name__ == "__main__":
    main()
