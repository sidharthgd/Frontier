"""
Baselines on the frontier_concurrency workloads, rescored offline under a grid
of ttft_hinge weights (the reward otherwise as in the config: e2e_normalized +
w * ttft_hinge, one FCFS-q95 TTFT threshold per episode):

    python scripts/rl/concurrency_baseline_sweep.py \
        --config configs/rl/frontier_concurrency.yaml --output-dir rl_runs/x \
        --eval-seeds 1000000 ... --train-seeds 0 ... --workers 64

Each task runs every baseline once on one (workload, seed); improvement over
FCFS is 1 - cost / fcfs_cost, as frontier.rl.train logs it. A baseline given as
ckpt:<path> is that checkpoint's policy with argmax decoding (as in the
trainer's evals); --reference-cache-dir reuses the trainers' cached FCFS runs. Writes
<output-dir>/episodes.csv (one row per workload, seed, baseline: unweighted
reward components and summary metrics) and prints the weight grid.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from frontier.rl.preload import pool_context  # noqa: E402
from frontier.rl.reward import episode_cost_components  # noqa: E402
from frontier.rl.train import load_config  # noqa: E402

_STATE = {}


def _init(config_path, output_dir, reference_cache_dir=None):
    import torch

    # one thread per worker (as frontier/rl/rollout.py): torch's default is
    # every core, and policy inference in N workers would start N x cores threads
    torch.set_num_threads(1)
    config = load_config(config_path, [])
    reward = config.reward_config()
    workloads = {
        w.name: w
        for w in config.workload_configs() + config.eval_workload_configs()
    }
    _STATE.update(
        workloads=workloads,
        reward=reward,
        observation=config.observation_config(),
        levels=config.num_priority_levels,
        output_dir=output_dir,
        reference_cache_dir=reference_cache_dir,
        envs={},
        models={},
    )


def _agent(name, seed):
    from frontier.rl.agents import baseline_agent
    from frontier.rl.torch_agent import TorchAgent
    from frontier.rl.train import load_checkpoint

    if not name.startswith("ckpt:"):
        return baseline_agent(name, seed=seed)
    path = name[len("ckpt:"):]
    if path not in _STATE["models"]:
        _STATE["models"][path] = load_checkpoint(path)["model_object"]
    return TorchAgent(_STATE["models"][path], seed, "argmax")


def _run(task):
    workload, seed, baselines = task
    s = _STATE
    if workload not in s["envs"]:
        s["envs"][workload] = s["workloads"][workload].make_env(
            s["reward"], s["observation"], s["levels"],
            os.path.join(s["output_dir"], "sim"),
        )
        if s["reference_cache_dir"]:
            from frontier.rl.rollout import reference_dir

            s["envs"][workload].reference_cache_dir = reference_dir(
                s["reference_cache_dir"], s["workloads"][workload], s["levels"]
            )
    env = s["envs"][workload]
    thresholds = env.reward_context(s["reward"], seed).request_thresholds
    rows = []
    for name in baselines:
        start = time.time()
        if name.startswith("ckpt:"):
            from frontier.rl.reward import RewardConfig

            episode = env.run_episode(_agent(name, seed), seed, reward_config=RewardConfig())
        else:
            episode = env.reference_episode(name, seed)
        components = episode_cost_components(episode, s["reward"], thresholds)
        rows.append(
            {
                "workload": workload,
                "seed": seed,
                "baseline": name,
                "wall_s": time.time() - start,
                **{f"c_{k}": v for k, v in components.items()},
                **episode.summary,
            }
        )
    return rows


def weight_grid(df: pd.DataFrame, weights) -> pd.DataFrame:
    out = []
    for w in weights:
        cost = df["c_e2e_normalized"] + w * df["c_ttft_hinge"]
        d = df.assign(cost=cost)
        fcfs = d[d.baseline == "fcfs"].set_index(["workload", "seed"])["cost"]
        d = d.join(fcfs.rename("fcfs_cost"), on=["workload", "seed"])
        d["improvement"] = 1.0 - d["cost"] / d["fcfs_cost"]
        g = d.groupby(["workload", "baseline"])["improvement"].mean().unstack(0)
        g.insert(0, "w", w)
        out.append(g)
    return pd.concat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--baselines", nargs="+", default=[
        "fcfs", "sjf_prefill_tokens", "sjf_prefill_tokens_age100",
        "sjf_prefill_tokens_age200", "sjf_prefill_tokens_age400",
        "sjf_prefill_tokens_age800"])
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[])
    p.add_argument("--train-seeds", type=int, nargs="+", default=[])
    p.add_argument("--weights", type=float, nargs="+",
                   default=[0, 0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1])
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--reference-cache-dir", default=None)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config, [])
    tasks = [(w.name, s, args.baselines)
             for w in config.eval_workload_configs() for s in args.eval_seeds]
    tasks += [(w.name, s, args.baselines)
              for w in config.workload_configs() for s in args.train_seeds]
    # resume: keep the finished (workload, seed) tasks of an earlier run
    rows = []
    path = os.path.join(args.output_dir, "episodes.csv")
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old.baseline.isin(args.baselines)]
        complete = old.groupby(["workload", "seed"]).baseline.nunique() == len(args.baselines)
        done = set(complete[complete].index)
        rows = [r for r in old.to_dict("records") if (r["workload"], r["seed"]) in done]
        tasks = [t for t in tasks if (t[0], t[1]) not in done]
        print(f"resuming: {len(done)} tasks already done", flush=True)
    print(f"{len(tasks)} tasks x {len(args.baselines)} baselines", flush=True)

    # workers fork from one process holding the predictors (~7 GB each)
    ctx = pool_context(config.workload_configs() + config.eval_workload_configs(),
                       os.path.join(args.output_dir, "preload"))
    with ctx.Pool(min(args.workers, len(tasks)), _init,
                  (args.config, args.output_dir, args.reference_cache_dir)) as pool:
        for i, result in enumerate(pool.imap_unordered(_run, tasks)):
            rows += result
            print(f"done {i + 1}/{len(tasks)}: {result[0]['workload']} "
                  f"seed {result[0]['seed']}", flush=True)
            pd.DataFrame(rows).to_csv(
                os.path.join(args.output_dir, "episodes.csv"), index=False)
    df = pd.DataFrame(rows)

    metrics = ["mean_e2e", "p99_e2e", "mean_ttft", "p99_ttft", "num_restarts",
               "prefix_hit_rate"]
    metrics = [m for m in metrics if m in df]
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", 500,
                           "display.float_format", "{:.3f}".format):
        print("\nseed means")
        print(df.groupby(["workload", "baseline"])[metrics].mean())
        fcfs = df[df.baseline == "fcfs"].set_index(["workload", "seed"])[metrics]
        rel = df.join(fcfs, on=["workload", "seed"], rsuffix="_fcfs")
        for m in metrics:
            rel[m] = rel[m] / rel[f"{m}_fcfs"]
        print("\nrelative to FCFS (seed mean of ratios)")
        print(rel.groupby(["workload", "baseline"])[metrics].mean())
        grid = weight_grid(df, args.weights)
        eval_names = [w.name for w in config.eval_workload_configs()]
        grid["eval_mean"] = grid[[c for c in eval_names if c in grid]].mean(axis=1)
        grid["eval_worst"] = grid[[c for c in eval_names if c in grid]].min(axis=1)
        grid.to_csv(os.path.join(args.output_dir, "weight_grid.csv"))
        print("\nimprovement over FCFS by ttft_hinge weight")
        print(grid)


if __name__ == "__main__":
    main()
