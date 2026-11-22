"""
Where the TTFT tail comes from: per-request queueing (arrival -> first
admission) and service (first admission -> first token: chunked prefill,
preemption, recompute), restarts and prompt/cache sizes, for baselines and
checkpoints on the frontier_concurrency eval workloads:

    python scripts/rl/ttft_tail_anatomy.py --config configs/rl/frontier_concurrency_long.yaml \
        --output rl_runs/x/requests.csv --seeds 1000000 ... --workers 8 \
        --policies fcfs sjf_prefill_tokens_age200 ckpt:rl_runs/.../best.pt

Writes one row per scored request (workload, seed, policy, ttft, queue,
service, restarts, prompt, cached, decode, session position).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from frontier.rl.preload import pool_context  # noqa: E402
from frontier.rl.reward import RewardConfig  # noqa: E402
from frontier.rl.train import load_config  # noqa: E402

_STATE = {}


def _init(config_path, output_dir):
    import torch

    torch.set_num_threads(1)
    config = load_config(config_path, [])
    _STATE.update(
        workloads={w.name: w for w in config.eval_workload_configs()},
        observation=config.observation_config(),
        levels=config.num_priority_levels,
        output_dir=output_dir,
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
    workload, seed, policies = task
    s = _STATE
    if workload not in s["envs"]:
        s["envs"][workload] = s["workloads"][workload].make_env(
            RewardConfig(), s["observation"], s["levels"],
            os.path.join(s["output_dir"], "sim"),
        )
    env = s["envs"][workload]
    rows = []
    for policy in policies:
        episode = env.run_episode(_agent(policy, seed), seed, reward_config=RewardConfig())
        admitted_at = {}
        for t in episode.transitions:
            ids = t.obs.request_ids
            for i in np.flatnonzero(t.admitted):
                admitted_at.setdefault(int(ids[i]), t.time)
        position = {}
        for r in sorted(episode.requests, key=lambda r: r.arrived_at):
            if r.session_id is not None:
                position[r.id] = position.get(("s", r.session_id), 0)
                position[("s", r.session_id)] = position[r.id] + 1
        for r in episode.requests:
            if not r.completed:
                continue
            first = admitted_at.get(r.id, np.nan)
            rows.append(
                {
                    "workload": workload,
                    "seed": seed,
                    "policy": policy,
                    "ttft": r.prefill_completed_at - r.arrived_at,
                    "queue": first - r.arrived_at,
                    "service": r.prefill_completed_at - first,
                    "e2e": r.completed_at - r.arrived_at,
                    "restarts": r.num_restarts,
                    "prompt": r.num_prefill_tokens,
                    "cached": r.num_cached_prefill_tokens,
                    "decode": r.num_decode_tokens,
                    "session_position": position.get(r.id, -1),
                    "session": -1 if r.session_id is None else r.session_id,
                    "arrived": r.arrived_at,
                }
            )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--policies", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)
    config = load_config(args.config, [])
    # one task per (workload, seed, policy) so many policies still spread
    # over every worker; rows are appended, and finished tasks are skipped on
    # a rerun
    done = set()
    if os.path.exists(args.output):
        old = pd.read_csv(args.output, usecols=["workload", "seed", "policy"])
        done = set(map(tuple, old.drop_duplicates().itertuples(index=False)))
    tasks = [
        (w.name, s, (p,))
        for p in args.policies for w in config.eval_workload_configs() for s in args.seeds
        if (w.name, s, p) not in done
    ]
    if not tasks:
        return
    ctx = pool_context(config.eval_workload_configs(), os.path.join(out_dir, "preload"))
    with ctx.Pool(min(args.workers, len(tasks)), _init, (args.config, out_dir)) as pool:
        for i, result in enumerate(pool.imap_unordered(_run, tasks)):
            header = not os.path.exists(args.output)
            pd.DataFrame(result).to_csv(args.output, mode="a", header=header, index=False)
            print(f"done {i + 1}/{len(tasks)}", flush=True)

if __name__ == "__main__":
    main()
