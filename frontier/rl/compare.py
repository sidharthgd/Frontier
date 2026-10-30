"""
Headroom check: heuristic baselines on the same seeds, paired against FCFS
(ported from vidur/rl/compare.py):

    python -m frontier.rl.compare --output-dir rl_runs/compare --num-workers 8

Writes episodes.csv (one row per workload, seed, policy) and summary.txt (mean
% change vs FCFS with paired 95% intervals). The question it answers before
any training: does the oracle land well below the prompt-SJF aging curve? If
not, reordering has little room on these workloads.
"""

import argparse
import multiprocessing as mp
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from frontier.rl.env import WorkloadSpec

TRACES = {
    # name: (length trace, FCFS capacity in qps on H200 + Llama-2-7B, from
    # rl_runs/capacity.txt: batch-cap bound for conv/sharegpt, prefill bound for code)
    "conv": ("data/processed_traces/splitwise_conv.csv", 27.0),
    "code": ("data/processed_traces/splitwise_code.csv", 22.0),
    "sharegpt": ("data/processed_traces/sharegpt_llama3_tokenizer.csv", 27.0),
}


def _grid():
    out = {}
    for trace, (path, capacity) in TRACES.items():
        base = dict(length_trace=path, prediction_rounding_bits=5)
        for load in (0.7, 0.9, 1.1):
            out[f"{trace}_load{int(load * 100)}"] = dict(base, qps=round(capacity * load, 1))
        out[f"{trace}_load90_mem4000"] = dict(base, qps=round(capacity * 0.9, 1), num_blocks=4000)
        out[f"{trace}_load90_bursty"] = dict(base, qps=round(capacity * 0.9, 1), arrivals="gamma", cv=3.0)
    return out


WORKLOADS = _grid()
AGING = (0, 200, 400, 800, 1600)
DEFAULT_POLICIES = (
    ["fcfs"]
    + [f"sjf_prefill_age{a}" for a in AGING]
    + ["sjf_total", "sjf_total_age400"]
)
METRICS = ("mean_e2e", "p99_e2e", "mean_ttft", "p99_ttft", "mean_tbt", "p99_tbt")


def run_task(args) -> List[Dict]:
    name, kwargs, num_requests, seed, policies = args
    from frontier.rl.agents import baseline_agent
    from frontier.rl.env import FrontierSchedulingEnv

    env = FrontierSchedulingEnv(WorkloadSpec(name=name, num_requests=num_requests, **kwargs))
    rows = []
    for policy in policies:
        episode = env.run_episode(baseline_agent(policy, seed), seed)
        rows.append({"workload": name, "seed": seed, "policy": policy, **episode.summary})
    print(f"done {name} seed {seed}", flush=True)
    return rows


def summarize(table: pd.DataFrame) -> str:
    from scipy import stats

    lines = ["% change vs fcfs, mean over seeds ± paired 95% CI (negative = faster)", ""]
    for workload, group in table.groupby("workload", sort=False):
        ref = group[group.policy == "fcfs"].set_index("seed")
        lines.append(f"== {workload} (n={len(ref)} seeds)")
        lines.append(f"{'policy':22s}" + "".join(f"{m:>17s}" for m in METRICS))
        for policy, rows in group.groupby("policy", sort=False):
            rows = rows.set_index("seed")
            cells = ""
            for m in METRICS:
                ratio = 100 * (rows[m] / ref.loc[rows.index, m] - 1)
                half = (
                    stats.t.ppf(0.975, len(ratio) - 1) * ratio.std(ddof=1) / np.sqrt(len(ratio))
                    if len(ratio) > 1 else float("nan")
                )
                cells += f"{ratio.mean():+8.1f} ±{half:5.1f}  "
            lines.append(f"{policy:22s}{cells}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policies", nargs="*", default=list(DEFAULT_POLICIES))
    parser.add_argument("--workloads", nargs="*", default=list(WORKLOADS))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(range(2000000, 2000006)))
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() - 2))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = [
        (w, WORKLOADS[w], args.num_requests, s, args.policies)
        for w in args.workloads for s in args.seeds
    ]
    with mp.get_context("spawn").Pool(args.num_workers) as pool:
        rows = [r for task_rows in pool.imap_unordered(run_task, tasks) for r in task_rows]
    table = pd.DataFrame(rows).sort_values(["workload", "seed"], kind="stable")
    table.to_csv(os.path.join(args.output_dir, "episodes.csv"), index=False)
    summary = summarize(table)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
