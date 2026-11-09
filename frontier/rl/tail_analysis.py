"""
Who waits? Per-call time to first token under FCFS and a rule, by call size,
on agent sessions with a shared system prompt:

    python -m frontier.rl.tail_analysis --output-dir rl_runs/tail_analysis

Writes requests.csv (one row per call) and summary.txt: for each load and
policy, calls bucketed by prompt size and by uncached tokens (the prefill the
call needed when it ran): count, mean and p99 TTFT, and each bucket's share of
the slowest 1% of calls.
"""

import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

CALLS = "rl_runs/agent_traces/tracelab_calls.csv"
BUCKETS = [0, 2_000, 8_000, 32_000, 10**9]
LABELS = ["<2k", "2-8k", "8-32k", ">32k"]


def run_task(args):
    num_sessions, seed, policies = args
    from frontier.rl.agent_trace import DEFAULT_SHARED_PREFIX, SessionWorkload
    from frontier.rl.agents import baseline_agent
    from frontier.rl.env import FrontierSchedulingEnv, a100_llama31_8b

    workload = SessionWorkload(
        CALLS, num_sessions=num_sessions, calls_per_session=20,
        session_rate=num_sessions / 60.0, shared_prefix_tokens=DEFAULT_SHARED_PREFIX,
    )
    env = FrontierSchedulingEnv(a100_llama31_8b(
        name=f"tail_{num_sessions}", sessions=workload,
        enable_prefix_caching=True, prediction_rounding_bits=5,
    ))
    rows = []
    for policy in policies:
        episode = env.run_episode(baseline_agent(policy, seed), seed)
        order = {}
        for r in sorted(episode.requests, key=lambda r: (r.session_id, r.arrived_at)):
            order[r.id] = sum(1 for k in order.values() if k[0] == r.session_id)
            order[r.id] = (r.session_id, order[r.id])
        for r in episode.requests:
            if r.first_token_at is None:
                continue
            rows.append({
                "load": num_sessions, "seed": seed, "policy": policy, "session": r.session_id,
                "call_index": order[r.id][1], "prompt": r.num_prefill_tokens,
                "uncached": r.num_prefill_tokens - r.num_cached_prefill_tokens,
                "output": r.num_decode_tokens, "ttft": r.first_token_at - r.arrived_at,
                "e2e": r.completed_at - r.arrived_at if r.completed_at is not None else np.nan,
            })
    print(f"done load={num_sessions} seed={seed}", flush=True)
    return rows


def summarize(table: pd.DataFrame) -> str:
    lines = []
    for load, by_load in table.groupby("load"):
        for view in ("prompt", "uncached"):
            lines.append(f"== {load} sessions, calls bucketed by {view} tokens")
            lines.append(f"{'policy':14s}{'bucket':>8s}{'calls':>8s}{'mean ttft':>11s}{'p99 ttft':>10s}{'share of slowest 1%':>21s}")
            for policy, rows in by_load.groupby("policy", sort=False):
                cut = rows.ttft.quantile(0.99)
                bucket = pd.cut(rows[view], BUCKETS, labels=LABELS, right=False)
                for label in LABELS:
                    sub = rows[bucket == label]
                    if not len(sub):
                        continue
                    share = (sub.ttft >= cut).sum() / max(1, (rows.ttft >= cut).sum())
                    lines.append(f"{policy:14s}{label:>8s}{len(sub):>8d}{sub.ttft.mean():>10.1f}s{sub.ttft.quantile(0.99):>9.1f}s{100 * share:>20.0f}%")
            lines.append("")
        first = by_load[by_load.call_index == 0]
        lines.append(f"== {load} sessions, first call of each session (cold start)")
        for policy, rows in first.groupby("policy", sort=False):
            lines.append(f"{policy:14s} mean ttft {rows.ttft.mean():.1f}s  p99 {rows.ttft.quantile(0.99):.1f}s  ({len(rows)} calls)")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--loads", type=int, nargs="*", default=[16, 24])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--policies", nargs="*", default=["fcfs", "sjf_uncached"])
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tasks = [(n, s, args.policies) for n in args.loads for s in args.seeds]
    with mp.get_context("spawn").Pool(args.num_workers) as pool:
        rows = [r for task_rows in pool.imap_unordered(run_task, tasks) for r in task_rows]
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.output_dir, "requests.csv"), index=False)
    summary = summarize(table)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
